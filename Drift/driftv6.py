import asyncio
import datetime
import logging
import os
import pickle
import sys
import uuid
import base64
import httpx
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Response, Header
from sklearn.neighbors import NearestNeighbors
import redis.asyncio as redis
from datetime import timezone

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ood-detector")
RETRY_DELAY = 0.1 # 100 ms
RETRY=10
REDIS_HOST = os.getenv("REDIS_HOST", "localhost") #Redis is into the ood detector pos, sidcar istio service mesh, redis and ood containers share the same local host/Pod
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_IMAGE_TTL = int(os.getenv("REDIS_IMAGE_TTL", "600")) #10min
REDIS_OOD_EXTENDED_TTL = int(os.getenv("REDIS_OOD_EXTENDED_TTL", "3600")) #i hour 
# every batch ood deteted the store and forward component take all the batch redis items and send to the data lake
OOD_FORWARD_ENDPOINT = os.getenv("OOD_FORWARD_ENDPOINT", "")
OOD_FORWARD_BATCH_SIZE = int(os.getenv("OOD_FORWARD_BATCH_SIZE", "10"))
#because the system can scale to zero and it is statefull i have to store the ood detector state before the system scale into a pvc-pv persistent storage
STATE_KEY = "detector_state"
FALLBACK_FILE = "/data/queue/detector_state.pkl" #state snapshot
redis_client: Optional[redis.Redis] = None
ood_buffer = [] #track the ood batch
##############################################################  DEVELOPED OOD DRIFT DETECTOR CLASS ###############################################################################################
class RealTimeOODDetector:
    def __init__( self,  cent: np.ndarray,  inv_cov: np.ndarray,  win_sz: int,  batch_sz: int,  init_th: Optional[float] = None,  smooth: float = 0.9,  min_s: int = 50,  perc: float = 90.0, 
        max_perc: float = 95.0,  safe_th: Optional[float] = None,  max_drop: int = 5,  max_up: int = 5,  med_win: int = 5,  smooth_safe: bool = False ):
        if inv_cov is None:
            raise ValueError("inv_cov required")
        if safe_th is None and smooth_safe:
            raise ValueError("smooth_safe needs safe_th")
        self.cent = cent
        self.inv_cov = inv_cov
        self.knn = NearestNeighbors(n_neighbors=1).fit(cent)
        self.dist_buf = deque(maxlen=win_sz)
        self.batch_cnt = 0
        self.th = init_th
        self.safe_th = safe_th
        self.batch_sz = batch_sz
        self.smooth = smooth
        self.min_s = min_s
        self.perc = perc
        self.max_perc = max_perc
        self.max_drop = max_drop
        self.max_up = max_up
        self.drop_cnt = self.up_cnt = 0
        self.med_win = med_win
        self.batch_percs = deque(maxlen=med_win)
        self.smooth_safe = smooth_safe
        self.th_flow = deque(maxlen=max(max_drop, max_up) + 1)
        self.cnt = 0

    def _raw_th(self, dists: np.ndarray) -> Optional[float]:
        return None if len(dists) < self.min_s else float(np.percentile(self.dist_buf, self.perc))

    def _fallback(self, dists: np.ndarray) -> float:
        med = float(np.median(dists))
        return float(med + 2.0 * np.median(np.abs(dists - med)))

    def _update_th(self) -> None:
        if len(self.dist_buf) < self.min_s:
            return
        buf = np.asarray(self.dist_buf)
        thr = self._raw_th(buf) or self._fallback(buf)
        self.batch_percs.append(thr)
        thr_med = float(np.median(self.batch_percs)) if len(self.batch_percs) >= 2 else thr
        cand = min(thr_med, float(np.percentile(buf, self.max_perc)))

        if self.th is None:
            self.th = cand
        else:
            if cand > self.th:
                self.up_cnt += 1
                self.drop_cnt = 0
            elif cand < self.th:
                self.drop_cnt += 1
                self.up_cnt = 0
            else:
                self.up_cnt = self.drop_cnt = 0

            target, reset = cand, False

            if self.safe_th is not None and (self.drop_cnt >= self.max_drop or self.up_cnt >= self.max_up):
                target = self.safe_th
                self.drop_cnt = self.up_cnt = 0
                if not self.smooth_safe:
                    reset = True
            elif self.drop_cnt >= self.max_drop and self.safe_th is None:
                idx = self.drop_cnt + 1
                if len(self.th_flow) >= idx:
                    target = self.th_flow[-idx]
                self.drop_cnt = self.up_cnt = 0
            elif self.up_cnt >= self.max_up and self.safe_th is None:
                idx = self.up_cnt + 1
                if len(self.th_flow) >= idx:
                    target = self.th_flow[-idx]
                self.up_cnt = self.drop_cnt = 0

            self.th = target if reset else self.smooth * self.th + (1.0 - self.smooth) * target

        self.th_flow.append(cand)

    def process(self, emb: np.ndarray) -> tuple[float, bool, Optional[float]]:
        emb = np.asarray(emb).reshape(1, -1)
        idx = self.knn.kneighbors(emb, return_distance=False)[0][0]
        diff = emb.ravel() - self.cent[idx]
        dist = float(np.sqrt(diff @ self.inv_cov @ diff.T))

        self.dist_buf.append(dist)
        self.batch_cnt += 1
        self.cnt += 1

        if self.batch_cnt >= self.batch_sz:
            self.batch_cnt = 0
            self._update_th()

        if self.th is None and len(self.dist_buf) >= self.min_s:
            buf = np.asarray(self.dist_buf)
            thr = self._raw_th(buf) or self._fallback(buf)
            self.th = min(thr, float(np.percentile(buf, self.max_perc)))

        is_ood = self.th is not None and dist > self.th
        return dist, is_ood, self.th

    def get_state(self) -> dict: #get the ood state before the snapshot 
        return { k: getattr(self, k) for k in ( "knn", "cent", "inv_cov", "dist_buf", "batch_cnt", "th", "safe_th",  "batch_sz", "smooth", "min_s", "perc", "max_perc", "max_drop", "max_up",
                "drop_cnt", "up_cnt", "med_win", "batch_percs", "smooth_safe", "th_flow", "cnt") }

    def set_state(self, st: dict) -> None: #upload the snapshot state
        for k, v in st.items():
            if k in ("dist_buf", "batch_percs", "th_flow"):
                maxlen = getattr(self, k).maxlen
                setattr(self, k, deque(v, maxlen=maxlen))
            else:
                setattr(self, k, v)
#######################################################################################################################################################
detector = None
queue = asyncio.Queue(maxsize=5000) #security check
worker = None

def init_detector() -> RealTimeOODDetector:
    try:
        centroids_path = os.getenv("CENTROIDS_PATH", "centroids.npy")
        inv_cov_path = os.getenv("INV_COV_MATRIX_PATH", "inv_cov_matrix.npy")
        cent = np.load(centroids_path)
        inv = np.load(inv_cov_path)
    except FileNotFoundError as e:
        log.error(f"Missing configuration files: {e}")
        raise RuntimeError("Missing centroids or inv_cov matrices") from e

    init_th = float(os.getenv("OOD_INITIAL_THRESHOLD", "0.5"))
    params = {
        "cent": cent, 
        "inv_cov": inv,
        "win_sz": int(os.getenv("OOD_WINDOW_SIZE", "50")),
        "batch_sz": int(os.getenv("OOD_BATCH_SIZE", "5")),
        "init_th": init_th,
        "smooth": float(os.getenv("OOD_SMOOTHING", "0.95")),
        "min_s": int(os.getenv("OOD_MIN_SAMPLES", "20")),
        "perc": float(os.getenv("OOD_PERCENTILE", "95")),
        "max_perc": float(os.getenv("OOD_MAX_PERCENTILE", "99")),
        "safe_th": float(os.getenv("OOD_SAFE_THRESHOLD", str(init_th))),
        "max_drop": int(os.getenv("OOD_MAX_CONSECUTIVE_DROPS", "3")),
        "max_up": int(os.getenv("OOD_MAX_CONSECUTIVE_UPS", "3")),
        "med_win": int(os.getenv("OOD_MEDIAN_WINDOW", "5")),
        "smooth_safe": os.getenv("OOD_SMOOTH_SAFETY_TH", "true").lower() == "true",
    }
    return RealTimeOODDetector(**params)
###################################################################################################################################################
async def save_state(): #save the ood detector snapshot and redis in memory
    if detector is None:
        return
    log.info("Saving state to Redis and fallback file...")
    state = detector.get_state()
    try:
        os.makedirs( os.path.dirname(FALLBACK_FILE), exist_ok=True) #create is not present a dir for the snapshot file into the persistent pvc-pv
        with open(FALLBACK_FILE, "wb") as f:
            pickle.dump(state, f) #Synch write into it 
        log.info(f"  ===== OK ======= State saved to fallback file {FALLBACK_FILE}")
    except Exception as e:
        log.error(f"Failed to write fallback file: {e}")

async def restore_state():
    global detector
    if os.path.exists(FALLBACK_FILE):
        try:
            with open(FALLBACK_FILE, "rb") as f:
                st = pickle.load(f)
            detector = init_detector()
            detector.set_state(st)
            log.info(f" ===== OK ======= State restored from fallback file {FALLBACK_FILE}")
            return
        except Exception as e:
            log.warning(f"Fallback file restore failed: {e}")

    log.info("No state found, creating new detector")
    detector = init_detector()
##############################################################################################################################################################
async def forward_ood_batch():
    global ood_buffer

    if not ood_buffer or not redis_client:
        return

    items_to_forward = list(ood_buffer) #local function copy 
    ood_buffer.clear() #buffer clean up

    payload_batch = []
    redis_keys_to_delete = []

    log.info(f"Preparing to process OOD batch of {len(items_to_forward)} items...")

    for item in items_to_forward: # for each ood items into the buffer snapshot 
        img_redis_key = item["img_redis_key"]
        meta_redis_key = item["meta_redis_key"]
        redis_keys_to_delete.extend( [img_redis_key, meta_redis_key] )
        try:
            img_bytes = await redis_client.get(img_redis_key) #very fast in memory read
            meta_dict_raw = await redis_client.hgetall(meta_redis_key) #very fast in memory read
 
            if not img_bytes or not meta_dict_raw:
                log.warning(f"Incomplete Redis data for {img_redis_key}. Skipping payload assembly.")
                continue
            # Data Lake Formatting -> change this operations based of the data format
            #########################################################################
            #redis give back bynary key-value data after the fatch
            meta_dict = { # utf 8 key and value formattig operation
                k.decode("utf-8") if isinstance(k, bytes) else k: 
                v.decode("utf-8") if isinstance(v, bytes) else v
                for k, v in meta_dict_raw.items()
            }
            payload_batch.append({ # the http data lake operation architecture can be change based on the use case 
                "image_key": item["image_key"],
                "distance": item["distance"],
                "threshold": item["threshold"],
                "detected_at": item["detected_at"],
                "image_b64": base64.b64encode(img_bytes).decode("utf-8"), # this is a http json request body
                "metadata": meta_dict
            })
            ########################################################################
        except Exception as e:
            log.error(f"Error fetching data from Redis for key {img_redis_key}: {e}")
    
    async def cleanup_redis_keys(): #corutine in memory batch clean up function
        if not redis_keys_to_delete:
            return
        try: #single redis transaction that clean up the entire buffer batch (redis in memory very pfast O(1))
            async with redis_client.pipeline(transaction=True) as pipe:
                for k in redis_keys_to_delete:
                    pipe.delete(k)
                await pipe.execute() # asynch wait for the transaction end up
            log.info(f"Deleted {len(redis_keys_to_delete)} keys from Redis.")
        except Exception as e:
            log.error(f"Failed to delete keys from Redis: {e}")

    if not payload_batch or not OOD_FORWARD_ENDPOINT:
        if not payload_batch:
            log.warning("No valid payloads assembled for batch forwarding.")
        else:
            log.info("OOD_FORWARD_ENDPOINT not configured. Skipping HTTP forward.")
        
        await cleanup_redis_keys() #aynch wait for the corutine end up
        return

    # network call the possibile data lake end point 
    try:
        """
        async with httpx.AsyncClient(timeout=0.0) as client:
            res = await client.post(OOD_FORWARD_ENDPOINT, json={"batch": payload_batch})
            res.raise_for_status()
            log.info(f"Successfully forwarded batch of {len(payload_batch)} OOD samples to {OOD_FORWARD_ENDPOINT}")
        """
    except Exception as e:
        log.error(f"Failed to forward OOD batch to endpoint {OOD_FORWARD_ENDPOINT}: {e}. Cleaning up Redis anyway.")
    finally:
        # Guarantee cleanup runs regardless of HTTP success or failure
        await cleanup_redis_keys()
##########################################################################################################################################################
async def wait_for_redis_image(  img_key: str,  meta_key: str,  retries: int = 10):
    for attempt in range(RETRY):
        img = await redis_client.get(img_key)
        meta = await redis_client.hgetall(meta_key)
        if img is not None and meta:
            if attempt > 0:
                log.info( f"Redis object {img_key} became available after {attempt} retries." )
            return img, meta
        await asyncio.sleep(RETRY_DELAY)
    return None, None
async def worker_loop():
    log.info("Worker started")
    try:
        while True:

            item = await queue.get() #even loop get msg from the queue 

            insts = item["instances"]
            eid = item.get("event_id", "unknown")
            image_key = item.get("image_key", "unknown-key")
            log.info(f"Processing {len(insts)} elements from event {eid} (Image Key: {image_key})")
            
            for i, inst in enumerate(insts): #because the inference batching more that one emd is possibile but here we have only onece
                emb = inst
                if emb is None:
                    log.warning("Element extraction failed or empty embedding. Skipping sample.")
                    continue
                
                dist, ood, th = await asyncio.to_thread(detector.process, emb) #cpu boud operation scheduled into another thread

                if image_key != "unknown-key" and redis_client is not None: 
                    #clean_id = image_key.removeprefix("image:") 
                    img_redis_key = f"image:{image_key}"
                    meta_redis_key = f"{img_redis_key}:meta"

                    if ood: # if a ood has been detected
                        log.warning(f"OOD detected with redis key -> {image_key}")
                        try:#check the already stored item
                            img_bytes, meta_dict = await wait_for_redis_image( img_redis_key, meta_redis_key, )
                            if img_bytes is None or not meta_dict:
                                log.error( f"Skipping TTL extension for key {img_redis_key}: " "Image or metadata not available after retries." )
                                continue
                            """"
                            img_bytes = await redis_client.get(img_redis_key) 
                            meta_dict = await redis_client.hgetall(meta_redis_key)
                            
                            if not img_bytes or not meta_dict: #security check 
                                log.error(
                                    f"Skipping TTL extension for key {img_redis_key}: "
                                    f"Image exists? {bool(img_bytes)} | Metadata exists? {bool(meta_dict)}. "
                                    "Allowing initial TTL to expire remaining keys automatically."
                                )
                                continue #skip the item elaboration 
                            """
                            log.info(f"Retrieved OOD image blob ({len(img_bytes)} bytes) for key {img_redis_key}")
                            log.info(f"Retrieved OOD metadata: {meta_dict}")
                            #because the system have received the kantive inference eventing is ok update the ood ttl increasing the life of the component
                            async with redis_client.pipeline(transaction=True) as pipe: # very fast O(1) transaction for the system clean up because redis is a in memory key value store
                                pipe.expire(img_redis_key, REDIS_OOD_EXTENDED_TTL)
                                pipe.expire(meta_redis_key, REDIS_OOD_EXTENDED_TTL)
                                pipe.incr("metrics:redis_success_ttl") # the system  find out the ood image redis key coming from kafka and increare the entry ttl
                                await pipe.execute()
                                
                            log.info(f"TTL extended to {REDIS_OOD_EXTENDED_TTL}s for key: {img_redis_key}")

                            ood_buffer.append({ "image_key": image_key, "img_redis_key": img_redis_key, #add ood into the memory buffer
                                "meta_redis_key": meta_redis_key, "distance": dist, "threshold": th,
                                "detected_at": datetime.datetime.now(timezone.utc).isoformat() })

                            if len(ood_buffer) >= OOD_FORWARD_BATCH_SIZE: #memory limits of the batch has been reach 
                                log.info(f"OOD buffer limit reached ({len(ood_buffer)}/{OOD_FORWARD_BATCH_SIZE}). Triggering forwarder.")
                                await forward_ood_batch() #redcue the data lake  network call of a batch factor
                        except Exception as redis_err:
                            log.error(f"~~~~~  ~~~~ ~~~~~~  ~~~~ ERROR reading/updating Redis for key {image_key}: {redis_err}")

                    else: #no odd, eliminate the item and clean up the system
                        log.info(f"In-Distribution sample detected. Deleting Redis key: {img_redis_key}")
                        try:
                            async with redis_client.pipeline(transaction=True) as pipe: # very fast O(1) transaction for the system clean up because redis is a in memory key value store
                                pipe.delete(img_redis_key)
                                pipe.delete(meta_redis_key)
                                await pipe.execute()
                            log.info(f"Successfully deleted {img_redis_key} and {meta_redis_key} from Redis.")
                        except Exception as redis_err:
                            log.error(f"~~~~~  ~~~~ ~~~~~~  ~~~~ ERROR deleting Redis keys for {image_key}: {redis_err}")
            queue.task_done()
            log.info(f"Finished processing event {eid}")
    except asyncio.CancelledError:
        log.info("Worker cancelled, draining queue")
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break
##################################################################################################################################
async def increment_redis_error():
    if redis_client:
        try:
            await redis_client.incr("metrics:redis_errors") #incr increment
        except Exception:
            pass
async def increment_redis_success():
    if redis_client:
        try:
            await redis_client.incr("metrics:redis_success_ops")
        except Exception:
            pass
async def increment_redis_ttl():
    if redis_client:
        try:
            await redis_client.incr("metrics:redis_success_ttl")
        except Exception:
            pass
##################################################################################################################################
@asynccontextmanager
async def lifespan(app: FastAPI):# manager of the rest api lifecyle
    global detector, worker, redis_client  #se the connections
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False) #set redis client (redis is located as sindcar into the same ood pod)
    log.info(f"Redis client connected to {REDIS_HOST}:{REDIS_PORT}")
    await restore_state() # operation important if the system reboot after a scale to zero
    worker = asyncio.create_task( worker_loop() ) #start the asynch event loop 

    yield # after this marcks start the code executed before the process shutdown (the process receive the scale to zero signal)

    log.info("Shutting down...")
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass
    # if the system scale to zero, before doing so i have to flash all the iteam into the redis in memory buffer !!!! 
    if ood_buffer: #flush
        log.info(f"Flushing remaining {len(ood_buffer)} OOD items during shutdown...")
        await forward_ood_batch()

    await save_state() #save the state of the drift detector 

    try:
        await redis_client.save() #if we execute the flush it is not usefull for the blob images but useful for the metadata log
        log.info("Redis data saved to disk")
    except Exception as e:
        log.warning(f"Redis save failed: {e}")
    await redis_client.close()
    log.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

######################################################################################################################################
@app.post("/") #endpoint for the knative eventing cloud event coming from kafka
async def receive(req: Request):

    try:
        ev = await req.json() #cloud event msg is encoded into json format
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    
    eid = req.headers.get("Ce-Id", "unknown")
    image_key = req.headers.get("X-Image-Key") or ev.get("image_key", "unknown-key") #very important infomration because allow to link the system transaction
    insts = ev.get("instances")

    if insts is None:
        raise HTTPException(400, "Missing 'instances'")

    try:
        await queue.put({"event_id": eid, "image_key": image_key, "instances": insts}) #asynch elavoration into a event loop queue
        log.info(f"Enqueued {eid} | Key: {image_key} ({len(insts)}) queue={queue.qsize()}")
    except asyncio.QueueFull:
        log.warning(f"Queue full, rejecting {eid}")
        raise HTTPException(503, "Overloaded")
        
    return Response(status_code=204)
#######################################################################################################################################
@app.get("/health")
async def health():
    redis_errors = 0
    redis_success = 0
    if redis_client:
        try:
            err_val = await redis_client.get("metrics:redis_errors")
            succ_val = await redis_client.get("metrics:redis_success_ops")
            succ_tt_val = await redis_client.get("metrics:redis_success_ttl")
            redis_errors = int(err_val) if err_val else 0
            redis_success = int(succ_val) if succ_val else 0
            redis_success_ttl = int(succ_tt_val) if succ_tt_val else 0
        except Exception:
            pass

    return { 
        "status": "ok" if detector else "not_ready",  
        "queue_size": queue.qsize(),  
        "threshold": detector.th if detector else None, 
        "processed_counter": detector.cnt if detector else 0, 
        "ood_buffer_size": len(ood_buffer),
        "redis_success_ops": redis_success,
        "redis_errors": redis_errors,
        "redis_ttl_update": redis_success_ttl
    }

@app.post("/store_image") #this endpoint receive the images raw/png by the rest api, all the metadata are into the header of the http call
async def store_image( request: Request, content_type: Optional[str] = Header(None, alias="Content-Type"),  x_filename: Optional[str] = Header(None, alias="X-Filename"),
    x_ttl: Optional[int] = Header(None, alias="X-TTL"),  x_metadata: Optional[str] = Header(None, alias="X-Metadata"), x_image_key: Optional[str] = Header(None, alias="X-Image-Key") ):

    if redis_client is None:
        raise HTTPException(503, "Redis not available")
    
    if not x_image_key:
        raise HTTPException(400, "Missing required header: X-Image-Key")
    
    img_bytes = await request.body() #get the images byte into the body

    if not img_bytes:
        raise HTTPException(400, "Empty payload body")
    
    #create the redis key for the image byte blob storing
    key = f"image:{x_image_key}"
    meta_key = f"{key}:meta"
    ttl = x_ttl or REDIS_IMAGE_TTL
    # having the content type of the stored image is very important because allow to forward binary data together with the metadata for re-create the original image
    meta = { "filename": x_filename or "unknown", "content_type": content_type or "application/octet-stream", "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
        "ttl": ttl,  "metadata": x_metadata or "", "resolved_key": key, }
    try:
        async with redis_client.pipeline(transaction=True) as pipe: #redis in memroy transactions
            pipe.setex(key, ttl, img_bytes) #set the image blob into redis and set a ttl 
            pipe.hset(meta_key, mapping=meta) #store metadata as hash set 
            pipe.expire(meta_key, ttl) #define the same ttl for the image related metadata hash set 
            pipe.incr("metrics:redis_success_ops") #log the operation 
            await pipe.execute()
    except Exception as e:
        await increment_redis_error()
        log.error(f"Failed to store image in Redis: {e}")
        raise HTTPException(500, "Failed to store image in Redis")

    log.info("Image saved in redis with key value: %s", key)
    return {"image_id": x_image_key, "ttl": ttl, "redis_key": key}
