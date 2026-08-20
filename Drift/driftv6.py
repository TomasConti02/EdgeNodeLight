import asyncio
import datetime
import json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("ood-detector")

RETRY_DELAY = 0.05  
RETRY = 5          
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_IMAGE_TTL = int(os.getenv("REDIS_IMAGE_TTL", "600"))
REDIS_OOD_EXTENDED_TTL = int(os.getenv("REDIS_OOD_EXTENDED_TTL", "3600"))
OOD_FORWARD_ENDPOINT = os.getenv("OOD_FORWARD_ENDPOINT", "")
OOD_FORWARD_BATCH_SIZE = int(os.getenv("OOD_FORWARD_BATCH_SIZE", "100"))
STATE_KEY = "detector_state"
FALLBACK_FILE = "/data/queue/detector_state.pkl"
redis_client: Optional[redis.Redis] = None
class RealTimeOODDetector:
    def __init__(self, cent: np.ndarray, inv_cov: np.ndarray, win_sz: int, batch_sz: int, init_th: Optional[float] = None, smooth: float = 0.9, min_s: int = 50, perc: float = 90.0, 
        max_perc: float = 95.0, safe_th: Optional[float] = None, max_drop: int = 5, max_up: int = 5, med_win: int = 5, smooth_safe: bool = False):
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
        log.debug(f"RealTimeOODDetector initialized with win_sz={win_sz}, batch_sz={batch_sz}, init_th={init_th}")

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

        old_th = self.th
        if self.th is None:
            self.th = cand
            log.info(f"[Detector] Initialized dynamic threshold to {self.th:.4f}")
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
            log.info(f"[Detector] First-time threshold set: {self.th:.4f}")

        is_ood = self.th is not None and dist > self.th
        return dist, is_ood, self.th

    def get_state(self) -> dict:
        return {k: getattr(self, k) for k in ( "knn", "cent", "inv_cov", "dist_buf", "batch_cnt", "th", "safe_th",  "batch_sz", "smooth", "min_s", "perc", "max_perc", "max_drop", "max_up",
            "drop_cnt", "up_cnt", "med_win", "batch_percs", "smooth_safe", "th_flow", "cnt"  )}

    def set_state(self, st: dict) -> None:
        for k, v in st.items():
            if k in ("dist_buf", "batch_percs", "th_flow"):
                maxlen = getattr(self, k).maxlen
                setattr(self, k, deque(v, maxlen=maxlen))
            else:
                setattr(self, k, v)
        log.info("[Detector] State successfully loaded and restored.")


detector = None
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "5000")) 
queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
workers = []
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4")) # parallel workers


def init_detector() -> RealTimeOODDetector:
    centroids_path = os.getenv("CENTROIDS_PATH", "centroids.npy")
    inv_cov_path = os.getenv("INV_COV_MATRIX_PATH", "inv_cov_matrix.npy")
    log.info(f"Loading centroids from {centroids_path} and inverse covariance matrix from {inv_cov_path}")
    cent = np.load(centroids_path)
    inv = np.load(inv_cov_path)

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


async def save_state():
    if detector is None:
        return
    state = detector.get_state()
    try:
        os.makedirs(os.path.dirname(FALLBACK_FILE), exist_ok=True)
        with open(FALLBACK_FILE, "wb") as f:
            pickle.dump(state, f)
        log.info(f"[State] State successfully written to {FALLBACK_FILE}")
    except Exception as e:
        log.error(f"[State] Failed to write fallback file {FALLBACK_FILE}: {e}")


async def restore_state():
    global detector
    if os.path.exists(FALLBACK_FILE):
        try:
            with open(FALLBACK_FILE, "rb") as f:
                st = pickle.load(f)
            detector = init_detector()
            detector.set_state(st)
            return
        except Exception as e:
            log.warning(f"[State] Fallback file restore failed: {e}. Initializing fresh detector.")
    detector = init_detector()

async def forward_ood_batch(): #clean up the ood detected queue, here we can add the forward to the data lake system 
    if not redis_client:
        return
    try:
        queue_len = await redis_client.llen("queue:ood_to_flush") #items into the queue
    except Exception as e:
        log.error(f"[Forwarder] Failed to get queue length: {e}")
        return
    if queue_len == 0:
        return
    batch_size = min(queue_len, OOD_FORWARD_BATCH_SIZE)
    try:
        async with redis_client.pipeline(transaction=True) as pipe:
            pipe.lrange("queue:ood_to_flush", 0, batch_size - 1) #fifo queue reading
            pipe.ltrim("queue:ood_to_flush", batch_size, -1) #clean up the queue tail and keep the items from batch size to the end -1
            results = await pipe.execute() #collect the results 
    except Exception as e:
        log.error(f"[Forwarder] Redis pipeline error during range/trim: {e}")
        return
    raw_items = results[0]
    if not raw_items:
        return
    redis_keys_to_delete = [] #collect all the key 
    for raw in raw_items:
        try:
            item = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw) # in dictionary
            img_redis_key = item.get("img_redis_key")
            meta_redis_key = item.get("meta_redis_key")

            if img_redis_key:
                redis_keys_to_delete.append(img_redis_key)
            if meta_redis_key:
                redis_keys_to_delete.append(meta_redis_key)
        except Exception as parse_err:
            log.warning(f"[Forwarder] Failed to parse OOD queue item: {parse_err}")

    if redis_keys_to_delete:
        try:
            async with redis_client.pipeline(transaction=True) as pipe:
                for k in redis_keys_to_delete:
                    pipe.delete(k) #delete all the blob image, metadata from the keys 
                await pipe.execute()
            log.info(f"[Forwarder] Successfully cleaned up {len(redis_keys_to_delete)} Redis keys (images & metadata) from OOD batch.")
        except Exception as del_err:
            log.error(f"[Forwarder] Failed to delete Redis keys during cleanup: {del_err}")


#if the embedding arrive before the image the corutine have to retrain with a delay before considering the sample lost
async def wait_for_redis_image(img_key: str, meta_key: str, retries: int = RETRY):
    for attempt in range(retries):
        img = await redis_client.get(img_key)
        meta = await redis_client.hgetall(meta_key)
        if img is not None and meta:
            return img, meta
        await asyncio.sleep(RETRY_DELAY)
    return None, None

async def worker_loop(worker_id: int):
    log.info(f"[Worker-{worker_id}] Background processing loop started.")
    try:
        while True:
            item = await queue.get()
            eid = item.get("event_id", "unknown")
            image_key = item.get("image_key", "unknown-key")
            inst = item.get("embedding") 
            
            try:
                if inst is not None:
                    dist, ood, th = await asyncio.to_thread(detector.process, inst)

                    if image_key != "unknown-key" and redis_client is not None:
                        img_redis_key = f"image:{image_key}"
                        meta_redis_key = f"{img_redis_key}:meta"

                        if ood:
                            log.warning(f"[OOD Detector] OOD DETECTED! Key: {image_key}")
                            img_bytes, meta_dict = await wait_for_redis_image(img_redis_key, meta_redis_key)
                            
                            if img_bytes is None or not meta_dict:
                                log.warning(f"[Worker-{worker_id}] Image not yet ready in Redis for OOD key {image_key}. Skipping persistence.")
                                continue

                            ood_record = json.dumps({ "img_redis_key": img_redis_key, "meta_redis_key": meta_redis_key })

                            async with redis_client.pipeline(transaction=True) as pipe:
                                pipe.expire(img_redis_key, REDIS_OOD_EXTENDED_TTL)
                                pipe.expire(meta_redis_key, REDIS_OOD_EXTENDED_TTL)
                                pipe.rpush("queue:ood_to_flush", ood_record)
                                pipe.incr("metrics:redis_success_ttl")
                                await pipe.execute()

                            queue_len = await redis_client.llen("queue:ood_to_flush")
                            if queue_len >= OOD_FORWARD_BATCH_SIZE:
                                await forward_ood_batch()
                        else:
                            try:
                                async with redis_client.pipeline(transaction=True) as pipe:
                                    pipe.delete(img_redis_key)
                                    pipe.delete(meta_redis_key)
                                    await pipe.execute()
                            except Exception:
                                pass
            except Exception as worker_err:
                log.error(f"[Worker-{worker_id}] Error processing event {eid}: {worker_err}")
            finally:
                queue.task_done()
                log.info(f"[Worker-{worker_id}] Completed processing event ID: {eid} | Remaining Queue: {queue.qsize()}")
    except asyncio.CancelledError:
        log.info(f"[Worker-{worker_id}] Worker loop cancelled.")

async def increment_redis_error():
    if redis_client:
        try:
            await redis_client.incr("metrics:redis_errors")
        except Exception:
            pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, workers, redis_client
    log.info(f"[Lifespan] Initializing Redis connection to {REDIS_HOST}:{REDIS_PORT}...")
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    
    try:
        await redis_client.ping()
        log.info("[Lifespan] Redis connection verified successfully.")
    except Exception as e:
        log.error(f"[Lifespan] CRITICAL: Redis connection failed: {e}")

    await restore_state()
    
    log.info(f"[Lifespan] Spawning {NUM_WORKERS} background workers...")
    workers = [asyncio.create_task(worker_loop(i + 1)) for i in range(NUM_WORKERS)]

    yield
    #after the scale to zero, that is a kernel singnal for the process 
    log.info("[Lifespan] Shutting down application...")
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    try:
        if redis_client:
            remaining_len = await redis_client.llen("queue:ood_to_flush")
            if remaining_len > 0:
                await forward_ood_batch()
    except Exception:
        pass
    await save_state() #save the ood detector state 
    try:
        await redis_client.close()
    except Exception:
        pass
    log.info("[Lifespan] Shutdown complete.")


app = FastAPI(lifespan=lifespan) #start the application

#payload = {"image_key": image_key, "embedding": embedding.tolist()}
#headers = {"Host": self.broker_host, "Ce-Id": uuid.uuid4().hex, "Ce-Specversion": "1.0", "Ce-Type": self.ce_type,"Ce-Source": self.name,"Content-Type": "application/json","X-Image-Key": image_key }
@app.post("/") # receive the http cloud events from the knative event driven support
async def receive(req: Request):
    try:
        ev = await req.json() #req payload into json
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    eid = req.headers.get("Ce-Id", "unknown")
    image_key = req.headers.get("X-Image-Key") or ev.get("image_key", "unknown-key") #image key link
    
    emb = ev.get("embedding")

    if emb is None:
        raise HTTPException(400, "Missing 'embedding'")
    try:
        
        queue.put_nowait({"event_id": eid, "image_key": image_key, "embedding": emb})
    except asyncio.QueueFull:
        raise HTTPException(503, "Overloaded")

    return Response(status_code=204) #aynch response

@app.get("/health")
async def health():
    redis_errors = 0
    redis_success = 0
    redis_success_ttl = 0
    redis_ood_queue_size = 0
    
    if redis_client:
        try:
            err_val = await redis_client.get("metrics:redis_errors")
            succ_val = await redis_client.get("metrics:redis_success_ops")
            succ_tt_val = await redis_client.get("metrics:redis_success_ttl")
            redis_ood_queue_size = await redis_client.llen("queue:ood_to_flush")
            
            redis_errors = int(err_val) if err_val else 0
            redis_success = int(succ_val) if succ_val else 0
            redis_success_ttl = int(succ_tt_val) if succ_tt_val else 0
        except Exception:
            pass

    return {"status": "ok" if detector else "not_ready", "queue_size": queue.qsize(), "threshold": detector.th if detector else None, 
            "processed_counter": detector.cnt if detector else 0, "ood_buffer_size": redis_ood_queue_size,
            "redis_success_ops": redis_success, "redis_errors": redis_errors, "redis_ttl_update": redis_success_ttl}

#headers = { "Host": self.detector_host, "Content-Type": content_type, "X-Filename": filename, "X-TTL": "600",  "X-Metadata": self.name,  "X-Image-Key": image_key } payload with image type raw byte
@app.post("/store_image")
async def store_image(request: Request, content_type: Optional[str] = Header(None, alias="Content-Type"), x_filename: Optional[str] = Header(None, alias="X-Filename"),
    x_ttl: Optional[int] = Header(None, alias="X-TTL"), x_metadata: Optional[str] = Header(None, alias="X-Metadata"), x_image_key: Optional[str] = Header(None, alias="X-Image-Key")):

    if redis_client is None:
        raise HTTPException(503, "Redis not available")

    if not x_image_key:
        raise HTTPException(400, "Missing required header: X-Image-Key")

    img_bytes = await request.body()
    if not img_bytes:
        raise HTTPException(400, "Empty payload body")

    key = f"image:{x_image_key}" #key for the image raw binary blob
    meta_key = f"{key}:meta" #key for the image's metadata
    ttl = x_ttl or REDIS_IMAGE_TTL
    
    meta = { "filename": x_filename or "unknown", "content_type": content_type or "unknown",
             "timestamp": datetime.datetime.now(timezone.utc).isoformat(), "ttl": str(ttl), "metadata": x_metadata or "", "resolved_key": key, }
    try:
        async with redis_client.pipeline(transaction=True) as pipe: #aynch redis transaction for saving metdata and image blob
            pipe.setex(key, ttl, img_bytes) #save image raw binary blob related to the key + small ttl
            pipe.hset(meta_key, mapping=meta) #save image metadata related to the key 
            pipe.expire(meta_key, ttl) # ttl
            pipe.incr("metrics:redis_success_ops") # ++1 to the success counter
            await pipe.execute()
    except Exception as e:
        await increment_redis_error()
        log.error(f"[StoreImage] Failed to store image for key {x_image_key}: {e}")
        raise HTTPException(500, "Failed to store image in Redis")

    log.info(f"[StoreImage] Stored image successfully | Key: {key} | Size: {len(img_bytes)} bytes")
    return {"image_id": x_image_key, "ttl": ttl, "redis_key": key} #aynch response
