import os

# https://grpc.github.io/grpc/php/md_doc_fork_support.html
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "epoll1"

import argparse
import asyncio
import logging
import uuid
from typing import Dict, Optional

import aiohttp
import cv2
import grpc
import numpy as np

from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from kserve import Model, ModelServer, InferRequest, InferResponse, InferInput, InferOutput

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3

CLIENT_TIMEOUT = aiohttp.ClientTimeout(total=10)
GRPC_OPTIONS = [("grpc.max_receive_message_length", 100 * 1024 * 1024), ("grpc.max_send_message_length", 100 * 1024 * 1024)] #increase 

def numpy_to_tensor_proto(array: np.ndarray, dtype: types_pb2.DataType) -> tensor_pb2.TensorProto:
    tensor = tensor_pb2.TensorProto()
    tensor.dtype = dtype
    tensor.tensor_shape.CopyFrom( tensor_shape_pb2.TensorShapeProto( dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in array.shape] ) )
    tensor.tensor_content = array.tobytes()
    return tensor

def tensor_proto_to_numpy(tensor: tensor_pb2.TensorProto, target_dtype: np.dtype) -> np.ndarray:
    shape = [int(dim.size) for dim in tensor.tensor_shape.dim]
    # no byte into format content 
    if len(tensor.float_val) > 0:
        return np.asarray(tensor.float_val, dtype=target_dtype).reshape(shape)
    elif len(tensor.int_val) > 0:
        return np.asarray(tensor.int_val, dtype=target_dtype).reshape(shape)
        
    raise ValueError(f"Tensor fields float_val and int_val are both empty for shape: {shape}")

class ImageTransformer(Model):
    def __init__(self, name: str, predictor_host: str, broker: str, broker_host: str, ce_type: str, istio_gateway: str):
        super().__init__(name)
        self.name = name
        self.predictor_host = predictor_host
        self.ready = True
        self.broker = broker
        self.broker_host = broker_host
        self.ce_type = ce_type
        self.istio_gateway = istio_gateway
        self.detector_host = f"ood-detector-{name}.default.example.com"
        self.detector_url = f"{self.istio_gateway}/store_image"

        self._session: Optional[aiohttp.ClientSession] = None
        self._grpc_channel: Optional[grpc.aio.Channel] = None
        self._grpc_stub: Optional[prediction_service_pb2_grpc.PredictionServiceStub] = None

        self._base_kafka_headers = { "Host": self.broker_host, "Ce-Specversion": "1.0", "Ce-Type": self.ce_type, "Ce-Source": self.name, "Content-Type": "application/json", }
        self._base_detector_headers = { "Host": self.detector_host, "X-TTL": "600", "X-Metadata": self.name, }

    async def _get_tf_stub(self) -> prediction_service_pb2_grpc.PredictionServiceStub:
        if self._grpc_stub is None:
            self._grpc_channel = grpc.aio.insecure_channel(self.predictor_host, options=GRPC_OPTIONS)
            self._grpc_stub = prediction_service_pb2_grpc.PredictionServiceStub(self._grpc_channel)
        return self._grpc_stub

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=2000, limit_per_host=500, keepalive_timeout=120, force_close=False)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _send_to_kafka(self, embedding, image_key: str):
        try:
            payload = {"image_key": image_key, "embedding": embedding.tolist()}
            headers = self._base_kafka_headers.copy()
            headers["Ce-Id"] = image_key
            headers["X-Image-Key"] = image_key

            session = await self._get_session()
            async with session.post(self.broker, json=payload, headers=headers, timeout=CLIENT_TIMEOUT) as response:
                if response.status >= 300:
                    logger.error("[%s] ERROR Kafka HTTP %s [%s]", self.name, response.status, image_key)
        except Exception as exc:
            logger.error("[%s] Kafka Exception [%s]: %s", self.name, image_key, exc)

    async def _store_image(self, image_bytes: bytes, filename: str, content_type: str, image_key: str):
        try:
            headers = self._base_detector_headers.copy()
            headers["Content-Type"] = content_type
            headers["X-Filename"] = filename
            headers["X-Image-Key"] = image_key

            session = await self._get_session()
            async with session.post(self.detector_url, headers=headers, data=image_bytes, timeout=CLIENT_TIMEOUT) as response:
                if response.status >= 300:
                    logger.error("[%s] ERROR Detector HTTP %s [%s]", self.name, response.status, image_key)
        except Exception as exc:
            logger.error("[%s] Detector Exception [%s]: %s", self.name, image_key, exc)
        
    async def preprocess(self, payload: InferRequest, headers: Optional[Dict[str, str]] = None) -> InferRequest:
        headers = headers or {}
        image_key = uuid.uuid4().hex 
        filename = headers.get("x-filename", "unknown.png")
        image_type = headers.get("x-custom-param", "image/png") 

        input_tensor = payload.get_input_by_name("input") #collect raw png byte 
        if input_tensor is None:
            logger.error("[%s] NO Input V2", self.name)
            raise ValueError("Empty Input V2")
 
        raw_arr = input_tensor.as_numpy() # to numpy
        nparr = np.asarray(raw_arr, dtype=np.uint8).ravel() #as unit8 and 1D array 
        
        asyncio.create_task(self._store_image(nparr.tobytes(), filename, image_type, image_key)) #fire and forgot

        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) #decode as opencv image matrix
        if image is None:
            logger.error("[%s] ERROR image decode, image key: %s", self.name, image_key)
            raise ValueError("ERROR image decode")

        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image) # image matrix to rgb

        if image.shape[0] != IMAGE_HEIGHT or image.shape[1] != IMAGE_WIDTH:
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR) # resize in case

        tensor_data = np.expand_dims(image.astype(np.float32, copy=False), axis=0) #create the input tensor with also the batch dim (1, 256, 256, 3)
        return InferRequest(model_name=self.name, request_id=image_key, infer_inputs=[InferInput(name="input", shape=list(tensor_data.shape), datatype="FP32", data=tensor_data)])

    async def predict(self, payload: InferRequest, headers=None, response_headers=None) -> InferResponse:
        image_key = getattr(payload, "id", "N/A") # remap with id by the framework
        input_tensor = payload.get_input_by_name("input")
        
        array = input_tensor.as_numpy()

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = "serving_default"
        
        request.inputs["input_image"].CopyFrom(numpy_to_tensor_proto(array, types_pb2.DT_FLOAT))

        try:
            stub = await self._get_tf_stub()
            tf_response = await stub.Predict(request, timeout=30.0) #asynch
        except Exception as exc:
            logger.error("[%s] TF Serving Error [%s]: %s", self.name, image_key, exc)
            raise
            
        pred_class = tensor_proto_to_numpy(tf_response.outputs["predicted_class"], np.int32)
        embeddings = tensor_proto_to_numpy(tf_response.outputs["embedding"], np.float32)

        outputs = [ InferOutput(name="predicted_class", shape=list(pred_class.shape), datatype="INT32", data=pred_class), InferOutput(name="embedding", shape=list(embeddings.shape), datatype="FP32", data=embeddings) ]
        return InferResponse(response_id=image_key, model_name=self.name, infer_outputs=outputs)

    async def postprocess(self, response: InferResponse, headers=None) -> InferResponse:
        image_key = getattr(response, "id", "N/A")
        predicted = response.get_output_by_name("predicted_class")
        if predicted is None:
            logger.error("[%s] Output 'predicted_class' not found", self.name)
            raise ValueError("Output 'predicted_class' not found")

        pred_array = predicted.as_numpy()
        predicted_class = int(pred_array.flatten()[0])

        embedding = response.get_output_by_name("embedding")
        asyncio.create_task(self._send_to_kafka(embedding.as_numpy(), image_key)) #fire and forgot
        
        return InferResponse(response_id=image_key, model_name=self.name, infer_outputs=[InferOutput(name="predicted_class", shape=[1], datatype="INT32", data=[predicted_class])])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", required=True)
    parser.add_argument("--predictor_host", required=True)
    parser.add_argument("--broker", default="http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker")
    parser.add_argument("--broker_host", default="kafka-broker-ingress.knative-eventing.svc.cluster.local")
    parser.add_argument("--ce_type", default="org.kubeflow.serving.inference.request")
    parser.add_argument("--istio_gateway", default="http://istio-ingressgateway.istio-system.svc.cluster.local")

    args, _ = parser.parse_known_args()
    models = [m.strip() for m in args.model_names.split(",") if m.strip()]
    if not models:
        raise ValueError("No model specified")

    workers = int(os.getenv("WORKERS", "4"))
    
    transformers = [ ImageTransformer(name=m, predictor_host=args.predictor_host, broker=args.broker, broker_host=args.broker_host, ce_type=args.ce_type, istio_gateway=args.istio_gateway) for m in models ]
    ModelServer(http_port=8080, grpc_port=8081, workers=workers, enable_grpc=True).start(transformers)
