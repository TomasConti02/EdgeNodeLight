import argparse
import asyncio
import logging
import os
import uuid
from typing import Dict, Optional

import aiohttp
import cv2
import grpc
import numpy as np

from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from kserve import Model, ModelServer, InferRequest, InferResponse, InferInput, InferOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
PRE_ALLOCATED_SHAPE = (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def numpy_to_tensor_proto(array: np.ndarray, base_tensor: tensor_pb2.TensorProto) -> tensor_pb2.TensorProto:
    tensor = tensor_pb2.TensorProto()
    tensor.CopyFrom(base_tensor)
    tensor.tensor_content = array.tobytes()
    return tensor

def tensor_proto_to_numpy(tensor: tensor_pb2.TensorProto, target_dtype: np.dtype) -> np.ndarray:
    shape = [int(dim.size) for dim in tensor.tensor_shape.dim]
    
    if tensor.tensor_content:
        array = np.frombuffer(tensor.tensor_content, dtype=target_dtype)
    elif target_dtype in (np.float32, np.float64):
        array = np.fromiter(tensor.float_val, dtype=target_dtype, count=len(tensor.float_val))
    elif target_dtype in (np.int32, np.int64):
        array = np.fromiter(tensor.int_val, dtype=target_dtype, count=len(tensor.int_val))
    else:
        raise ValueError(f"Dtype non supportato per TensorProto: {target_dtype}")
        
    return array.reshape(shape)

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

        self._session: Optional[aiohttp.ClientSession] = None
        self._grpc_channel: Optional[grpc.aio.Channel] = None
        self._grpc_stub: Optional[prediction_service_pb2_grpc.PredictionServiceStub] = None

        self._base_input_tensor = tensor_pb2.TensorProto()
        self._base_input_tensor.dtype = types_pb2.DT_FLOAT
        for dim in PRE_ALLOCATED_SHAPE:
            self._base_input_tensor.tensor_shape.dim.add(size=dim)

        self._output_buffer = np.empty(PRE_ALLOCATED_SHAPE, dtype=np.float32)
        logger.info("[%s] ImageTransformer inizializzato con successo.", self.name)

    async def _get_tf_stub(self) -> prediction_service_pb2_grpc.PredictionServiceStub:
        if self._grpc_stub is None:
            options = [
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ("grpc.max_send_message_length", 100 * 1024 * 1024)
            ]
            self._grpc_channel = grpc.aio.insecure_channel(self.predictor_host, options=options)
            self._grpc_stub = prediction_service_pb2_grpc.PredictionServiceStub(self._grpc_channel)
        return self._grpc_stub

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=2000, limit_per_host=500, keepalive_timeout=120, force_close=False)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _send_to_kafka(self, embedding: np.ndarray, image_key: str):
        try:
            if embedding.size == 0:
                return
            payload = {"image_key": image_key, "embedding": embedding.tolist()}
            headers = {
                "Host": self.broker_host,
                "Ce-Id": image_key,
                "Ce-Specversion": "1.0",
                "Ce-Type": self.ce_type,
                "Ce-Source": self.name,
                "Content-Type": "application/json",
                "X-Image-Key": image_key
            }
            session = await self._get_session()
            async with session.post(self.broker, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status >= 300:
                    logger.error("[%s] Errore Kafka HTTP %s [%s]", self.name, response.status, image_key)
        except Exception as exc:
            logger.error("[%s] Eccezione Kafka [%s]: %s", self.name, image_key, exc)

    async def _store_image(self, image: bytes, filename: str, content_type: str, image_key: str):
        try:
            headers = {
                "Host": self.detector_host,
                "Content-Type": content_type,
                "X-Filename": filename,
                "X-TTL": "600",
                "X-Metadata": self.name,
                "X-Image-Key": image_key
            }
            url = f"{self.istio_gateway}/store_image"
            session = await self._get_session()
            async with session.post(url, headers=headers, data=image, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status >= 300:
                    logger.error("[%s] Errore Detector HTTP %s [%s]", self.name, response.status, image_key)
        except Exception as exc:
            logger.error("[%s] Eccezione Detector [%s]: %s", self.name, image_key, exc)

    async def preprocess(self, payload: InferRequest, headers: Optional[Dict[str, str]] = None) -> InferRequest:
        headers = headers or {}
        image_key = uuid.uuid4().hex
        filename = headers.get("x-filename", "unknown.png")
        image_type = headers.get("x-custom-param", "image/png")

        input_tensor = payload.get_input_by_name("input")
        if input_tensor is None:
            logger.error("[%s] Input V2 'input' mancante", self.name)
            raise ValueError("Input V2 'input' mancante")

        # Estrazione diretta: dai log sappiamo che as_numpy() restituisce np.ndarray(uint8)
        img_bytes = input_tensor.as_numpy().tobytes()

        # Decodifica buffer con OpenCV
        nparr = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            logger.error("[%s] Impossibile decodificare l'immagine via cv2 per key: %s", self.name, image_key)
            raise ValueError("Immagine non decodificabile tramite cv2.imdecode")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape[0] != IMAGE_HEIGHT or image.shape[1] != IMAGE_WIDTH:
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        np.copyto(self._output_buffer[0], image.astype(np.float32, copy=False))

        try:
            raw_bytes_out = self._output_buffer.astype(np.uint8).tobytes()
            asyncio.create_task(self._store_image(raw_bytes_out, filename, image_type, image_key))
        except Exception as exc:
            logger.warning("[%s] Errore nella schedulazione di _store_image [%s]: %s", self.name, image_key, exc)

        return InferRequest(
            model_name=self.name,
            request_id=image_key,
            infer_inputs=[InferInput(name="input", shape=list(self._output_buffer.shape), datatype="FP32", data=self._output_buffer)]
        )

    async def predict(self, payload: InferRequest, headers=None, response_headers=None) -> InferResponse:
        image_key = getattr(payload, "id", "N/A")
        input_tensor = payload.get_input_by_name("input")
        array = np.asarray(input_tensor.data, dtype=np.float32).reshape(input_tensor.shape)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = "serving_default"
        request.inputs["input_image"].CopyFrom(numpy_to_tensor_proto(array, self._base_input_tensor))

        try:
            stub = await self._get_tf_stub()
            tf_response = await stub.Predict(request, timeout=30.0)
        except Exception as exc:
            logger.error("[%s] Errore TF Serving [%s]: %s", self.name, image_key, exc)
            raise

        if "output" in tf_response.outputs:
            preds = tensor_proto_to_numpy(tf_response.outputs["output"], np.float32)
        else:
            first_key = list(tf_response.outputs.keys())[0]
            preds = tensor_proto_to_numpy(tf_response.outputs[first_key], np.float32)

        outputs = [
            InferOutput(name="embedding", shape=list(preds.shape), datatype="FP32", data=preds),
            InferOutput(name="predicted_class", shape=list(preds.shape), datatype="FP32", data=preds)
        ]
        return InferResponse(response_id=image_key, model_name=self.name, infer_outputs=outputs)

    async def postprocess(self, response: InferResponse, headers=None) -> InferResponse:
        image_key = getattr(response, "id", "N/A")
        predicted = response.get_output_by_name("predicted_class")
        if predicted is None:
            logger.error("[%s] Output 'predicted_class' non trovato", self.name)
            raise ValueError("Output 'predicted_class' non trovato")

        logits_array = predicted.as_numpy()
        predicted_class = int(np.argmax(logits_array, axis=-1).flatten()[0])

        embedding = response.get_output_by_name("embedding")
        if embedding is not None: 
            asyncio.create_task(self._send_to_kafka(embedding.as_numpy(), image_key))
        else:
            logger.warning("[%s] Output 'embedding' assente per key: %s", self.name, image_key)

        return InferResponse(
            response_id=image_key,
            model_name=self.name,
            infer_outputs=[InferOutput(name="predicted_class", shape=[1], datatype="INT32", data=[predicted_class])]
        )

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
        raise ValueError("Nessun modello specificato")

    workers = int(os.getenv("WORKERS", "4"))
    
    transformers = [
        ImageTransformer(
            name=m,
            predictor_host=args.predictor_host,
            broker=args.broker,
            broker_host=args.broker_host,
            ce_type=args.ce_type,
            istio_gateway=args.istio_gateway
        ) for m in models
    ]
    ModelServer(http_port=8080, grpc_port=8081, workers=workers, enable_grpc=True).start(transformers)
