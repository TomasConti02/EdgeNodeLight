import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';
import { textSummary } from "https://jslib.k6.io/k6-summary/0.1.0/index.js";

const inferenceDuration = new Trend('inference_processing_time');

export const options = {
  stages: [
    { duration: '30s', target: 5 },   // Warm-up 
    { duration: '5m',  target: 12 },  // stable
    { duration: '30s', target: 0 },   // Cool-down 
  ],
  thresholds: {
    'http_req_failed': ['rate<0.01'],
    'inference_processing_time': ['p(95)<3000', 'p(99)<4000'],
  },
  summaryTrendStats: ['count', 'avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
};

const rawImage = open('./immagine_256.png', 'b');
const imageBytes = new Uint8Array(rawImage);

const header = {
  inputs: [{
    name: "input",
    shape: [imageBytes.byteLength],
    datatype: "UINT8",
    parameters: {
      binary_data_size: imageBytes.byteLength
    }
  }]
};

const headerBytes = new TextEncoder().encode(JSON.stringify(header));
const headerLen = headerBytes.byteLength;

const payloadBuffer = new Uint8Array(headerLen + imageBytes.byteLength);
payloadBuffer.set(headerBytes, 0);
payloadBuffer.set(imageBytes, headerLen);

const INGRESS_HOST = '192.168.17.37';
const INGRESS_PORT = '31978';

export function testModel(modelName) {
  const url = `http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${modelName}/infer`;
  
  const params = {
    headers: {
      'Content-Type': 'application/octet-stream',
      'Host': `${modelName}-predictor.default.example.com`,
      'Inference-Header-Content-Length': headerLen.toString(),
      'X-Image-Key': `k6-test-${Date.now()}-${Math.random()}`,
      'x-filename': 'immagine.png',
      'content-type': 'image/png'
    },
    timeout: '10s',
  };

  const res = http.post(url, payloadBuffer.buffer, params);

  const success = check(res, {
    'status is 200': (r) => r.status === 200,
    'has predicted_class output': (r) => {
      try {
        const json = r.json();
        return json && json.outputs && json.outputs.some(o => o.name === 'predicted_class');
      } catch (e) {
        return false;
      }
    },
  });

  if (success) {
    inferenceDuration.add(res.timings.duration);
  }
  sleep(0.3); //300ms
}

export default function() {
  const modelName = __ITER % 2 === 0 ? 'simple-cnn' : 'simple-cnn-test';
  testModel(modelName);
}

export function handleSummary(data) {
  return {
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}
