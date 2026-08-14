import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend } from 'k6/metrics';
import { htmlReport } from "https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.1.0/index.js";

const inferenceDuration = new Trend('inference_processing_time');

export const options = {
  stages: [
    { duration: '1m', target: 5 },
    { duration: '2m', target: 15 },
    { duration: '3m', target: 30 },
    { duration: '3m', target: 30 },
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    'http_req_failed': ['rate<0.01'],
    'inference_processing_time': ['p(95)<4000', 'p(99)<5000'],
  },
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
};

const rawImage = open('./immagine.png', 'b');
const imageBytes = new Uint8Array(rawImage);

const INGRESS_HOST = '192.168.17.37';
const INGRESS_PORT = '31978';

export function testModel(modelName) {
  const url = `http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${modelName}/infer`;
  const hostName = `${modelName}-predictor.default.example.com`;

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

  const headerStr = JSON.stringify(header);
  const headerBytes = new TextEncoder().encode(headerStr);
  const headerLen = headerBytes.byteLength;

  const payload = new Uint8Array(headerLen + imageBytes.byteLength);
  payload.set(headerBytes, 0);
  payload.set(imageBytes, headerLen);

  const params = {
    headers: {
      'Content-Type': 'application/octet-stream',
      'Host': hostName,
      'Inference-Header-Content-Length': headerLen.toString(),
      'X-Image-Key': `k6-test-${Date.now()}`
    },
    timeout: '15s',
  };

  const res = http.post(url, payload.buffer, params);
  
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

  sleep(0.2);
}

export default function() {
  const modelName = __ITER % 2 === 0 ? 'simple-cnn' : 'simple-cnn-test';
  testModel(modelName);
}

export function handleSummary(data) {
  return {
    "report-simple-cnn-test.html": htmlReport(data),
    stdout: textSummary(data, { indent: " ", enableColors: true }),
  };
}
