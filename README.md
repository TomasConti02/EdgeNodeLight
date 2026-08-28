# EdgeNodeLight
```bash
FILE="immagine.png"
SIZE=$(stat -f%z "$FILE" 2>/dev/null || stat -c%s "$FILE")
JSON_HEADER=$(printf '{"inputs":[{"name":"input","shape":[%d],"datatype":"UINT8","parameters":{"binary_data_size":%d}}]}' "$SIZE" "$SIZE")
HEADER_LEN=$(printf '%s' "$JSON_HEADER" | wc -c | tr -d ' ')

{
  printf '%s' "$JSON_HEADER"
  cat "$FILE"
} | curl -X POST \
  "http://192.168.17.37:31978/v2/models/simple-cnn/infer" \
  -H "Host: simple-cnn-predictor.default.example.com" \
  -H "Content-Type: application/octet-stream" \
  -H "Inference-Header-Content-Length: $HEADER_LEN" \
  -H "X-Image-Key: k6-test-manual" \
  -H "x-filename: $FILE" \
  --data-binary @-

{"model_name":"simple-cnn","model_version":null,"id":"3400267f72d642e69c8a2251d48661cc","parameters":null,"outputs":[{"name":"predicted_class","shape":[1],"datatype":"INT32","parameters":null,"data":[2]}]}
```
```bash
FILE="immagine.png"
SIZE=$(stat -f%z "$FILE" 2>/dev/null || stat -c%s "$FILE")
JSON_HEADER=$(printf '{"inputs":[{"name":"input","shape":[%d],"datatype":"UINT8","parameters":{"binary_data_size":%d}}]}' "$SIZE" "$SIZE")
HEADER_LEN=$(printf '%s' "$JSON_HEADER" | wc -c | tr -d ' ')

{
  printf '%s' "$JSON_HEADER"
  cat "$FILE"
} | curl -X POST \
  "http://192.168.17.37:31978/v2/models/simple-cnn-test/infer" \
  -H "Host: simple-cnn-test-predictor.default.example.com" \
  -H "Content-Type: application/octet-stream" \
  -H "Inference-Header-Content-Length: $HEADER_LEN" \
  -H "X-Image-Key: k6-test-manual" \
  -H "x-filename: $FILE" \
  --data-binary @-

{"model_name":"simple-cnn-test","model_version":null,"id":"b34a1500faec48f89f3e764167235059","parameters":null,"outputs":[{"name":"predicted_class","shape":[1],"datatype":"INT32","parameters":null,"data":[2]}]}
```

```text
kubectl get pods -o wide
NAME                                                             READY   STATUS    RESTARTS   AGE   IP            NODE                             NOMINATED NODE   READINESS GATES
ood-detector-simple-cnn-00001-deployment-6d5d544fdf-wk2zv        3/3     Running   0          74s   10.42.1.235   tconti-mscthesis-a.mmwunibo.it   <none>           <none>
ood-detector-simple-cnn-test-00001-deployment-859c56696c-mnfbq   3/3     Running   0          73s   10.42.1.236   tconti-mscthesis-a.mmwunibo.it   <none>           <none>
simple-cnn-predictor-00001-deployment-59f8f58465-2d5sb           3/3     Running   0          90s   10.42.3.109   nvidia-ca7                       <none>           <none>
simple-cnn-test-predictor-00001-deployment-7df5f67574-xxpj2      3/3     Running   0          90s   10.42.3.110   nvidia-ca7                       <none>           <none>

```
