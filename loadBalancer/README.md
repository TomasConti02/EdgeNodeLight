### Configure the LoadBalancer for the cluster
In kind there is not a load balander provide by the cloud architecture.

Apply the CRD of the loadbalancer

```bash
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.16.1/config/manifests/metallb-native.yaml
```

Check if all pod are running:
```bash
kubectl get pods -n metallb-system
NAME                          READY   STATUS    RESTARTS   AGE
controller-867646bb46-w86tl   1/1     Running   0          4m43s
speaker-855tl                 1/1     Running   0          4m43s
speaker-9kt8c                 1/1     Running   0          4m43s
speaker-l6v6b                 1/1     Running   0          4m43s
```

Check the network of Kind k8s containers
```bash
docker network inspect -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}' kind
172.18.0.1fc00:f853:ccd:e793::1
```

Check if possible ip conflict issue:
```bash
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker ps -q)
172.18.0.3
172.18.0.4
172.18.0.2
```

Deploy the load balancer:
```bash
kubectl apply -f loadbalancer.yaml 
ipaddresspool.metallb.io/kind-pool created
l2advertisement.metallb.io/l2advertisement created
```
Check is now gataway is bound:

```bash
kubectl get svc -n istio-system istio-ingressgateway
NAME                   TYPE           CLUSTER-IP    EXTERNAL-IP      PORT(S)                                      AGE
istio-ingressgateway   LoadBalancer   10.96.64.84   172.18.255.200   15021:32269/TCP,80:32020/TCP,443:32762/TCP   3h53m
```
Now there is a testing loadbalancer and can be avoid port forwarding.
Go back to the ./model_testing dir and launch:

```bash
curl -X POST http://172.18.255.200/v1/models/simple-cnn:predict \
  -H "Host: simple-cnn.default.example.com" \
  -H "Content-Type: application/json" \
  -d @image.json
```
