## Deploy event driven architecture
In order to execute the deploy
```bash
python3 deployv1.py
```
-------------------------------
## Check system's configuration results

1. Knative eventing:
```bash
kubectl get pods -n knative-eventing
NAME                                      READY   STATUS    RESTARTS      AGE
eventing-controller-58fcb4489c-wkqnw      1/1     Running   2 (13m ago)   14m
eventing-webhook-6fff4fbc5-n5lql          1/1     Running   2 (13m ago)   14m
job-sink-7c7468fc55-2plrf                 1/1     Running   3 (13m ago)   14m
kafka-broker-receiver-67c54b9f54-ptk9f    1/1     Running   0             13m
kafka-controller-f7ccc8bb9-9mln9          1/1     Running   0             13m
kafka-webhook-eventing-596b56f567-rf946   1/1     Running   0             13m

```

2. Kafka Cluster:
```bash
kubectl get pods -n kafka
NAME                                          READY   STATUS    RESTARTS   AGE
my-cluster-dual-role-0                        1/1     Running   0          9m11s
my-cluster-dual-role-1                        1/1     Running   0          9m11s
my-cluster-dual-role-2                        1/1     Running   0          9m11s
my-cluster-entity-operator-5bb77687bc-fn75m   2/2     Running   0          5m50s
strimzi-cluster-operator-798fbc76f7-znvw8     1/1     Running   0          11m

```

3. Broker:
```bash
kubectl get broker  
NAME          URL                                                                                   AGE    READY   REASON
kafka-broker   http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker   6m7s   True    
```
