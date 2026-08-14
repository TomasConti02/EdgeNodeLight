## Kafka cluster

This deployment supports two distinct **Kafka cluster** configurations: a **Single-Node** Cluster for development and lightweight testing, and a **Multi-Node** Cluster featuring three active Kafka nodes for high availability.

The Kafka cluster deployment is managed by the **Strimzi Operator** ([strimzi.io](https://strimzi.io/)), which simplifies deploying and configuring Kafka clusters inside Kubernetes. An operator is a custom Kubernetes controller that extends the native Kubernetes API, automatically managing the deployment, configuration, and full lifecycle of a specific resource.

For cluster coordination and data synchronization, this architecture uses the **KRaft (Kafka Raft)** consensus algorithm instead of **Apache ZooKeeper**. Under KRaft, a message is committed to the log as soon as a majority of the replica nodes acknowledge they have received it. In the event of a broker node failure, a automatic leader election is triggered, selecting the node with the most up-to-date local log state. In terms of the **CAP theorem**, this system prioritizes Consistency and Availability within a single-fault domain system (such as a single Kubernetes cluster), ensuring a highly robust, fault-tolerant environment.

```bash
kubectl get pods -n kafka
NAME                                          READY   STATUS    RESTARTS        AGE
my-cluster-dual-role-0                        1/1     Running   0               69m
my-cluster-dual-role-1                        1/1     Running   0               69m
my-cluster-dual-role-2                        1/1     Running   0               69m
my-cluster-entity-operator-5659c749d8-66qqh   2/2     Running   0               66m
strimzi-cluster-operator-798fbc76f7-rz7xt     1/1     Running   2 (8m55s ago)   71m
```

------------------------------------------

## kafka topic configuration

By default, the Kafka configuration creates a topic with only one partition, which is then replicated across the broker nodes. For example:

```text
Topic: my-topic    TopicId: d7ZuzZuKQNm9OYvr6BqL3g    PartitionCount: 1    ReplicationFactor: 3    Configs: min.insync.replicas=2
    Topic: my-topic    Partition: 0    Leader: 0    Replicas: 0,1,2    Isr: 0,1,2
```

Part 1: Topic Overview
- Topic: my-topic: The name the message stream topic.

- PartitionCount: 1: The topic has only one data pipeline (Partition 0). All messages travel down this single lane, which guarantees they are processed in the exact order they are received.

- ReplicationFactor: 3: There are three exact copies of this partition running across the cluster to protect against data loss.

- min.insync.replicas=2: This is a data-safety rule. It means a producer cannot successfully write/commit a message unless at least 2 out of the 3 nodes confirm they have saved it [ Raft consistency ].

Part 2: Partition Detail
- Partition: 0: Identifies the specific partition being described (since count is 1, only Partition 0 exists).

- Leader: 0: Broker node 0 (my-cluster-dual-role-0) is the active "brain" for this partition. Messaggies goes directly to node 0 when send or read data.

- Replicas: 0,1,2: The physical copies of the data live on broker nodes 0, 1, and 2.

- Isr: 0,1,2 (In-Sync Replicas): This is the most important health indicator. It shows that all three nodes are alive, healthy, and completely caught up with the leader. If node 1 or 2 crashes, it will automatically drop out of this list until it recovers.


-------------------------------

## Kafka with knative eventing

list of topics:

```bash 
kubectl exec -it my-cluster-dual-role-0 -n kafka -- \
/opt/kafka/bin/kafka-consumer-groups.sh \
--bootstrap-server localhost:9092 \
--list

knative-trigger-default-inference-trigger
```

knative eventing topic topic:

```bash
kubectl exec -it my-cluster-dual-role-0 -n kafka -- \
/opt/kafka/bin/kafka-consumer-groups.sh \
--bootstrap-server localhost:9092 \
--describe \
--group knative-trigger-default-inference-trigger


GROUP                                     TOPIC                               PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG             CONSUMER-ID                                                                               HOST            CLIENT-ID
knative-trigger-default-inference-trigger knative-broker-default-kafka-broker 0          1               1               0               consumer-knative-trigger-default-inference-trigger-1-8d3e04b8-cc7a-4f19-9527-3b258c00b264 /10.244.1.12    consumer-knative-trigger-default-inference-trigger-1
knative-trigger-default-inference-trigger knative-broker-default-kafka-broker 1          3               3               0               consumer-knative-trigger-default-inference-trigger-1-8d3e04b8-cc7a-4f19-9527-3b258c00b264 /10.244.1.12    consumer-knative-trigger-default-inference-trigger-1
knative-trigger-default-inference-trigger knative-broker-default-kafka-broker 2          2               2               0               consumer-knative-trigger-default-inference-trigger-1-8d3e04b8-cc7a-4f19-9527-3b258c00b264 /10.244.1.12    consumer-knative-trigger-default-inference-trigger-1
 
```

Real time stream of Knative Kafka msg:

```bash
 kubectl exec -it my-cluster-dual-role-0 -n kafka -- \
/opt/kafka/bin/kafka-console-consumer.sh \
--bootstrap-server localhost:9092 \
--topic knative-broker-default-kafka-broker \
--from-beginning

```
