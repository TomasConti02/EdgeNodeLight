## Knative cluster configuration

The cluster currently runs a **Knative version 1.19.8**. Outdated version could cause compatibility issues if we allow Edge nodes to update dynamically.

To guarantee system stability is required a stable Knative version and validate it before release a new versione of the Edge node code base.

```bash
kubectl get pods -n knative-eventing
NAME                                      READY   STATUS    RESTARTS      AGE
eventing-controller-58fcb4489c-zpxln      1/1     Running   2 (74m ago)   75m
eventing-webhook-6fff4fbc5-22dqb          1/1     Running   1 (74m ago)   75m
job-sink-7c7468fc55-s2nfm                 1/1     Running   2 (74m ago)   75m
kafka-broker-receiver-67c54b9f54-zrbzd    1/1     Running   0             74m
kafka-controller-f7ccc8bb9-bscmd          1/1     Running   0             74m
kafka-webhook-eventing-596b56f567-nvt6b   1/1     Running   0             74m
```

Knative Eventing use Apache Kafka as the underlying messaging backbone (Message Exchange Support).

Using Kafka as the backing store for Knative Brokers ensures high throughput, persistence, and better fault tolerance compared to the default in-memory implementation.

Architecture Overview

The architecture relies on an event-driven, decoupled in time and space pattern consisting of two main components:
1. ConfigMap: Defines how Knative connects to the Kafka cluster and how it should provision underlying Kafka topics.
2. Broker: Acts as the central event hub and entry point for all incoming events within the namespace.

The ConfigMap centralizes the connection details and default topic behaviors for the Kafka-backed infrastructure.
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kafka-broker-config
  namespace: knative-eventing
data:
  bootstrap.servers: "my-cluster-kafka-bootstrap.kafka.svc:9092" # The internal cluster URL/endpoint of kafka bootstrap service
  default.topic.partitions: "3" ## Default number of partitions for automatically created Kafka topics
  default.topic.replication.factor: "1" # Replication factor for the topics (Note: Set to 3 in Production environments)
```
The Broker is the decoupled entry point of the event-driven system. It receives CloudEvents via HTTP and routes them based on Triggers filtering roles.
```yaml
apiVersion: eventing.knative.dev/v1
kind: Broker
metadata:
  name: kafka-broker
  namespace: default
  annotations:
    eventing.knative.dev/broker.class: Kafka # Ask to Broker to use Kafka as event support
spec:
  config:
    apiVersion: v1
    kind: ConfigMap # link Broker with the even driven support of knative + Kafka
    name: kafka-broker-config
    namespace: knative-eventing
```
To verify that the Kafka Broker is successfully initialized and ready to accept traffic by running:
```bash
kubectl get broker kafka-broker -n default
NAME           URL                                                                                   AGE   READY   REASON
kafka-broker   http://kafka-broker-ingress.knative-eventing.svc.cluster.local/default/kafka-broker   10m   True    
```
Broker is the central event hub. Supported by Kafka, it securely stores incoming events(that are messages) streams, enabling the system to decouple services in both time and space. Trigger is the router or "bridge" between the Broker and your Consumer service. It defines a declarative routing rule by filtering events based on their metadata attributes and allow routing to the consumer target.

1. Send: The producer sends an HTTP request containing a CloudEvent directly to the local cluster URL of the kafka-broker. CloudEvent is a standard HTTP REST request (usually an HTTP POST with a JSON body), but it includes specific metadata headers (like ce-type, ce-id, and ce-source).

2. Store: The kafka-broker receives the event and securely stores it inside a dedicated Apache Kafka topic.

3. Inspect: The inference-trigger constantly monitors the kafka-broker and inspects the metadata attributes of every incoming event.

4. Filter: The Trigger checks if the event type attribute matches org.kubeflow.serving.inference.request.

5. Deliver: Once a match is found, the Trigger automatically forwards the event via an HTTP POST request to your consumer Knative Service.

Trigger yaml example:

```yaml
apiVersion: eventing.knative.dev/v1
kind: Trigger
metadata:
  name: inference-trigger
  namespace: default
spec:
  broker: kafka-broker  # 1. Which broker to listen to
  filter:
    attributes:
      type: org.kubeflow.serving.inference.request # 2. What kind of events to look for
  subscriber:
    ref:
      apiVersion: v1
      kind: Service
      name: consumer # 3. Where to send the matching events
```
