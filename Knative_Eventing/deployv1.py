#!/usr/bin/env python3
import os
import sys
import time
import glob
import yaml
import subprocess
import logging
import json
from typing import List, Optional, Tuple
from kubernetes import client, config
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from kubernetes.client.rest import ApiException

logging.basicConfig(  level=logging.INFO,  format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S' )
logger = logging.getLogger(__name__)

KNATIVE_DIR = "./knative-manifests"
KNATIVE_FILES = [ "eventing-crds.yaml",  "eventing-core.yaml",  "eventing-kafka-controller.yaml",  "eventing-kafka-broker.yaml" ]

KAFKA_DIR = "./Kafka"
KAFKA_FILES = [ "strimzi-operator.yaml","Kafka_cluster_multi_node.yaml" ] #  "Kafka_cluster_single_node.yaml"

KAFKA_CONFIGMAP_FILE = "Kafka_KnativeEventing_conf.yaml"
KAFKA_BROKER_FILE = "Broker_Conf.yaml"

MAX_RETRIES = 3
RETRY_DELAY = 10

class KubernetesDeployer:
    def __init__(self):
        self.k8s_client = None
        self.dyn_client = None
        self.v1 = None
        self._connect()

    def _connect(self): # Connect to Kubernetes cluster 
        for attempt in range(MAX_RETRIES):
            try:
                config.load_kube_config()
                self.k8s_client = client.ApiClient()
                self.dyn_client = DynamicClient(self.k8s_client)
                self.v1 = client.CoreV1Api()
                logger.info("Connected to Kubernetes cluster")
                return
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                time.sleep(RETRY_DELAY)
        logger.error("Failed to connect to Kubernetes cluster")
        sys.exit(1)
######################################################################################################################################################################################################################################
    def ensure_namespace_ready(self, namespace: str, timeout: int = 300) -> bool: # check if the name space define is ready 
        logger.info(f"Ensuring namespace '{namespace}' is ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                ns = self.v1.read_namespace(name=namespace)
                if ns.status.phase == "Terminating": #force the clean up, It is a tipical situation after a system re deployment
                    logger.warning(f"Namespace '{namespace}' is terminating. Force cleaning...")
                    self._force_delete_namespace(namespace) # allow a better security and prevent issues
                    self._create_namespace(namespace) # re create the name space because it was in terminating state and after delated
                    return True
                elif ns.status.phase == "Active": #ok name space is present
                    logger.info(f"Namespace '{namespace}' is active")
                    return True
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    logger.info(f"Creating namespace '{namespace}'...")
                    self._create_namespace(namespace) #create the name space
                    return True
                else:
                    logger.error(f"Error checking namespace: {e}")
            time.sleep(5)
        logger.error(f"Timeout ensuring namespace '{namespace}'")
        return False
    
    def _create_namespace(self, namespace: str) -> bool: #create a name space from scratch
        for attempt in range(MAX_RETRIES):
            try:
                ns_metadata = client.V1ObjectMeta(name=namespace)
                ns_body = client.V1Namespace(metadata=ns_metadata)
                self.v1.create_namespace(body=ns_body)
                logger.info(f"Namespace '{namespace}' created")
                return True
            except client.exceptions.ApiException as e:
                if e.status == 409:  # Already exists
                    logger.info(f"Namespace '{namespace}' already exists")
                    return True
                logger.warning(f"Create namespace attempt {attempt + 1} failed: {e}")
                time.sleep(RETRY_DELAY)
        logger.error(f"Failed to create namespace '{namespace}'")
        return False
    
    def _force_delete_namespace(self, namespace: str) -> bool:
        logger.info(f"Force deleting namespace '{namespace}'...")
        try: #finalizer are resource dependency constrange that can create issue in the resource delation
            cmd = [  "kubectl", "patch", "namespace", namespace, "-p", '{"metadata":{"finalizers":[]}}',"--type=merge" ] #empty the finalizer queue for the namespace
            subprocess.run(cmd, capture_output=True, text=True, check=False)
            try: #use api as an alternative 
                ns = self.v1.read_namespace(name=namespace)
                ns.metadata.finalizers = []
                self.v1.replace_namespace(name=namespace, body=ns) #sed to completely replace (update) an existing Namespace object in your cluster, cleaning the finalizer 
                logger.info(f"Removed finalizers from namespace '{namespace}'")
            except Exception as e:
                logger.warning(f"API finalizer removal failed: {e}")
            return self._wait_for_namespace_deletion(namespace)
            
        except Exception as e:
            logger.error(f"Force delete failed: {e}")
            return False
    
    def _wait_for_namespace_deletion(self, namespace: str, timeout: int = 180) -> bool:
        logger.info(f"Waiting for namespace '{namespace}' to be deleted...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.v1.read_namespace(name=namespace)
                elapsed = int(time.time() - start_time)
                if elapsed % 10 == 0:
                    logger.debug(f"[{elapsed}s] Namespace still exists...")
                time.sleep(5)
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    logger.info(f"Namespace '{namespace}' deleted")
                    return True
                else:
                    logger.error(f"Error checking namespace: {e}")
                    return False
        logger.warning(f"Timeout waiting for namespace '{namespace}' deletion")
        return False
#######################################################################################################################################################################################################################################
    def apply_manifests(self, manifests_dir: str, file_list: List[str] = None, force: bool = False) -> bool: #apply k8s yaml manifest 
        logger.info(f"Applying manifests from: {manifests_dir}")
        files_to_process = self._get_manifest_files(manifests_dir, file_list)
        if not files_to_process:
            logger.warning(f"No manifest files found in {manifests_dir}")
            return True
        success = True
        for filename in files_to_process: #apply all the yaml manifests find out 
            if not self._apply_manifest_file(manifests_dir, filename, force):
                success = False
        
        return success
    
    def _get_manifest_files(self, manifests_dir: str, file_list: List[str] = None) -> List[str]:
        if file_list:
            return file_list
        if not os.path.exists(manifests_dir):
            logger.warning(f"Directory {manifests_dir} not found")
            return []
        files = glob.glob(os.path.join(manifests_dir, "*.yaml")) + \
                glob.glob(os.path.join(manifests_dir, "*.yml"))
        return sorted([os.path.basename(f) for f in files])
    
    def _apply_manifest_file(self, manifests_dir: str, filename: str, force: bool) -> bool: #apply single manifest file
        file_path = os.path.join(manifests_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found")
            return True
        logger.info(f"Processing: {filename}")
        try:
            with open(file_path, 'r') as f:
                docs = list(yaml.safe_load_all(f))
            for doc in docs: # manifest yaml contain many resources splitted by "---", each one is a doc item
                if not doc or 'kind' not in doc:
                    continue
                if not self._apply_manifest_doc(doc, force):
                    return False
                time.sleep(0.5) 
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return False
        
        return True
    
    def _apply_manifest_doc(self, doc: dict, force: bool) -> bool: #operation executed for each resource 
        kind = doc['kind']
        api_version = doc['apiVersion']
        name = doc['metadata']['name']
        namespace = doc['metadata'].get('namespace', 'default')
        try: #dyn_client because standard client required the resource api already install!!! here we work with crd (api are not present by deafault )
            resource_api = self.dyn_client.resources.get(api_version=api_version, kind=kind)
        except ResourceNotFoundError:
            logger.error(f"Resource kind '{kind}' not supported")
            return False
        
        for attempt in range(MAX_RETRIES):
            try:
                exists = False
                try:
                    if resource_api.namespaced: # is resource global or bounded to a name space
                        resource_api.get(name=name, namespace=namespace)
                    else:
                        resource_api.get(name=name)
                    exists = True
                except ApiException as e:
                    if e.status != 404:
                        raise
                
                if exists:
                    # For CRDs and critical resources, skip if they exist 
                    if kind in ['CustomResourceDefinition', 'Broker', 'Kafka', 'KafkaNodePool']:
                        # Check if it's truly ready (for CRDs)
                        if kind == 'CustomResourceDefinition':
                            if self._is_crd_ready(name):
                                logger.debug(f"CRD {name} already ready, skipping")
                                return True
                            else:
                                logger.info(f"CRD {name} exists but not ready, waiting...")
                                time.sleep(10)
                                continue
                        else:
                            logger.debug(f" Skipping existing {kind}/{name}")
                            return True
                    else:
                        logger.debug(f"⏭Skipping existing {kind}/{name}")
                        return True
                
                # Create resource
                logger.debug(f"Creating {kind}/{name} in namespace {namespace}")
                #if resource is not present yet start  the creation 
                if resource_api.namespaced:
                    resource_api.create(body=doc, namespace=namespace)
                else:
                    resource_api.create(body=doc)
                logger.debug(f"Created {kind}/{name}")
                if kind == 'CustomResourceDefinition': #if crd are required wait
                    if not self._wait_for_crd_ready(name, timeout=60): #wait for the crd installation and helthy state 
                        logger.warning(f"CRD {name} created but not ready yet")
                return True
                
            except ApiException as e:
                if e.status == 409: 
                    logger.debug(f"Conflict creating {kind}/{name}, may already be creating...")
                    time.sleep(5)
                    continue
                elif "NamespaceTerminating" in str(e):
                    logger.error(f"Namespace '{namespace}' is terminating ")
                    return False
                else:
                    logger.warning(f"API error on {kind}/{name}: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    return False
            except Exception as e:
                logger.error(f"Unexpected error on {kind}/{name}: {e}")
                return False
        
        return False
    
    def _is_crd_ready(self, crd_name: str) -> bool: #check is crd are ready after the deploy
        try:
            cmd = ["kubectl", "get", "crd", crd_name, "-o", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                conditions = data.get('status', {}).get('conditions', [])
                for cond in conditions:
                    if cond.get('type') == 'Established' and cond.get('status') == 'True':
                        return True
            return False
        except:
            return False
    
    def _wait_for_crd_ready(self, crd_name: str, timeout: int = 120) -> bool: #wait until crd are ready and available
        logger.info(f"Waiting for CRD {crd_name} to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self._is_crd_ready(crd_name):
                logger.info(f"CRD {crd_name} is ready")
                return True
            time.sleep(5)
        
        logger.warning(f"Timeout waiting for CRD {crd_name}")
        return False
    ######################################################################################################################################################################################################################################
    def wait_for_pods(self, namespace: str, timeout: int = 600,  pod_filter: str = None, min_pods: int = 1) -> bool:
        logger.info(f"Waiting for pods in namespace '{namespace}'...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                pods = self.v1.list_namespaced_pod(namespace=namespace).items
                if pod_filter: # if there is a pod filter
                    pods = [p for p in pods if pod_filter in p.metadata.name] #get pod only if match with the pod filter specification
                if not pods:
                    elapsed = int(time.time() - start_time)
                    if elapsed % 10 == 0:
                        logger.debug(f"[{elapsed}s] No pods found yet...")
                    time.sleep(5)
                    continue
                ready_pods = 0
                for pod in pods:
                    if self._is_pod_ready(pod):
                        ready_pods += 1
                if ready_pods >= min_pods and ready_pods == len(pods):
                    elapsed = int(time.time() - start_time)
                    logger.info(f"All {len(pods)} pods ready (took {elapsed}s)")
                    return True
                elapsed = int(time.time() - start_time)
                if elapsed % 10 == 0:
                    logger.debug(f"[{elapsed}s] {ready_pods}/{len(pods)} pods ready")
                time.sleep(5)
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    logger.debug(f"Namespace '{namespace}' not found")
                    time.sleep(5)
                    continue
                logger.error(f"Error monitoring pods: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return False
        logger.error(f"Timeout waiting for pods in '{namespace}'")
        return False
    
    def _is_pod_ready(self, pod) -> bool:
        if pod.status.phase != "Running":
            return False
        if not pod.status.conditions:
            return False
        for condition in pod.status.conditions:
            if condition.type == "Ready" and condition.status == "True":
                return True
        return False
#######################################################################################################################################################################################################################################
    def wait_for_configmap(self, name: str, namespace: str,   expected_key: str = None, expected_value: str = None, timeout: int = 120) -> bool:
        logger.info(f"Waiting for ConfigMap '{namespace}/{name}'...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                cm = self.v1.read_namespaced_config_map(name=name, namespace=namespace)
                logger.info(f"ConfigMap '{name}' found")
                if expected_key:
                    if expected_key in cm.data:
                        if expected_value is None or cm.data[expected_key] == expected_value:
                            logger.info(f"ConfigMap key '{expected_key}' verified")
                            return True
                        else:
                            logger.warning(f"Key '{expected_key}' has value '{cm.data[expected_key]}'")
                    else:
                        logger.debug(f"Key '{expected_key}' not yet present")
                else:
                    return True
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    logger.debug("ConfigMap not found yet...")
                else:
                    logger.error(f"Error reading ConfigMap: {e}")
                    return False
            time.sleep(5)
        logger.error(f"Timeout waiting for ConfigMap '{namespace}/{name}'")
        return False
######################################################################################################################################################################################################################################
    def wait_for_broker(self, name: str = "kafka-broker",  namespace: str = "default",  timeout: int = 300) -> bool:
        logger.info(f"Waiting for Broker '{namespace}/{name}'...")
        if not self._wait_for_crd("eventing.knative.dev/v1", "Broker"): #wait for Broker CRD
            return False
        try:
            broker_api = self.dyn_client.resources.get(api_version="eventing.knative.dev/v1", kind="Broker")
        except ResourceNotFoundError:
            logger.error("Broker CRD not found")
            return False
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                broker = broker_api.get(name=name, namespace=namespace)
                conditions = broker.status.get('conditions', []) if broker.status else []
                # Check if ready
                if any(c.get('type') == 'Ready' and c.get('status') == 'True' for c in conditions):
                    elapsed = int(time.time() - start_time)
                    logger.info(f"Broker '{name}' ready (took {elapsed}s)")
                    return True
                # Check for errors
                for c in conditions:
                    if c.get('type') == 'Ready' and c.get('status') == 'False':
                        reason = c.get('reason', '')
                        message = c.get('message', '')
                        if "Failed to create topic" in reason:
                            logger.warning(f"Broker not ready: {reason} - {message}")
                
                elapsed = int(time.time() - start_time)
                if elapsed % 10 == 0:
                    logger.debug(f"[{elapsed}s] Broker not ready yet...")
                time.sleep(5)
                
            except ApiException as e:
                if e.status == 404:
                    logger.debug("Broker not found yet...")
                else:
                    logger.error(f"Error checking broker: {e}")
                    return False
                time.sleep(5)
        
        logger.error(f"Timeout waiting for Broker '{name}'")
        return False
    
    def _wait_for_crd(self, api_version: str, kind: str, timeout: int = 120) -> bool:
        logger.info(f"Waiting for CRD {api_version}/{kind}...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.dyn_client.resources.get(api_version=api_version, kind=kind)
                logger.info(f"CRD {api_version}/{kind} found")
                return True
            except ResourceNotFoundError:
                elapsed = int(time.time() - start_time)
                if elapsed % 10 == 0:
                    logger.debug(f"[{elapsed}s] Waiting for CRD...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error checking CRD: {e}")
                time.sleep(5)
        
        logger.error(f"Timeout waiting for CRD {api_version}/{kind}")
        return False
####################################################################################################################################################################################################################################
    def apply_with_kubectl(self, filepath: str, namespace: str = None) -> bool: #for broker and conf there are some deploy issue if not used kubectl !!!!!!!!!!! IDK WHY
        logger.info(f"Applying {filepath} with kubectl...")
        cmd = ["kubectl", "apply", "-f", filepath]
        if namespace:
            cmd.extend(["-n", namespace])
        for attempt in range(MAX_RETRIES):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"Applied {filepath}")
                    if result.stdout:
                        logger.debug(result.stdout.strip())
                    return True
                if "already exists" in result.stderr.lower() or "unchanged" in result.stdout.lower():
                    logger.info(f"{filepath} already applied")
                    return True
                logger.warning(f"Attempt {attempt + 1} failed: {result.stderr}")
                time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error(f"Error applying {filepath}: {e}")
                return False
        logger.error(f"Failed to apply {filepath}")
        return False
########################################################################################################################################################################################################################################
    def delete_namespace(self, namespace: str) -> bool:
        logger.info(f"Deleting namespace '{namespace}'...")
        try:
            self._clean_namespace_resources(namespace)
            self.v1.delete_namespace(name=namespace)
            return self._wait_for_namespace_deletion(namespace)
            
        except client.exceptions.ApiException as e:
            if e.status == 404:
                logger.info(f"Namespace '{namespace}' already deleted")
                return True
            logger.error(f"Error deleting namespace: {e}")
            return False
    
    def _clean_namespace_resources(self, namespace: str):
        try:
            cmd = ["kubectl", "delete", "all", "--all", "-n", namespace]
            subprocess.run(cmd, capture_output=True, text=True, check=False)
            cmd = ["kubectl", "delete", "crd", "-l", f"namespace={namespace}"]
            subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as e:
            logger.warning(f"Error cleaning namespace resources: {e}")
    
    def delete_all_resources(self): #massive clean up of the system 
        logger.info("Starting cleanup of all resources...")
        self.apply_manifests(KAFKA_DIR, [KAFKA_FILES[1]])
        time.sleep(5)
        self.apply_manifests(KAFKA_DIR, [KAFKA_FILES[0]])
        time.sleep(10)
        self.apply_manifests(KNATIVE_DIR, KNATIVE_FILES)
        time.sleep(10)
        self.delete_namespace("kafka")
        self.delete_namespace("knative-eventing")
        logger.info("Cleanup completed!")
##########################################################################################################################################################################################################################################

def main():
    deployer = KubernetesDeployer() #initialize the deployment class
    if len(sys.argv) > 1 and sys.argv[1] in ["delete", "cleanup", "--delete", "--cleanup", "-d"]:
        deployer.delete_all_resources() # if the user ask by script input arg to delete the cluster start the clean up of the deployment
        return
    logger.info("Starting deployment")
    #--------------------------------------------------------------------------------------------# KNATIVE EVENTING
    logger.info("Phase 1: Deploying Knative Eventing")
    if not deployer.ensure_namespace_ready("knative-eventing"): #check namespace of kantive
        sys.exit(1)
    if not deployer.apply_manifests(KNATIVE_DIR, KNATIVE_FILES, force=False): #deploy knative crd manifests resources into local dir 
        logger.error("Failed to deploy Knative Eventing")
        sys.exit(1)
    logger.info("Waiting for CRDs to register...") #very fast operation, required only passing trough kube api
    time.sleep(10)
    if not deployer.wait_for_pods("knative-eventing", timeout=300): #wait until all the knative pod are available
        logger.error("Knative Eventing pods not ready")
        sys.exit(1)
    #---------------------------------------------------------------------------------------------# KAFKA CLUSTER
    logger.info("Phase 2: Setting up Kafka namespace")
    if not deployer.ensure_namespace_ready("kafka"): #check namespace 
        sys.exit(1)
    if not deployer.apply_manifests(KAFKA_DIR, [KAFKA_FILES[0]]): #deploy Strimzi Operator
        logger.error("Failed to deploy Strimzi Operator")
        sys.exit(1)
    if not deployer.wait_for_pods("kafka", timeout=200, min_pods=1): #wait for the strimzi operator pod creation 
        logger.error("Strimzi Operator not ready")
        sys.exit(1)
    #---------------------------------- After Strimzi operator deploy real cluster ---------------------------------------------------------------#
    time.sleep(5)
    if not deployer.apply_manifests(KAFKA_DIR, [KAFKA_FILES[1]]): #deploy the single or multi kafka cluster conf by strimzi operator
        logger.error("Failed to deploy Kafka Cluster")
        sys.exit(1)
    if not deployer.wait_for_pods("kafka", timeout=600): #wait for the cluster creation 
        logger.error("Kafka pods not ready")
        sys.exit(1)
    if not deployer.wait_for_pods("kafka", timeout=180, pod_filter="entity-operator"): #important ----> cluster is considered complete and ready for the broker alter the entry poit is ready 
        logger.warning("Entity operator not found, continuing anyway...")
    logger.info(" Kafka require extra time for stabilization")
    time.sleep(30)
    #---------------------------------------------------------------------------------------------# KNATIVE KAFKA BROKER
    logger.info("Phase 3: Configuring Knative Kafka Broker")
    configmap_path = os.path.join(KNATIVE_DIR, KAFKA_CONFIGMAP_FILE)  #required knative configuration for kafka cluster as broker support
    if not deployer.apply_with_kubectl(configmap_path, "knative-eventing"): #deploy work well only by kubectl !!!!! IDK WHY 
        logger.error("Failed to apply ConfigMap")
        sys.exit(1)
    if not deployer.wait_for_configmap( name="kafka-broker-config", namespace="knative-eventing", expected_key="bootstrap.servers", expected_value="my-cluster-kafka-bootstrap.kafka.svc:9092" ):
        logger.error("ConfigMap verification failed")
        sys.exit(1)
    time.sleep(5)
    broker_path = os.path.join(KNATIVE_DIR, KAFKA_BROKER_FILE)
    if not deployer.apply_with_kubectl(broker_path, "default"):
        logger.error("Failed to apply Broker")
        sys.exit(1)
    if not deployer.wait_for_broker(timeout=300):
        logger.error("Broker not ready")
        sys.exit(1)
    logger.info("End up of the deployment")
    #deployer._show_status()
if __name__ == "__main__":
    main()
