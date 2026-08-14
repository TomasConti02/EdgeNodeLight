#!/bin/bash
set -e  # Stop script on error

NAMESPACE="istio-system"

echo "==> Creating namespace $NAMESPACE (if not exists)"
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

echo "==> Adding Helm repo for Prometheus"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

echo "==> Installing Prometheus"
helm upgrade --install prometheus prometheus-community/prometheus \
  --namespace $NAMESPACE \
  --set server.persistentVolume.enabled=false \
  --set alertmanager.enabled=false \
  --set pushgateway.enabled=false

echo "==> Adding Helm repo for Grafana"
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

echo "==> Installing Grafana with preconfigured Prometheus datasource"
helm upgrade --install grafana grafana/grafana \
  --namespace $NAMESPACE \
  --set persistence.enabled=false \
  --set adminPassword="admin" \
  --set service.port=80 \
  --set "datasources.datasources\.yaml.apiVersion=1" \
  --set "datasources.datasources\.yaml.datasources[0].name=Prometheus" \
  --set "datasources.datasources\.yaml.datasources[0].type=prometheus" \
  --set "datasources.datasources\.yaml.datasources[0].url=http://prometheus-server.$NAMESPACE:80" \
  --set "datasources.datasources\.yaml.datasources[0].access=proxy" \
  --set "datasources.datasources\.yaml.datasources[0].isDefault=true"

echo "==> Adding Helm repo for Kiali"
helm repo add kiali https://kiali.org/helm-charts
helm repo update

echo "==> Installing Kiali"
helm upgrade --install kiali kiali/kiali-server \
  --namespace $NAMESPACE \
  --set auth.strategy="anonymous" \
  --set external_services.prometheus.url="http://prometheus-server.$NAMESPACE:80" \
  --set external_services.grafana.url="http://grafana.$NAMESPACE:80" \
  --set external_services.grafana.internal_url="http://grafana.$NAMESPACE:80"

echo "==> Waiting for all pods to be ready (max 300 seconds)"
kubectl wait --for=condition=ready pod --all -n $NAMESPACE --timeout=300s

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "To access Kiali, run in another terminal:"
echo "  kubectl port-forward svc/kiali -n $NAMESPACE 20001:20001"
echo "Then open your browser at http://localhost:20001"
echo ""
echo "To access Grafana (admin/admin):"
echo "  kubectl port-forward svc/grafana -n $NAMESPACE 3000:80"
echo "Then open http://localhost:3000"
echo ""
echo "To access Prometheus directly:"
echo "  kubectl port-forward svc/prometheus-server -n $NAMESPACE 9090:80"
echo "Then open http://localhost:9090"
echo ""
echo "To generate test traffic (if you have a request_script.py):"
echo "  for i in {1..60}; do python3 request_script.py; sleep 1; done"
