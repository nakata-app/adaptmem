# Kubernetes deployment example

Reference manifests for running `adaptmem serve` on Kubernetes. Each
file is a starting point — adjust resource limits, image tag, and
host names for your cluster.

## Quickstart

```bash
# 1. Build + push the image (or use the public one once it's published).
docker build -t your-registry.example/adaptmem:0.5.1 ../..
docker push your-registry.example/adaptmem:0.5.1

# 2. Edit deployment.yaml — image: line, replica count, resources.

# 3. Create the secret with your API key. NEVER commit secret.yaml.
cp secret.yaml.example secret.yaml
# edit secret.yaml — fill in your real ADAPTMEM_API_KEY base64
kubectl apply -f secret.yaml

# 4. Apply the rest.
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml          # optional: autoscale on CPU
kubectl apply -f servicemonitor.yaml  # optional: Prometheus Operator

# 5. Check status.
kubectl rollout status deployment/adaptmem
kubectl logs -l app=adaptmem -f
kubectl port-forward svc/adaptmem 7800:7800
curl http://127.0.0.1:7800/healthz
```

## Files

| File | Purpose |
|---|---|
| `configmap.yaml` | Non-secret env vars (rate limit, log level, encoder name) |
| `secret.yaml.example` | API key + optional TLS certs (copy → secret.yaml, fill in) |
| `deployment.yaml` | 3 replicas by default, with /healthz + /readyz probes wired |
| `service.yaml` | ClusterIP exposing 7800; internal traffic only |
| `hpa.yaml` | HorizontalPodAutoscaler — scale 2-10 pods on CPU > 70% |
| `servicemonitor.yaml` | Prometheus Operator: scrape /metrics every 15s |

## Production checklist

Before going live:

- [ ] Set `ADAPTMEM_API_KEY` in `secret.yaml` (or use `--api-keys-file` for RBAC)
- [ ] Mount a `PersistentVolumeClaim` for `--persist-dir` if you want corpora to survive pod restarts (deployment.yaml comments show the wiring; default uses `emptyDir` for dev simplicity)
- [ ] Configure `Ingress` + TLS (cert-manager + Let's Encrypt is the usual path)
- [ ] Point Prometheus / Honeycomb at `/metrics` and the OTLP endpoint
- [ ] Set `OTEL_EXPORTER_OTLP_ENDPOINT` in configmap.yaml when ready to ship traces
- [ ] Tighten `ADAPTMEM_RATE_LIMIT` for public exposure (default 120/minute is generous)
- [ ] Pin the image tag (don't use `latest` in production)

## Helm chart

Not provided yet. The manifests above are the "raw kubectl" path. A
Helm chart that templates the same shape lands in v0.6 — until then,
copy these files into your chart's `templates/` directory and
parameterise as needed.
