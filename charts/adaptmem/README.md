# adaptmem Helm chart

Templated equivalent of [`examples/k8s/`](../../examples/k8s/).

## Install

```bash
# from the chart dir
helm install adaptmem ./charts/adaptmem

# from a custom values file
helm install adaptmem ./charts/adaptmem --values my-values.yaml

# enable auth + persistence + tracing in one go
helm install adaptmem ./charts/adaptmem \
    --set auth.enabled=true \
    --set auth.apiKey=$(openssl rand -hex 32) \
    --set persistence.enabled=true \
    --set persistence.size=20Gi \
    --set config.otelEndpoint=https://api.honeycomb.io
```

## Upgrade

```bash
helm upgrade adaptmem ./charts/adaptmem -f my-values.yaml
```

## Uninstall

```bash
helm uninstall adaptmem
```

## Values

See [`values.yaml`](values.yaml) for the full list. Highlights:

| Path | Default | What it does |
|---|---|---|
| `image.repository` | `nakata-app/adaptmem` | Container image |
| `image.tag` | `""` (uses Chart.appVersion) | Image tag — pin in production |
| `replicaCount` | `3` | Pod count when autoscaling is off |
| `config.baseModel` | `all-MiniLM-L6-v2` | Encoder model the daemon loads |
| `config.rateLimit` | `120/minute` | Per-IP cap; tighten for public APIs |
| `config.otelEndpoint` | `""` | Set to enable OTel tracing |
| `auth.enabled` | `false` | Toggle Bearer-token auth |
| `auth.apiKey` | `""` | Single-key mode; chart creates the Secret |
| `auth.existingSecret` | `""` | Use a pre-existing Secret instead |
| `persistence.enabled` | `false` | PVC for `corpora.db` (otherwise emptyDir) |
| `persistence.size` | `10Gi` | PVC size |
| `autoscaling.enabled` | `true` | HPA on CPU + memory |
| `autoscaling.maxReplicas` | `10` | Hard cap for HPA |
| `serviceMonitor.enabled` | `false` | kube-prometheus-stack scrape |
| `ingress.enabled` | `false` | Expose via Ingress |

## Production checklist

- [ ] Pin `image.tag` to a specific version
- [ ] Set `auth.enabled: true` with a strong key (`openssl rand -hex 32`)
- [ ] Enable `persistence` for corpus durability
- [ ] Configure `ingress` with TLS via cert-manager
- [ ] Set `config.otelEndpoint` + secret with `OTEL_EXPORTER_OTLP_HEADERS`
- [ ] Tighten `config.rateLimit` if exposed publicly
- [ ] Enable `serviceMonitor` for Prometheus scrape (if you run kube-prometheus-stack)

## Notes

- `appVersion` in `Chart.yaml` tracks the adaptmem package version.
- Bump `version` (chart version) when you change the chart shape, even
  if `appVersion` stays the same.
- `image.tag` empty → falls back to `appVersion` so the chart and the
  package stay in sync by default.
