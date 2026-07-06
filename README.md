# LLM Billing Proxy

HTTP proxy that sits between end users and LLM servers and bills them per token.

## Requirements

* Python >=3.11
* Python packages from [pyproject.toml](pyproject.toml) (dependencies)

## Quick start

```sh
python3 -m venv venv
source venv/bin/activate

python3 -m llmproxy
```

The SQLite schema is created automatically on first run (and by
`llmproxyctl`). You can pre-create it with `sqlite3 db.sqlite <
llmproxy/schema.sql`, but that step is optional; the schema script is
idempotent.

## Configuration

See [llmproxy/config.toml](llmproxy/config.toml) for an example configuration file.
The program will update the list of configured backends from the config file
on SIGHUP.

Each backend may define `max_model_len`, the real context limit of the
deployment in tokens (prompt plus completion). This value is exposed through
`/v1/models` so clients can avoid sending requests that exceed the backend
limit. It should match the backend deployment setting (for example vLLM
`--max-model-len`), not just the public model card.

```sh
pkill -f -HUP 'python3? .*llmproxy'

# or in Docker
docker kill -s=SIGHUP CONTAINER
```

## Monitoring (Prometheus metrics)

The proxy exposes Prometheus-compatible metrics at `/metrics` by default.
This endpoint can be scraped by Prometheus to monitor request volume,
error rates, backend latency, and token usage — enabling alerts when
traffic spikes or error rates increase.

### Configuration

Metrics are controlled by the `[metrics]` section in
[config.toml](llmproxy/config.toml):

```toml
[metrics]
enabled = true        # set to false to disable the /metrics endpoint
path = "/metrics"     # path under which metrics are served
```

### Available metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `llmproxy_requests_total` | Counter | `method`, `path`, `status` | Total HTTP requests received |
| `llmproxy_request_duration_seconds` | Histogram | `method`, `path`, `status` | End-to-end request latency |
| `llmproxy_active_requests` | Gauge | — | In-flight requests |
| `llmproxy_backend_requests_total` | Counter | `model`, `status` | Requests forwarded to backends |
| `llmproxy_backend_duration_seconds` | Histogram | `model` | Backend response latency |
| `llmproxy_backend_errors_total` | Counter | `model`, `error_type` | Backend errors (timeout, connection, client_error) |
| `llmproxy_tokens_total` | Counter | `model`, `type` | Tokens processed (prompt, completion, embedding) |
| `llmproxy_audio_seconds_total` | Counter | `model` | Seconds of audio transcribed |

### Quick start (Kubernetes + Prometheus Operator)

If you run Prometheus Operator (kube-prometheus-stack), the full path
from deploy to Grafana is:

1. **Deploy** the new llmproxy version — the `/metrics` endpoint is
   active on port 8080 by default.

2. **Create a ServiceMonitor** so Prometheus Operator automatically
   starts scraping:

   ```sh
   # Check your Prometheus selector first:
   kubectl get prometheus -A -o jsonpath='{.items[*].spec.serviceMonitorSelector}'

   # Apply the ServiceMonitor (adjust the release label to match):
   kubectl apply -f k8s/servicemonitor.yaml
   ```

   If llmproxy has no Kubernetes Service, use the PodMonitor instead:

   ```sh
   kubectl apply -f k8s/podmonitor.yaml
   ```

3. **Verify** Prometheus sees the target:

   ```sh
   # Target should appear as "up":
   kubectl port-forward -n monitoring svc/prometheus-operated 9090:9090
   # Open http://localhost:9090/targets → look for "llmproxy"
   ```

4. **Import the Grafana dashboard**:

   - Grafana → Dashboards → New → Import → Upload JSON file
   - Select [grafana/dashboard.json](grafana/dashboard.json)
   - Choose your Prometheus datasource
   - The dashboard has 10 panels: request rate, error rate, status codes,
     active requests, latency percentiles, backend latency/errors, token usage

### Prometheus scrape config (standalone, non-Operator)

If you run Prometheus without the Operator, add a scrape job to your
`prometheus.yml` instead:

```yaml
scrape_configs:
  - job_name: "llmproxy"
    metrics_path: /metrics
    static_configs:
      - targets: ["llm-billing-proxy:8080"]
```

See [examples/prometheus-scrape.yml](examples/prometheus-scrape.yml) for
a complete example.

### Alert rules

Example alert rules for high request volume and high error rates are
provided in [examples/prometheus-alerts.yml](examples/prometheus-alerts.yml).
Key alerts:

* **LLMProxyHighRequestRate** — warning when request rate exceeds 100 req/s
* **LLMProxyVeryHighRequestRate** — critical when request rate exceeds 500 req/s
* **LLMProxyHighErrorRate** — warning when 5xx error rate exceeds 5%
* **LLMProxyCriticalErrorRate** — critical when 5xx error rate exceeds 20%
* **LLMProxyBackendErrors** — per-model backend errors
* **LLMProxyHighBackendLatency** — p95 backend latency above 30s

### Grafana dashboard

A pre-built Grafana dashboard is available at
[grafana/dashboard.json](grafana/dashboard.json).
Import it via Dashboards → New → Import → Upload JSON file.

## Testing

```sh
python3 -m unittest
```

## Building the image

To build the Docker image, run the following command:

```sh
docker build -t ghcr.io/comtegra/llmproxy:master .
```

## Deployment

The following instructions assume that you already have a Docker compose
repository and file.

### SQLite

1. Create an SQLite database according to
   [llmproxy/schema.sql](llmproxy/schema.sql). This is optional: the proxy
   creates the schema automatically on first run. To pre-create it (the script
   is idempotent):

```sh
sqlite3 db.sqlite < llmproxy/schema.sql
```

2. Copy [llmproxy/config.toml](llmproxy/config.toml) to your repository. A good
relative path would be `secrets/llm-billing-proxy-config.toml`.
3. Add appropriate entries to your compose file's `services` and `secrets`
sections. There's a template [compose.yml](compose.yml) in this repository.
4. Bring the new service up and verify it started correctly
(e.g `docker compose up llm-billing-proxy`).

### MongoDB

1. Create a MongoDB user with the following privileges (see section
[Database](#database) below for JS snippets):
  * `{resource: {db: "cgc", collection: "api_keys"}, actions: ["find"]}`
  * `{resource: {db: "billing", collection: "events_oneoff"}, actions: ["insert"]}`
2. Copy [llmproxy/config.toml](llmproxy/config.toml) to your repository. A good
relative path would be `secrets/llm-billing-proxy-config.toml`.
3. In the config edit `uri` in section `db` so that it includes credentials for
the Mongo user (e.g. `mongodb://myuser:mypass@host:27017/?authSource=cgc`).
4. Add appropriate entries to your compose file's `services` and `secrets`
sections. There's a template [compose.yml](compose.yml) in this repository.
5. Bring the new service up and verify it started correctly
(e.g `docker compose up llm-billing-proxy`).

## Testing

Sample data can be added to an SQLite database by using [sample.sql](sample.sql).
Ditto for MongoDB in [sample.js](sample.js).

After loading sample data you may run a query like this:

```sh
curl -v -H'Content-Type: application/json' -H'Authorization: Bearer token2' \
    -d'{"messages": [{"role": "system", "content": "You are an assistant."}, {"role": "user", "content": "Write a limerick about python exceptions"}], "model": "llama31-70b", "stream": true}' \
    'http://localhost:8080/v1/chat/completions'
```

## Database

If using SQLite see [llmproxy/schema.sql](llmproxy/schema.sql).
Otherwise read on.

This program uses MongoDB for authentication and completion logging.
The schema is compatible with Comtergra GPU Core.
The MongoDB user needs the following privileges:

```js
db.getSiblingDB("cgc").createRole({
  role: "apiKeysReader",
  privileges: [
    {
      resource: {db: "cgc", collection: "api_keys"},
      actions: ["find"],
    },
  ],
  roles: [],
});
db.getSiblingDB("billing").createRole({
  role: "completionBillingWriter",
  privileges: [
    {
      resource: {db: "billing", collection: "events_oneoff"},
      actions: ["insert"],
    },
  ],
  roles: [],
});

db.getSiblingDB("cgc").createUser({
  user: "llm-billing-proxy",
  pwd: "mypass",
  roles: [
    { role: "completionBillingWriter", db: "cgc" },
    { role: "apiKeysReader", db: "billing" },
  ],
});
```

## Authentication

Users authenticate via bearer tokens.
Tokens are stored in the database as SHA256 hashes.
They can be generated by the following command:

```sh
python3 -c 'import hashlib, secrets; print("Token:", t:=secrets.token_urlsafe(64)); print("Hash:", hashlib.sha256(t.encode()).hexdigest())'
```

If using SQLite see [llmproxy/schema.sql](llmproxy/schema.sql).
Otherwise read on.

When a user attempts to authenticate, the following query is performed to the
`cgc.api_keys` collection.

```js
{
    "access_level": "LLM",
    "secret": TOKEN-HASH,
    "$or": [
        {"date_expiry": {"$gt": new Date()}},
        {"date_expiry": null},
    ],
}
```

`TOKEN-HASH` is replaced with the SHA256 hash of the bearer token.

The documents are required to have an additional field: `user_id`.

## Completion logging

If using SQLite see [llmproxy/schema.sql](llmproxy/schema.sql).
Otherwise read on.

When a user performs a completion, two events (prompt token and completion
token counts) are inserted into the collection `billing.events_oneoff`.
These documents have the following fields:

* `date_created` -- date and time when the request finished processing
* `user_id` -- `user_id` of the API key
* `api_key_id` -- id of the API key
* `product` -- a string in the following format: `MODEL/DEVICE/TYPE`, where
    * `MODEL` is the name of the backend
    * `DEVICE` is the name of the GPU where the model runs
    * `TYPE` is `prompt` or `completion`
* `quantity` -- token count
* `request_id` -- request ID to correlate prompt/completion counts
