"""Prometheus metrics for llmproxy.

Exposes a ``/metrics`` endpoint (via app.py) suitable for scraping by
Prometheus.  The following metric families are provided:

  llmproxy_requests_total{method, path, status}
      Counter — total HTTP requests received by the proxy.

  llmproxy_request_duration_seconds{method, path, status}
      Histogram — end-to-end request latency in seconds.

  llmproxy_active_requests
      Gauge — number of in-flight requests.

  llmproxy_backend_requests_total{model, status}
      Counter — total requests forwarded to backends.

  llmproxy_backend_duration_seconds{model}
      Histogram — backend response latency (time to receive response
      headers, not full body transfer).

  llmproxy_backend_errors_total{model, error_type}
      Counter — backend errors by type (timeout, connection, client_error).

  llmproxy_tokens_total{model, type}
      Counter — tokens processed, labelled by type: prompt, completion,
      embedding.

  llmproxy_audio_seconds_total{model}
      Counter — seconds of audio transcribed.
"""

import time

import aiohttp.web
import prometheus_client
import prometheus_client.core

# All metrics share the default global registry.
_REGISTRY = prometheus_client.core.REGISTRY

# ---------------------------------------------------------------------------
# HTTP request metrics (frontend)
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = prometheus_client.Counter(
    "llmproxy_requests_total",
    "Total number of HTTP requests received by the proxy.",
    labelnames=("method", "path", "status"),
    registry=_REGISTRY,
)

REQUEST_DURATION_SECONDS = prometheus_client.Histogram(
    "llmproxy_request_duration_seconds",
    "End-to-end HTTP request duration in seconds.",
    labelnames=("method", "path", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5,
             10, 30, 60, 120),
    registry=_REGISTRY,
)

ACTIVE_REQUESTS = prometheus_client.Gauge(
    "llmproxy_active_requests",
    "Number of active (in-flight) HTTP requests.",
    registry=_REGISTRY,
)

# ---------------------------------------------------------------------------
# Backend metrics
# ---------------------------------------------------------------------------

BACKEND_REQUESTS_TOTAL = prometheus_client.Counter(
    "llmproxy_backend_requests_total",
    "Total number of requests forwarded to backends.",
    labelnames=("model", "status"),
    registry=_REGISTRY,
)

BACKEND_DURATION_SECONDS = prometheus_client.Histogram(
    "llmproxy_backend_duration_seconds",
    "Backend response latency in seconds (time to response headers).",
    labelnames=("model",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5,
             10, 30, 60, 120),
    registry=_REGISTRY,
)

BACKEND_ERRORS_TOTAL = prometheus_client.Counter(
    "llmproxy_backend_errors_total",
    "Total number of backend errors by type.",
    labelnames=("model", "error_type"),
    registry=_REGISTRY,
)

# ---------------------------------------------------------------------------
# Token / usage metrics
# ---------------------------------------------------------------------------

TOKENS_TOTAL = prometheus_client.Counter(
    "llmproxy_tokens_total",
    "Total number of tokens processed.",
    labelnames=("model", "type"),
    registry=_REGISTRY,
)

AUDIO_SECONDS_TOTAL = prometheus_client.Counter(
    "llmproxy_audio_seconds_total",
    "Total seconds of audio transcribed.",
    labelnames=("model",),
    registry=_REGISTRY,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Known paths — keeps label cardinality bounded.  Anything else is
# collapsed to "unknown" to prevent a malicious or buggy client from
# inflating the label space.
_KNOWN_PATHS = frozenset({
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/models",
    "/v1/embeddings",
    "/v1/audio/transcriptions",
    "/metrics",
})


def _normalize_path(path):
    """Map a request path to a low-cardinality label value."""
    return path if path in _KNOWN_PATHS else "unknown"


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@aiohttp.web.middleware
async def metrics_middleware(req, handler):
    """Record HTTP-level metrics for every request.

    Wrapped around the full middleware + handler chain so that errors
    raised by inner middlewares (auth, CORS, …) are also captured.
    """
    ACTIVE_REQUESTS.inc()
    start = time.monotonic()
    status = 500
    try:
        res = await handler(req)
        status = res.status
        return res
    except aiohttp.web.HTTPException as e:
        status = e.status
        raise
    except Exception:
        # Unhandled exception — aiohttp will turn this into a 500.
        status = 500
        raise
    finally:
        duration = time.monotonic() - start
        ACTIVE_REQUESTS.dec()
        path = _normalize_path(req.rel_url.path)
        labels = (req.method, path, str(status))
        REQUESTS_TOTAL.labels(*labels).inc()
        REQUEST_DURATION_SECONDS.labels(*labels).observe(duration)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

async def metrics_handler(req):
    """Expose metrics in the Prometheus exposition format."""
    data = prometheus_client.generate_latest(_REGISTRY)
    return aiohttp.web.Response(
        body=data,
        headers={"Content-Type": prometheus_client.CONTENT_TYPE_LATEST},
    )
