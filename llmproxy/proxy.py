import contextlib
import json
import os
import time
import traceback

import aiohttp
import yarl
from aiohttp import http_exceptions

from . import errors, metrics, ratelimit


CONTEXT_LENGTH_MARKERS = (
    "context length",
    "maximum context",
    "max_model_len",
    "max model length",
    "prompt too long",
    "input too long",
)


def looks_like_context_length_error(body):
    text = body.decode("utf-8", errors="replace").lower()
    return any(marker in text for marker in CONTEXT_LENGTH_MARKERS)


def _model_error(f_req, message, code):
    # 404 like OpenAI for both an unknown model and a model that exists but is
    # not served on this endpoint ("This is not a chat model...").
    return errors.json_error(f_req, aiohttp.web.HTTPNotFound, message,
        openai_type="invalid_request_error", anthropic_type="not_found_error",
        code=code, param="model")


def _bad_request(f_req, message, param=None):
    # 400 for a body the proxy cannot parse or route on. Without these checks
    # the handler crashed and the client got a 500 (which SDKs retry).
    return errors.json_error(f_req, aiohttp.web.HTTPBadRequest, message,
        openai_type="invalid_request_error",
        anthropic_type="invalid_request_error", param=param)


# What aiohttp and the stdlib raise while reading a body the client got wrong:
# ValueError (bad JSON, text not valid in its charset, an int over Python's
# digit limit, a bad multipart boundary or base64), LookupError (unknown
# charset), RuntimeError (JSON nested too deep, unknown transfer encoding),
# HttpProcessingError (bad multipart part headers), RequestPayloadError (body
# doesn't match its Content-Encoding) and AssertionError (truncated multipart
# body, part without a name).
_MALFORMED_BODY_ERRORS = (ValueError, LookupError, RuntimeError,
    AssertionError, http_exceptions.HttpProcessingError,
    aiohttp.web.RequestPayloadError)


def _is_malformed_body(e):
    # KeyError and IndexError are LookupErrors too, but from aiohttp or json
    # they would be a bug. Those, and anything else (e.g. OSError spooling an
    # upload to disk), stay 500s.
    return (isinstance(e, _MALFORMED_BODY_ERRORS)
        and not isinstance(e, (KeyError, IndexError)))


def _raised_at(e):
    # Where in aiohttp/json it failed, so a bug caught as a "malformed body"
    # can still be told apart from bad input in the logs.
    tb = e.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    code = tb.tb_frame.f_code
    return "%s:%d %s" % (os.path.basename(code.co_filename), tb.tb_lineno,
        code.co_name)


async def _read_body(f_req):
    """Parse the JSON or multipart body, or raise a 400 if it is malformed."""
    multipart = f_req.content_type == "multipart/form-data"
    try:
        body = await (f_req.post() if multipart else f_req.json())
    except BaseException as e:
        # aiohttp only closes the temp files post() spools file parts to
        # after a successful post(). Otherwise they live on in its frame,
        # which this traceback (or an exception chained to it) keeps until
        # the cyclic GC runs. Clearing the frames frees them now, whatever
        # went wrong.
        traceback.clear_frames(e.__traceback__)
        if not _is_malformed_body(e):
            raise
        # Truncated: aiohttp's message can quote a whole line of the body.
        f_req.app.logger.info(
            "Malformed request body: request_id=%s error=%.200r at %s",
            f_req["request_id"], e, _raised_at(e))
        if multipart:
            message = "Malformed multipart/form-data body."
        elif isinstance(e, json.JSONDecodeError):
            message = "JSON decode error: %s" % e
        else:
            # The rest is Python's wording (codecs, digit limits, recursion).
            message = "Could not decode the request body."
        raise _bad_request(f_req, message)

    if not multipart and not isinstance(body, dict):
        raise _bad_request(f_req, "The request body must be a JSON object.")
    return body


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
@contextlib.asynccontextmanager
async def request(f_req, body_transform=None, user=None, path=None, *,
        backend_type):
    """Forward a client request to the backend selected by ``model``.

    ``backend_type`` is the backend type (config.BACKEND_TYPES) the calling
    endpoint serves; a model of any other type is rejected with 404 before
    rate limiting, the backend call and billing.

    ``path`` overrides the path appended to the backend URL. By default the
    frontend path is preserved (vLLM / whisper). File conversion rewrites to
    the marker microservice's ``/api/v1/marker`` instead.
    """
    app = f_req.app

    if f_req.content_type not in ("application/json", "multipart/form-data"):
        raise aiohttp.web.HTTPUnsupportedMediaType()
    f_body = await _read_body(f_req)

    b_name = f_body.get("model")
    # A list or object is unhashable (the lookup below would raise), and a
    # number or bool is never a model name. A missing model (None) falls
    # through to model_not_found.
    if b_name is not None and not isinstance(b_name, str):
        raise _bad_request(f_req,
            "Invalid type for 'model': expected a string.", param="model")
    b_cfg = app["config"].get("backends", {}).get(b_name)
    if b_cfg is None:
        # %r: a missing "model" reads as None, not as a model named 'None'.
        raise _model_error(f_req, "The model %r does not exist." % b_name,
            "model_not_found")

    if b_cfg["type"] != backend_type:
        raise _model_error(f_req,
            "The model %r has type %r and is not supported on %s." %
                (b_name, b_cfg["type"], f_req.rel_url.path),
            "model_not_supported")

    app.logger.debug("Frontend request: request_id=%s path=%s model=%s",
        f_req["request_id"], f_req.rel_url.path, b_name)

    # Forward the path only. Joining the full rel_url would percent-encode a
    # query string into the backend path ("?beta=true" -> "%3Fbeta=true"),
    # which the backend 404s -- Claude Code sends ?beta=true on every request.
    # Query params carry no semantics on these endpoints, so they are dropped.
    # ``path`` lets a handler rewrite (e.g. /v1/files/convert -> api/v1/marker).
    b_path = path if path is not None else f_req.rel_url.path[1:]
    b_url = yarl.URL(b_cfg["url"]) / b_path
    b_hdrs = {"Authorization": "Bearer %s" % b_cfg["token"]}

    b_body = f_body.copy()
    if (m := b_cfg.get("model")) is not None:
        b_body["model"] = m

    if body_transform is not None:
        body_transform(b_body)

    if f_req.content_type == "application/json":
        b_body = json.dumps(b_body)
        b_hdrs["Content-Type"] = "application/json"
    elif f_req.content_type == "multipart/form-data":
        # Manually add fields to FormData as aiohttp can't serialize FileField.
        # FileField.file may be a tempfile._TemporaryFileWrapper, which newer
        # aiohttp payload registries reject -- unwrap to the real file object
        # (or fall back to reading bytes) so audio/PDF uploads forward cleanly.
        d = aiohttp.FormData()
        for key, value in b_body.items():
            if isinstance(value, aiohttp.web.FileField):
                raw = value.file
                if hasattr(raw, "file"):
                    raw = raw.file
                d.add_field(key, raw, content_type=value.content_type,
                    filename=value.filename)
            else:
                d.add_field(key, value)
        b_body = d

    try:
        app.logger.debug("Sending backend request")
        ssl = None if b_cfg.get("verify_ssl", True) else False
        # Per-backend response timeout overrides the global sock_read (e.g. audio
        # transcription is silent for minutes; the global timeout would 504 it).
        timeout = aiohttp.ClientTimeout(
            connect=app["config"]["timeout_connect"],
            sock_read=b_cfg.get("timeout", app["config"]["timeout_read"]))
        b_start = time.monotonic()
        async with ratelimit.slot(f_req, user, b_name, b_cfg):
            async with app["client"].post(
                    b_url, headers=b_hdrs, data=b_body, ssl=ssl,
                    timeout=timeout) as b_res:
                metrics.BACKEND_DURATION_SECONDS.labels(b_name).observe(
                    time.monotonic() - b_start)
                metrics.BACKEND_REQUESTS_TOTAL.labels(
                    b_name, str(b_res.status)).inc()
                yield b_res, b_name, b_cfg
    except aiohttp.ServerTimeoutError as e:
        metrics.BACKEND_DURATION_SECONDS.labels(b_name).observe(
            time.monotonic() - b_start)
        metrics.BACKEND_ERRORS_TOTAL.labels(b_name, "timeout").inc()
        app.logger.error("Backend timeout: request_id=%s model=%s error=%s",
            f_req["request_id"], b_name, e)
        raise aiohttp.web.HTTPGatewayTimeout() from e
    except (aiohttp.ClientConnectorError, aiohttp.ServerConnectionError,
            aiohttp.ClientPayloadError, aiohttp.ClientResponseError,
            aiohttp.InvalidURL) as e:
        metrics.BACKEND_DURATION_SECONDS.labels(b_name).observe(
            time.monotonic() - b_start)
        metrics.BACKEND_ERRORS_TOTAL.labels(b_name, "connection").inc()
        app.logger.error("Backend error: request_id=%s model=%s error=%s",
            f_req["request_id"], b_name, e)
        raise aiohttp.web.HTTPBadGateway() from e
    except aiohttp.ClientError as e:
        metrics.BACKEND_DURATION_SECONDS.labels(b_name).observe(
            time.monotonic() - b_start)
        metrics.BACKEND_ERRORS_TOTAL.labels(b_name, "client_error").inc()
        app.logger.error("HTTP client error: request_id=%s model=%s error=%s",
            f_req["request_id"], b_name, e)
        raise aiohttp.web.HTTPInternalServerError() from e


async def check_response(app, b_name, b_res, expected_status=200,
        request_id=None):
    """Raise on backend responses that don't match the expected status.

    4xx errors are forwarded as-is (client errors from the backend).
    Everything else (5xx, unexpected 2xx/3xx) is masked with a generic 502
    to avoid leaking internals or parsing unexpected response bodies.
    """
    if b_res.status == expected_status:
        return

    body = await b_res.read()

    app.logger.error(
        'Backend "%s" unexpected status: request_id=%s status=%d '
        'content_type=%s body_len=%d body_preview=%s',
        b_name, request_id, b_res.status, b_res.content_type, len(body),
        body[:1024].decode("utf-8", errors="replace"))

    if 400 <= b_res.status < 500:
        exc = aiohttp.web.HTTPBadRequest(content_type=b_res.content_type)
        exc.set_status(b_res.status)
        exc.body = body
        raise exc

    if b_res.status >= 500 and looks_like_context_length_error(body):
        raise aiohttp.web.HTTPUnprocessableEntity(
            text=json.dumps({
                "error": {
                    "message": "Context length exceeded for this model.",
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded",
                },
            }),
            content_type="application/json",
        )

    raise aiohttp.web.HTTPBadGateway()
