import contextlib
import json

import aiohttp
import yarl


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


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
@contextlib.asynccontextmanager
async def request(f_req, body_transform=None):
    app = f_req.app

    if f_req.content_type == "application/json":
        try:
            f_body = await f_req.json()
        except json.decoder.JSONDecodeError as e:
            raise aiohttp.web.HTTPBadRequest(text="JSON decode error: %s" % e)
    elif f_req.content_type == "multipart/form-data":
        f_body = await f_req.post()
    else:
        raise aiohttp.web.HTTPUnsupportedMediaType()

    try:
        b_name = f_body["model"]
        b_cfg = app["config"].get("backends", {})[b_name]
    except KeyError:
        raise aiohttp.web.HTTPUnauthorized(text="Incorrect model")

    app.logger.debug("Frontend request: request_id=%s path=%s model=%s",
        f_req["request_id"], f_req.rel_url.path, b_name)

    b_url = yarl.URL(b_cfg["url"]) / str(f_req.rel_url)[1:]
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
        # Manually add fields to FormData as aiohttp can't serialize FileField
        d = aiohttp.FormData()
        for key, value in b_body.items():
            if isinstance(value, aiohttp.web.FileField):
                d.add_field(key, value.file, content_type=value.content_type,
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
        async with app["client"].post(
                b_url, headers=b_hdrs, data=b_body, ssl=ssl,
                timeout=timeout) as b_res:
            yield b_res, b_name, b_cfg
    except aiohttp.ServerTimeoutError as e:
        app.logger.error("Backend timeout: request_id=%s model=%s error=%s",
            f_req["request_id"], b_name, e)
        raise aiohttp.web.HTTPGatewayTimeout() from e
    except (aiohttp.ClientConnectorError, aiohttp.ServerConnectionError,
            aiohttp.ClientPayloadError, aiohttp.ClientResponseError,
            aiohttp.InvalidURL) as e:
        app.logger.error("Backend error: request_id=%s model=%s error=%s",
            f_req["request_id"], b_name, e)
        raise aiohttp.web.HTTPBadGateway() from e
    except aiohttp.ClientError as e:
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
