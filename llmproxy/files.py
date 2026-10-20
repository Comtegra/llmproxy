import json

import aiohttp
import yarl

from . import auth, billing, conversion_id, errors, metrics, provenance, proxy


# Marker Datalab-compat path (PDF -> Markdown). Not path-preserving:
# the public route is /v1/files/convert.
MARKER_PATH = "api/v1/marker"

OBJECT = "file.conversion"

# Seconds a client should wait before polling a background conversion again.
RETRY_AFTER = "5"

# Sent on every poll: the backend answers a completed job with
# delivery_claimed=true exactly once per job, which is when the proxy bills.
# The proxy keeps no state, so this one-time claim is what makes "bill at the
# first pickup" hold across repeated polls, restarts and replicas.
CLAIM_HEADER = "X-Delivery-Claim"

# Multipart values are strings; accept the spellings clients send for a bool.
_BACKGROUND = {"true": True, "1": True, "false": False, "0": False}

# Fields of the backend's poll response the converted text can never be
# stored under (mirrors the backend's choice of the content key).
_RESERVED_KEYS = frozenset({"status", "success", "images", "page_count",
    "output_format", "provenance", "delivery_claimed", "error_code", "error",
    "request_id"})


def _bad_background(f_req):
    return errors.json_error(f_req, aiohttp.web.HTTPBadRequest,
        "Invalid value for 'background': expected 'true' or 'false'.",
        openai_type="invalid_request_error",
        anthropic_type="invalid_request_error", param="background")


def _not_found(f_req):
    # One answer for a malformed id, another key's id, an unknown backend and
    # a job the backend no longer has: telling them apart would let a client
    # probe which ids exist.
    return errors.json_error(f_req, aiohttp.web.HTTPNotFound,
        "No file conversion found with that id.",
        openai_type="invalid_request_error", anthropic_type="not_found_error",
        code="conversion_not_found")


def prepare_marker_body(f_req, body):
    """Validate upload fields and strip llmproxy-only form keys.

    Marker expects ``file`` (and optional ``output_format``); ``model`` is only
    used by the proxy for backend selection / billing. Returns whether the
    client asked for a background conversion. ``background`` is always sent
    explicitly: the backend's own default (MARKER_DATALAB_COMPAT_SYNC) may be
    async for other clients, and a sync request must not inherit it.
    """
    file_field = body.get("file")
    if not isinstance(file_field, aiohttp.web.FileField):
        raise aiohttp.web.HTTPUnprocessableEntity(text="file is required")

    value = body.pop("background", "false")
    background = _BACKGROUND.get(value.lower()) \
        if isinstance(value, str) else None
    if background is None:
        raise _bad_background(f_req)
    body["background"] = "true" if background else "false"

    body.pop("model", None)
    if "output_format" not in body:
        body["output_format"] = "markdown"
    return background


def page_count_from_response(data):
    """Return a billable page count, or None if missing/unusable."""
    pages = data.get("page_count")
    if isinstance(pages, bool) or not isinstance(pages, int) or pages <= 0:
        return None
    return pages


def _poll_json(app, b_name, f_req, body):
    """Decode the backend's poll answer (a JSON object), or raise 502."""
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        data = None
    if not isinstance(data, dict):
        app.logger.error(
            "File-convert backend %s returned a non-JSON poll answer: "
            "request_id=%s", b_name, f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend returned invalid JSON")
    return data


def _ai_headers(app, b_res):
    """AI Act marking for a response carrying converted text.

    Header-only marking, same carve-out pattern as audio; Marker's own
    AI-processed headers are forwarded when present.
    """
    hdrs = provenance.headers(app["config"])
    for h in ("X-AI-Processed", "X-AI-Generated"):
        if h in b_res.headers and h not in hdrs:
            hdrs[h] = b_res.headers[h]
    return hdrs


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def convert(f_req):
    """POST /v1/files/convert -- PDF upload to Markdown via marker.

    Synchronous by default. With ``background=true`` it answers 202 with a
    conversion id as soon as the backend has queued the job; the result is
    picked up (and billed) through GET /v1/files/convert/{id}.
    """
    app = f_req.app

    user = await auth.require_auth(f_req)

    background = False

    def transform(body):
        nonlocal background
        background = prepare_marker_body(f_req, body)

    async with proxy.request(
            f_req, transform, user=user, path=MARKER_PATH,
            backend_type="conversion") as (b_res, b_name, b_cfg):
        app.logger.debug("Backend request completed")

        await proxy.check_response(app, b_name, b_res,
            request_id=f_req["request_id"])

        body = await b_res.content.read()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            app.logger.error(
                "File-convert backend %s returned non-JSON: request_id=%s",
                b_name, f_req["request_id"])
            raise aiohttp.web.HTTPBadGateway(
                text="File conversion backend returned invalid JSON")

        # Datalab Marker returns HTTP 200 even on conversion failure, with
        # success=false. Do not bill those.
        if not isinstance(data, dict) or data.get("success") is not True:
            app.logger.error(
                "File-convert backend %s reported failure: request_id=%s "
                "body_preview=%s",
                b_name, f_req["request_id"],
                body[:1024].decode("utf-8", errors="replace"))
            exc = aiohttp.web.HTTPUnprocessableEntity(
                content_type="application/json")
            exc.body = body
            raise exc

        if background:
            return _accepted(f_req, user, b_name, b_cfg, data)

        # Page count is the only billable quantity. Same fail-loud policy as
        # transcription duration: never emit a zero/corrupt billing row.
        pages = page_count_from_response(data)
        if pages is None:
            app.logger.error(
                "File-convert backend %s returned no billable page_count: "
                "request_id=%s", b_name, f_req["request_id"])
            raise aiohttp.web.HTTPBadGateway(
                text="File conversion backend returned no page_count")

        f_hdrs = {"Content-Type":
            b_res.headers.get("Content-Type", "application/json")}
        # Forward Marker's JSON (may already include its own provenance
        # object) byte-for-byte.
        f_hdrs.update(_ai_headers(app, b_res))

        f_res = aiohttp.web.Response(body=body, headers=f_hdrs)

        await billing.record(f_req, user, {
            "%s/%s/conversion" % (b_name, b_cfg["device"]): pages,
        })

        app.logger.info("Client used: %d page(s) of %s", pages, b_name)
        metrics.FILE_CONVERSIONS_TOTAL.labels(b_name).inc(pages)

        return f_res


def _accepted(f_req, user, b_name, b_cfg, data):
    """202 for a conversion the backend queued. Nothing is billed yet."""
    app = f_req.app

    b_request_id = data.get("request_id")
    if not isinstance(b_request_id, str) \
            or not conversion_id.REQUEST_ID_RE.fullmatch(b_request_id):
        # A backend that ignores background=true converts synchronously and
        # answers without a request id: it predates background conversions.
        app.logger.error(
            "File-convert backend %s returned no usable request_id for a "
            "background conversion (backend does not support background=true; "
            "update the file-processor): request_id=%s",
            b_name, f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend does not support background "
                "conversion")

    job_id = conversion_id.make(b_cfg["token"], user["id"], b_name,
        b_request_id)
    app.logger.info(
        "Background conversion queued: id=%s request_id=%s "
        "backend_request_id=%s model=%s",
        job_id, f_req["request_id"], b_request_id, b_name)

    return aiohttp.web.json_response({
        "id": job_id,
        "object": OBJECT,
        "model": b_name,
        "status": "queued",
    }, status=202, headers={
        "Location": "/v1/files/convert/%s" % job_id,
        "Retry-After": RETRY_AFTER,
    })


def _content_key(data):
    """Key the backend stored the converted text under (its output_format)."""
    fmt = data.get("output_format")
    if isinstance(fmt, str) and fmt.lower() not in _RESERVED_KEYS:
        return fmt.lower()
    return "markdown"


async def conversion(f_req):
    """GET /v1/files/convert/{id} -- state or result of a background conversion.

    Only the API key that submitted the conversion can read it. The result is
    billed the first time it is delivered; later polls return it unbilled.
    Polls do not count against the model's rpm/concurrency limits: a client
    polling every few seconds must not use up its budget for submitting.
    """
    app = f_req.app

    user = await auth.require_auth(f_req)

    job_id = f_req.match_info["id"]
    parsed = conversion_id.parse(job_id)
    if parsed is None:
        raise _not_found(f_req)
    b_name = parsed.backend
    b_cfg = app["config"].get("backends", {}).get(b_name)
    if b_cfg is None or b_cfg["type"] != "conversion" \
            or not conversion_id.verify(b_cfg.get("token"), user["id"],
                parsed):
        raise _not_found(f_req)

    b_url = yarl.URL(b_cfg["url"]) / MARKER_PATH / parsed.request_id
    b_hdrs = {
        "Authorization": "Bearer %s" % b_cfg["token"],
        CLAIM_HEADER: "true",
    }
    ssl = None if b_cfg.get("verify_ssl", True) else False
    # The global read timeout, not the backend's: a poll answers at once, the
    # backend's long timeout is for waiting out a synchronous conversion.
    timeout = aiohttp.ClientTimeout(
        connect=app["config"]["timeout_connect"],
        sock_read=app["config"]["timeout_read"])

    with proxy.backend_errors(f_req, b_name) as b_start:
        async with app["client"].get(b_url, headers=b_hdrs, ssl=ssl,
                timeout=timeout) as b_res:
            proxy.observe_backend_response(b_name, b_res, b_start)
            await proxy.check_response(app, b_name, b_res,
                request_id=f_req["request_id"])
            body = await b_res.read()
            data = _poll_json(app, b_name, f_req, body)
            ai_headers = _ai_headers(app, b_res)

    job = {"id": job_id, "object": OBJECT, "model": b_name}
    status = data.get("status")

    if status == "processing":
        return aiohttp.web.json_response({**job, "status": "in_progress"},
            headers={"Retry-After": RETRY_AFTER})

    if status == "failed":
        if data.get("error_code") == "JOB_NOT_FOUND":
            raise _not_found(f_req)
        message = data.get("error")
        return aiohttp.web.json_response({**job, "status": "failed",
            "error": {
                "message": message if isinstance(message, str)
                    else "Conversion failed.",
                "code": "conversion_failed",
            }})

    if status != "complete":
        app.logger.error(
            "File-convert backend %s returned unknown status %r for "
            "id=%s: request_id=%s", b_name, status, job_id,
            f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend returned an unknown status")

    return await _completed(f_req, user, b_name, b_cfg, parsed, job, data,
        ai_headers)


async def _completed(f_req, user, b_name, b_cfg, parsed, job, data,
        ai_headers):
    app = f_req.app

    # Every check that can refuse the result runs before billing and depends
    # only on what the backend stored, so a refused result is refused on
    # every poll: a result is never handed out without being billable, not
    # even after the one-time claim was spent on a refused attempt.
    pages = page_count_from_response(data)
    if pages is None:
        app.logger.error(
            "File-convert backend %s returned no billable page_count for "
            "id=%s: request_id=%s", b_name, job["id"], f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend returned no page_count")

    claimed = data.get("delivery_claimed")
    if not isinstance(claimed, bool):
        # Without the claim the proxy cannot tell the first pickup from the
        # tenth: billing every poll overcharges, billing none gives it away.
        app.logger.error(
            "File-convert backend %s did not report delivery_claimed for "
            "id=%s (backend does not support background billing; update the "
            "file-processor): request_id=%s",
            b_name, job["id"], f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend does not support background "
                "conversion")

    output = data.get(_content_key(data))
    if not isinstance(output, str):
        app.logger.error(
            "File-convert backend %s returned no converted text for id=%s: "
            "request_id=%s", b_name, job["id"], f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend returned no output")

    result = {**job, "status": "completed", "success": True,
        "output": output, "output_format": data.get("output_format"),
        "page_count": pages, "images": data.get("images", {})}
    if "provenance" in data:
        result["provenance"] = data["provenance"]
    f_res = aiohttp.web.json_response(result, headers=ai_headers)

    if claimed:
        await billing.record(f_req, user, {
            "%s/%s/conversion" % (b_name, b_cfg["device"]): pages,
        })
        app.logger.info("Client used: %d page(s) of %s", pages, b_name)
        metrics.FILE_CONVERSIONS_TOTAL.labels(b_name).inc(pages)

    app.logger.info(
        "Background conversion delivered: id=%s request_id=%s "
        "backend_request_id=%s pages=%d billed=%s",
        job["id"], f_req["request_id"], parsed.request_id, pages, claimed)

    return f_res
