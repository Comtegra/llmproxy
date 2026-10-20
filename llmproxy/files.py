import asyncio
import contextlib
import json

import aiohttp
import yarl

from . import (
    auth, billing, conversion_id, errors, metrics, provenance, proxy, streaming,
)


# Marker Datalab-compat path (PDF -> Markdown). Not path-preserving:
# the public route is /v1/files/convert.
MARKER_PATH = "api/v1/marker"

OBJECT = "file.conversion"

# Seconds a client should wait before polling a background conversion again.
RETRY_AFTER = "5"

# POST {job}/claim: the backend answers delivery_claimed=true exactly once per
# completed job, which is when the proxy bills. The proxy keeps no state, so
# this one-time claim is what makes "bill at the first pickup" hold across
# repeated polls, restarts and replicas. Two-phase: the GET that fetches the
# result changes nothing, and the claim is made only with a complete, valid
# result in hand, so a fetch that fails or is refused never spends it.
CLAIM_PATH = "claim"

# The claim answer is a one-field JSON object.
_MAX_CLAIM_RESPONSE_BYTES = 64 * 1024

# A result is handed to the client in pieces of this size, each of which must
# be taken within timeout_read: a client that stops reading is cut off instead
# of holding its slot and the result in memory for good, while a slow one that
# keeps reading gets the whole result.
_WRITE_CHUNK = 1024 * 1024

# Polls one API key may have in flight at once ([files] max_polls_in_flight).
# Polls skip the model's rate limits, so without a limit of their own one
# client could tie up the backend connection pool and the memory of the single
# worker with results, for every other client.
DEFAULT_MAX_POLLS_IN_FLIGHT = 4

# Largest poll answer read from the backend ([files] max_poll_response_bytes):
# a completed result carries the Markdown and base64 images. A poll holds a
# few copies of it while decoding and re-encoding, times the polls in flight,
# and decoding it blocks the event loop of the single worker (a thread would
# not help, json holds the GIL): this limit is what bounds both.
DEFAULT_MAX_POLL_RESPONSE_BYTES = 128 * 1024 ** 2

# API key id -> polls in flight. Process-local like the rate-limit counters.
_polls_in_flight = {}

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


def billing_product(f_req, b_name, b_cfg):
    """The billed resource, ``<backend>/<device>/conversion``, or raise 500.

    The config check guarantees a device; this guards the one place where a
    gap would cost money. A poll computes it before it asks the backend for
    anything, so a broken config can never fail after the one-time delivery
    claim was spent.
    """
    device = b_cfg.get("device")
    if not isinstance(device, str) or not device.strip():
        f_req.app.logger.error(
            "File-convert backend %s has no device to bill: request_id=%s",
            b_name, f_req["request_id"])
        raise aiohttp.web.HTTPInternalServerError()
    return "%s/%s/conversion" % (b_name, device)


def page_count_from_response(data):
    """Return a billable page count, or None if missing/unusable."""
    pages = data.get("page_count")
    if isinstance(pages, bool) or not isinstance(pages, int) or pages <= 0:
        return None
    return pages


_INVALID_JSON = object()


def _decode_json(body):
    try:
        return json.loads(body)
    except (ValueError, RecursionError):
        # ValueError covers bad JSON and bad UTF-8; RecursionError a body
        # nested deeper than the parser's recursion limit.
        return _INVALID_JSON


def _backend_json(f_req, b_name, body):
    """Decode a backend answer, or raise 502."""
    data = _decode_json(body)
    if data is _INVALID_JSON:
        f_req.app.logger.error(
            "File-convert backend %s returned invalid JSON: request_id=%s",
            b_name, f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend returned invalid JSON")
    return data


@contextlib.contextmanager
def _poll_slot(f_req, user, b_name):
    """Hold one of the API key's poll slots, or raise 429.

    Every poll that reaches the backend goes through here.
    """
    limit = f_req.app["config"].get("files", {}).get("max_polls_in_flight",
        DEFAULT_MAX_POLLS_IN_FLIGHT)
    key = user["id"]
    n = _polls_in_flight.get(key, 0)
    if n >= limit:
        metrics.RATE_LIMIT_REJECTIONS_TOTAL.labels(b_name,
            "poll_concurrency").inc()
        raise errors.json_error(f_req, aiohttp.web.HTTPTooManyRequests,
            "Too many polls in flight for this API key: %d." % limit,
            openai_type="rate_limit_exceeded",
            anthropic_type="rate_limit_error", code="rate_limit_exceeded",
            headers={"Retry-After": RETRY_AFTER})
    _polls_in_flight[key] = n + 1
    try:
        yield
    finally:
        n = _polls_in_flight[key] - 1
        if n:
            _polls_in_flight[key] = n
        else:
            del _polls_in_flight[key]


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
        data = _backend_json(f_req, b_name, body)

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
            billing_product(f_req, b_name, b_cfg): pages,
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
        # A file-processor that predates background conversions either
        # ignores background=true and converts synchronously (no request id)
        # or, with MARKER_DATALAB_COMPAT_SYNC=False, queues the job under its
        # 16-hex content hash, for which it reports no delivery claims.
        app.logger.error(
            "File-convert backend %s returned no usable request_id for a "
            "background conversion (got %s; the backend does not support "
            "background conversions, update the file-processor): "
            "request_id=%s", b_name,
            "a 16-hex content hash" if isinstance(b_request_id, str)
                and len(b_request_id) == 16 else "none or malformed",
            f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend does not support background "
                "conversion")

    job_id = conversion_id.make(b_cfg["token"], user["id"], b_name,
        b_request_id)
    log_id, log_rid = _log_ids(job_id, b_request_id)
    app.logger.info(
        "Background conversion queued: id=%s request_id=%s "
        "backend_request_id=%s model=%s",
        log_id, f_req["request_id"], log_rid, b_name)

    return aiohttp.web.json_response({
        "id": job_id,
        "object": OBJECT,
        "model": b_name,
        "status": "queued",
    }, status=202, headers={
        "Location": "/v1/files/convert/%s" % job_id,
        "Retry-After": RETRY_AFTER,
    })


def _log_ids(job_id, b_request_id):
    """Short forms of a conversion's ids for routine log lines.

    The backend's request id is enough to fetch the result from the backend
    with its token, and the conversion id carries it (base64), so neither is
    logged in full except where a person needs it to bill by hand ("Delivery
    claim outcome unknown"). The end of the id's tag and the start of the
    backend request id still tie the lines of one conversion together.
    """
    return "fc_...%s" % job_id[-8:], b_request_id[:8]


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
    polling every few seconds must not use up its budget for submitting. They
    have a limit of their own instead, on polls in flight per API key.
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

    with _poll_slot(f_req, user, b_name):
        return await _poll(f_req, user, job_id, parsed, b_name, b_cfg)


async def _poll(f_req, user, job_id, parsed, b_name, b_cfg):
    app = f_req.app

    product = billing_product(f_req, b_name, b_cfg)
    b_url = yarl.URL(b_cfg["url"]) / MARKER_PATH / parsed.request_id
    b_hdrs = {"Authorization": "Bearer %s" % b_cfg["token"]}
    ssl = None if b_cfg.get("verify_ssl", True) else False
    # The global read timeout, not the backend's (which is for waiting out a
    # synchronous conversion): a poll answers at once, and a backend that
    # hangs must not hold the key's slot and a pooled connection for
    # half an hour. A timed-out GET spends nothing, the claim comes after.
    timeout = aiohttp.ClientTimeout(
        connect=app["config"]["timeout_connect"],
        sock_read=app["config"]["timeout_read"])

    with proxy.backend_errors(f_req, b_name) as b_start:
        async with app["client"].get(b_url, headers=b_hdrs, ssl=ssl,
                timeout=timeout) as b_res:
            proxy.observe_backend_response(b_name, b_res, b_start)
            limit = app["config"].get("files", {}).get(
                "max_poll_response_bytes", DEFAULT_MAX_POLL_RESPONSE_BYTES)
            try:
                await proxy.check_response(app, b_name, b_res,
                    request_id=f_req["request_id"], max_bytes=limit)
                body = await proxy.read_limited(b_res, limit)
            except ValueError:
                app.logger.error(
                    "File-convert backend %s poll answer exceeds %d bytes "
                    "(raise [files] max_poll_response_bytes): request_id=%s",
                    b_name, limit, f_req["request_id"])
                raise aiohttp.web.HTTPBadGateway(
                    text="File conversion backend answer too large")
            ai_headers = _ai_headers(app, b_res)

    data = _backend_json(f_req, b_name, body)
    del body
    if not isinstance(data, dict):
        app.logger.error(
            "File-convert backend %s returned a poll answer that is not a "
            "JSON object: request_id=%s", b_name, f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend returned invalid JSON")

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
            "id=%s: request_id=%s", b_name, status,
            _log_ids(job_id, parsed.request_id)[0],
            f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend returned an unknown status")

    result = await _completed(f_req, user, b_name, b_cfg, product, parsed,
        job, data)
    # Only the encoded result is needed from here on; the write can take a
    # while, so the decoded answer must not stay in memory for it.
    del data
    return await _send_in_slot(f_req, result, ai_headers)


async def _completed(f_req, user, b_name, b_cfg, product, parsed, job, data):
    """Check, claim and bill a completed result; return it encoded."""
    app = f_req.app

    # Every check that can refuse the result runs before the delivery claim,
    # so a refused result leaves the claim for the first pickup that works.
    pages = page_count_from_response(data)
    if pages is None:
        app.logger.error(
            "File-convert backend %s returned no billable page_count for "
            "id=%s: request_id=%s", b_name,
            _log_ids(job["id"], parsed.request_id)[0], f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend returned no page_count")

    output = data.get(_content_key(data))
    if not isinstance(output, str):
        app.logger.error(
            "File-convert backend %s returned no converted text for id=%s: "
            "request_id=%s", b_name,
            _log_ids(job["id"], parsed.request_id)[0], f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend returned no output")

    result = {**job, "status": "completed", "success": True,
        "output": output, "output_format": data.get("output_format"),
        "page_count": pages, "images": data.get("images", {})}
    if "provenance" in data:
        result["provenance"] = data["provenance"]
    body = json.dumps(result).encode()

    claimed = await _claim(f_req, user, b_name, b_cfg, parsed, job["id"],
        pages)
    if claimed:
        await billing.record(f_req, user, {product: pages})
        app.logger.info("Client used: %d page(s) of %s", pages, b_name)
        metrics.FILE_CONVERSIONS_TOTAL.labels(b_name).inc(pages)

    log_id, log_rid = _log_ids(job["id"], parsed.request_id)
    app.logger.info(
        "Background conversion delivered: id=%s request_id=%s "
        "backend_request_id=%s pages=%d billed=%s",
        log_id, f_req["request_id"], log_rid, pages, claimed)

    return body


async def _claim(f_req, user, b_name, b_cfg, parsed, job_id, pages):
    """Claim the delivery of a completed job; True for its first pickup only.

    Anything but a 200 with a boolean (409: not completed after all, 404: the
    job is gone or the backend has no claims) is a 502 without the result:
    without the claim the proxy cannot tell the first pickup from the tenth,
    and billing every poll overcharges while billing none gives it away.

    A claim that fails without an answer (timeout, lost connection) may have
    been granted all the same: then the first pickup is spent and these pages
    are never billed. Logged with everything needed to bill them by hand.
    """
    app = f_req.app

    b_url = (yarl.URL(b_cfg["url"]) / MARKER_PATH / parsed.request_id
        / CLAIM_PATH)
    b_hdrs = {"Authorization": "Bearer %s" % b_cfg["token"]}
    ssl = None if b_cfg.get("verify_ssl", True) else False
    timeout = aiohttp.ClientTimeout(
        connect=app["config"]["timeout_connect"],
        sock_read=app["config"]["timeout_read"])

    try:
        with proxy.backend_errors(f_req, b_name) as b_start:
            async with app["client"].post(b_url, headers=b_hdrs, ssl=ssl,
                    timeout=timeout) as b_res:
                proxy.observe_backend_response(b_name, b_res, b_start)
                try:
                    body = await proxy.read_limited(b_res,
                        _MAX_CLAIM_RESPONSE_BYTES)
                except ValueError:
                    body = b""
                status = b_res.status
    except (aiohttp.web.HTTPException, asyncio.CancelledError):
        app.logger.error(
            "Delivery claim outcome unknown (if the backend granted it, these "
            "pages were not billed): id=%s model=%s backend_request_id=%s "
            "pages=%d api_key_id=%s request_id=%s",
            job_id, b_name, parsed.request_id, pages, user["id"],
            f_req["request_id"])
        raise

    data = _decode_json(body) if status == 200 else None
    claimed = data.get("delivery_claimed") if isinstance(data, dict) else None
    if not isinstance(claimed, bool):
        app.logger.error(
            "File-convert backend %s did not confirm the delivery claim for "
            "id=%s (status %d, body_preview=%s; a backend without claims "
            "needs a file-processor update): request_id=%s",
            b_name, _log_ids(job_id, parsed.request_id)[0], status,
            body[:256].decode("utf-8", errors="replace"), f_req["request_id"])
        raise aiohttp.web.HTTPBadGateway(
            text="File conversion backend did not confirm the delivery")
    return claimed


async def _send_in_slot(f_req, body, headers):
    """Write the result out before the poll's slot is released.

    A Response returned by the handler is written after the handler -- and
    the slot -- is gone: a client that never reads would pile results up in
    memory while its slots stay free for more polls. Sending here keeps every
    result being written inside one of the key's slots, and _WRITE_CHUNK
    bounds how long a client that stopped reading keeps it.
    """
    app = f_req.app
    timeout = app["config"]["timeout_read"]
    f_res = aiohttp.web.StreamResponse(headers=headers)
    f_res.content_type = "application/json"
    f_res.charset = "utf-8"
    f_res.content_length = len(body)
    streaming.set_prepared_headers(f_req, f_res)
    view = memoryview(body)
    try:
        await asyncio.wait_for(f_res.prepare(f_req), timeout)
        for start in range(0, len(view), _WRITE_CHUNK):
            await asyncio.wait_for(
                f_res.write(view[start:start + _WRITE_CHUNK]), timeout)
        await asyncio.wait_for(f_res.write_eof(), timeout)
    # Before OSError: asyncio.TimeoutError is an OSError since Python 3.11.
    except asyncio.TimeoutError:
        app.logger.warning(
            "Client stopped reading a background conversion result for %ss; "
            "connection dropped: request_id=%s", timeout, f_req["request_id"])
        # abort, not close: close would keep flushing the buffered result.
        if (transport := f_req.transport) is not None:
            transport.abort()
    except OSError as e:
        app.logger.info("Client disconnected: %s", e)
    return f_res
