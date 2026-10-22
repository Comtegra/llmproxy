"""Realtime transcription: a WebSocket proxy with usage metering.

The client speaks the OpenAI Realtime transcription subset to
``/v1/realtime``; the proxy authenticates the API key (Bearer header, or the
``bearer`` subprotocol a browser handshake carries), selects the backend from
the optional ``?model=`` query param, opens a second WebSocket to the backend
with the backend's own token and pumps frames both ways. The client's key
never reaches the backend: it sees only the configured ``token`` and no
subprotocols.

Downstream frames are sniffed for ``conversation.item.input_audio_transcription.completed``
events so ``usage.seconds`` lands in ``llmproxy_audio_seconds_total``.
Billing events are deliberately NOT written yet: the per-segment vs
per-session decision is pending, and the sniff hook is the single call site
they will plug into.
"""

import asyncio
import json
import time

import aiohttp
import aiohttp.web
import yarl

from . import auth, errors, metrics, proxy, provenance, ratelimit

COMPLETED = "conversation.item.input_audio_transcription.completed"

# WebSocket close codes. 1008 from the backend means the backend rejected the
# proxy's credentials or request -- a configuration fault, never the client's;
# it is masked as 1011 so no internal reason leaks. 1013 (try again later)
# passes through: it is the backend saying it is out of sessions.
BACKEND_AUTH_FAILED = 1008
SESSION_LIMIT = 1013
INTERNAL_ERROR = 1011
PASSTHROUGH_CLOSES = frozenset({1000, 1001, SESSION_LIMIT})

# Bounded drain after one side ends: the backend should close promptly once
# the proxy closes its end, but a stuck backend must not pin the handler.
DRAIN_TIMEOUT = 30

HEARTBEAT_DEFAULT = 30


def _select_backend(req):
    """Pick the realtime backend for this handshake.

    ``?model=`` is optional (OpenAI clients always send it): absent means
    "the" realtime backend and is a 400 only when the config offers more
    than one to choose from. A model of another type never reaches a
    backend (404), exactly as on the HTTP endpoints."""
    backends = req.app["config"].get("backends", {})
    b_name = req.query.get("model")

    if b_name is None:
        candidates = [name for name, cfg in backends.items()
            if cfg.get("type") == "realtime"]
        if len(candidates) == 1:
            b_name = candidates[0]
        elif not candidates:
            raise proxy._model_error(req,
                "No realtime model is configured.", "model_not_found")
        else:
            raise errors.bad_request(req,
                "Missing required parameter: 'model'.", param="model")

    b_cfg = backends.get(b_name)
    if b_cfg is None:
        raise proxy._model_error(req, "The model %r does not exist." % b_name,
            "model_not_found")
    if b_cfg["type"] != "realtime":
        raise proxy._model_error(req,
            "The model %r has type %r and is not supported on %s." %
                (b_name, b_cfg["type"], req.rel_url.path),
            "model_not_supported")
    return b_name, b_cfg


def _client_offered_bearer(req):
    """The browser negotiated 'bearer' (its key rode the subprotocols)."""
    protos = [p.strip() for p in
        req.headers.get("Sec-WebSocket-Protocol", "").split(",") if p.strip()]
    return "bearer" in protos


async def _forward_up(ws_client, ws_backend):
    """Client -> backend frames, verbatim."""
    async for msg in ws_client:
        if msg.type == aiohttp.WSMsgType.TEXT:
            await ws_backend.send_str(msg.data)
        elif msg.type == aiohttp.WSMsgType.BINARY:
            await ws_backend.send_bytes(msg.data)


async def _forward_down(ws_backend, ws_client, on_completed):
    """Backend -> client frames, sniffed for the usage hook on the way."""
    async for msg in ws_backend:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                evt = json.loads(msg.data)
                if isinstance(evt, dict) and evt.get("type") == COMPLETED:
                    on_completed(evt)
            except ValueError:
                pass  # not JSON: still forwarded, the pipe stays transparent
            try:
                await ws_client.send_str(msg.data)
            except (ConnectionError, aiohttp.ClientError):
                pass  # client gone: keep draining for the usage it owes
        elif msg.type == aiohttp.WSMsgType.BINARY:
            try:
                await ws_client.send_bytes(msg.data)
            except (ConnectionError, aiohttp.ClientError):
                pass


def _map_close_code(code):
    """Backend close code -> (client close code, log level)."""
    if code in PASSTHROUGH_CLOSES:
        return code, "info"
    # 1008 is the backend rejecting the proxy's request (bad token, unknown
    # model): the client cannot have caused it and must not see the reason.
    return INTERNAL_ERROR, "critical" if code == BACKEND_AUTH_FAILED else "error"


async def _connect_backend(req, b_name, b_cfg):
    """Open the backend WebSocket; map connect failures to 502/504."""
    app = req.app
    # Path-preserving like the HTTP endpoints: the backend serves the same
    # /v1/realtime path it would speak directly. The model is rewritten to
    # the backend's real id when the config provides one.
    b_url = (yarl.URL(b_cfg["url"]) / "v1/realtime").with_query(
        {"model": b_cfg.get("model") or b_name})
    b_start = time.monotonic()
    try:
        ssl = None if b_cfg.get("verify_ssl", True) else False
        ws_backend = await app["client"].ws_connect(
            b_url,
            headers={"Authorization": "Bearer %s" % b_cfg["token"]},
            heartbeat=b_cfg.get("heartbeat", HEARTBEAT_DEFAULT),
            ssl=ssl)
    except aiohttp.ServerTimeoutError as e:
        metrics.BACKEND_DURATION_SECONDS.labels(b_name).observe(
            time.monotonic() - b_start)
        metrics.BACKEND_ERRORS_TOTAL.labels(b_name, "timeout").inc()
        req.app.logger.error("Backend timeout: request_id=%s model=%s error=%s",
            req["request_id"], b_name, e)
        raise aiohttp.web.HTTPGatewayTimeout() from e
    except aiohttp.ClientError as e:
        metrics.BACKEND_DURATION_SECONDS.labels(b_name).observe(
            time.monotonic() - b_start)
        metrics.BACKEND_ERRORS_TOTAL.labels(b_name, "connection").inc()
        req.app.logger.error("Backend error: request_id=%s model=%s error=%s",
            req["request_id"], b_name, e)
        raise aiohttp.web.HTTPBadGateway() from e
    metrics.BACKEND_DURATION_SECONDS.labels(b_name).observe(
        time.monotonic() - b_start)
    metrics.BACKEND_REQUESTS_TOTAL.labels(b_name, "101").inc()
    return ws_backend


async def realtime(f_req):
    app = f_req.app

    try:
        user = await auth.require_token(f_req, auth.extract_token(f_req))
    except aiohttp.web.HTTPUnauthorized as exc:
        raise errors.json_error(f_req, aiohttp.web.HTTPUnauthorized,
            "Invalid or missing API key.",
            openai_type="authentication_error",
            anthropic_type="authentication_error") from exc

    # Release the request's database connection before the session starts:
    # the auth row is already in hand and a session can outlive many
    # connection generations. close_db would otherwise hold it for the
    # whole session (on SQLite, a dedicated connection and worker thread).
    db = f_req.pop("db", None)
    if db is not None:
        await db.close()

    b_name, b_cfg = _select_backend(f_req)

    # Only after auth and routing does a session exist. The concurrency slot
    # and the per-user gauge span the WHOLE session: a 429 never opens the
    # backend connection.
    async with ratelimit.slot(f_req, user, b_name, b_cfg):
        with metrics.track_user_request(user, b_name):
            ws_backend = await _connect_backend(f_req, b_name, b_cfg)
            try:
                ws_client = aiohttp.web.WebSocketResponse(
                    protocols=("bearer",) if _client_offered_bearer(f_req)
                    else None)
                # Transcript semantics come from the input audio (the Art.
                # 50(2) no-substantial-alteration carve-out): header-only
                # marking, and the 101 is the only header carrier a WS has.
                ws_client.headers.update(provenance.headers(app["config"]))
                await ws_client.prepare(f_req)

                def on_completed(evt):
                    usage = evt.get("usage") or {}
                    seconds = usage.get("seconds")
                    if isinstance(seconds, (int, float)) and seconds > 0:
                        metrics.AUDIO_SECONDS_TOTAL.labels(b_name).inc(
                            float(seconds))
                        # billing deferred: per-segment vs per-session is a
                        # pending decision; this is the single hook site.

                up = asyncio.create_task(_forward_up(ws_client, ws_backend))
                down = asyncio.create_task(
                    _forward_down(ws_backend, ws_client, on_completed))
                done, _ = await asyncio.wait({up, down},
                    return_when=asyncio.FIRST_COMPLETED)

                b_close = None
                if down in done:
                    b_close = ws_backend.close_code
                    code, level = _map_close_code(b_close)
                    if level == "critical":
                        app.logger.critical(
                            "Realtime backend %s rejected the session "
                            "(close code %d): check its token and model "
                            "config; request_id=%s", b_name, b_close,
                            f_req["request_id"])
                    elif level == "error":
                        app.logger.error(
                            "Realtime backend %s closed the session with "
                            "code %d; request_id=%s", b_name, b_close,
                            f_req["request_id"])
                    await ws_client.close(code=code)
                else:
                    # The client went first. Close the backend so it ends its
                    # session, then keep draining whatever it still owes (a
                    # completed event can be in flight).
                    await ws_backend.close(code=ws_client.close_code or 1000)
                    try:
                        await asyncio.wait_for(down, DRAIN_TIMEOUT)
                    except asyncio.TimeoutError:
                        down.cancel()
                up.cancel()
                await asyncio.gather(up, down, return_exceptions=True)
                return ws_client
            finally:
                await ws_backend.close()
