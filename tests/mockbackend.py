import asyncio
import json
import secrets

import aiohttp
import aiohttp.web_exceptions


async def health(req):
    return aiohttp.web.Response()


async def chat(req):
    b = await req.json()
    if b["model"] != "mymodel":
        raise aiohttp.web_exceptions.HTTPBadRequest(body="bad model")

    if b.get("_chunked_nonstream"):
        # Non-stream JSON delivered with chunked Transfer-Encoding (as a
        # buffering reverse proxy / HTTP-2 hop might re-frame it). Content-Type
        # stays application/json, so the proxy must NOT misdetect it as a
        # stream (which would silently drop billing).
        res = aiohttp.web.StreamResponse(
            headers={"Content-Type": "application/json"})
        res.enable_chunked_encoding()
        await res.prepare(req)
        await res.write(json.dumps({
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}).encode())
        await res.write_eof()
        return res

    if (err := b.get("_trigger_error")) is not None:
        if err == "context500":
            return aiohttp.web.json_response(
                {"error": {"message":
                    "maximum context length exceeded for this model"}},
                status=500)
        if err == "slow":
            await asyncio.sleep(2)
        if err == 201:
            return aiohttp.web.json_response(
                {"unexpected": "created"},
                status=201)
        if err == 400:
            return aiohttp.web.json_response(
                {"error": {"message": "Input too long",
                    "type": "invalid_request_error"}},
                status=400)
        if err == 422:
            return aiohttp.web.json_response(
                {"error": {"message": "Context length exceeded",
                    "type": "invalid_request_error"}},
                status=422)
        if err == 500:
            return aiohttp.web.json_response(
                {"error": {"message": "Internal server error"}},
                status=500)

    msg = b["messages"][0]["content"]

    if b.get("stream"):
        return await _chat_stream(req, msg, b.get("_stream_mode"))

    return aiohttp.web.json_response({
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        "choices": [{"message": {"content": "you said: %s" % msg}}]
    })


async def _chat_stream(req, msg, mode):
    res = aiohttp.web.StreamResponse(
        headers={"Content-Type": "text/event-stream"})
    res.enable_chunked_encoding()
    await res.prepare(req)

    async def send(obj):
        await res.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

    if mode == "split_usage":
        # Realistic vLLM: content chunk carries finish_reason but NO usage;
        # usage arrives in a SEPARATE trailing chunk with empty choices.
        await send({"choices": [{"delta": {"content": "hi"},
            "finish_reason": "stop"}]})
        await send({"choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    else:
        # Default: a single chunk carrying both content and usage.
        await send({"choices": [{"delta": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}})

    await res.write(b"data: [DONE]\n\n")
    await res.write_eof()
    return res


async def embeddings(req):
    b = await req.json()
    if b["model"] != "mymodel":
        raise aiohttp.web_exceptions.HTTPBadRequest(body="bad model")

    return aiohttp.web.json_response({
        "object": "list",
        "data": [{"object": "embedding", "index": 0,
            "embedding": [0.1, 0.2, 0.3]}],
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    })


async def transcriptions(req):
    # Multipart in. The proxy forces response_format=verbose_json and bills the
    # top-level `duration`. Fractional on purpose to exercise Decimal handling.
    post = await req.post()

    if post.get("_omit_duration"):
        # Simulate a backend that fails to report a billable duration.
        return aiohttp.web.json_response({
            "task": "transcribe", "language": "pl", "text": "no duration"})

    # Inject unbillable durations to exercise the proxy guard. json_response
    # emits bare NaN/Infinity (json.dumps allow_nan default), just like a real
    # Python backend serialized with the default settings.
    unbillable = {"zero": 0, "neg": -5, "nan": float("nan"),
        "inf": float("inf"), "true": True}
    duration = unbillable.get(post.get("_duration"), 12.5)

    return aiohttp.web.json_response({
        "task": "transcribe",
        "language": "pl",
        "duration": duration,
        "text": "you said something",
        "segments": [],
    })


async def _sse(req, events):
    res = aiohttp.web.StreamResponse(
        headers={"Content-Type": "text/event-stream"})
    res.enable_chunked_encoding()
    await res.prepare(req)
    for event, obj in events:
        await res.write(b"event: " + event.encode() + b"\ndata: "
            + json.dumps(obj).encode() + b"\n\n")
    await res.write_eof()
    return res


async def messages(req):
    b = await req.json()
    if b["model"] != "mymodel":
        raise aiohttp.web_exceptions.HTTPBadRequest(body="bad model")

    if b.get("stream"):
        # Anthropic splits usage: input (+cache) in message_start, cumulative
        # output in the final message_delta.
        return await _sse(req, [
            ("message_start", {"type": "message_start",
                "message": {"usage": {"input_tokens": 3, "output_tokens": 1}}}),
            ("content_block_delta", {"type": "content_block_delta",
                "delta": {"text": "hi"}}),
            ("message_delta", {"type": "message_delta",
                "usage": {"output_tokens": 5}}),
            ("message_stop", {"type": "message_stop"}),
        ])

    if b.get("_no_usage"):
        # 200 without a usage object -> the proxy must fail loud, not bill 0.
        return aiohttp.web.json_response({
            "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "hi"}]})

    if b.get("_no_input"):
        # usage present but missing input_tokens -> fail loud, not prompt=0.
        return aiohttp.web.json_response({
            "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"output_tokens": 5}})

    return aiohttp.web.json_response({
        "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 3, "output_tokens": 5},
    })


async def responses(req):
    b = await req.json()
    if b["model"] != "mymodel":
        raise aiohttp.web_exceptions.HTTPBadRequest(body="bad model")

    if b.get("stream"):
        # Usage only in the terminal event; no [DONE]. `_terminal` lets tests
        # end the stream with response.completed / .incomplete / .failed.
        terminal = b.get("_terminal", "response.completed")
        return await _sse(req, [
            ("response.created", {"type": "response.created",
                "response": {"id": "resp_1"}}),
            ("response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "hi"}),
            (terminal, {"type": terminal,
                "response": {"usage":
                    {"input_tokens": 7, "output_tokens": 9}}}),
        ])

    return aiohttp.web.json_response({
        "object": "response",
        "output": [{"type": "message",
            "content": [{"type": "output_text", "text": "hi"}]}],
        "usage": {"input_tokens": 7, "output_tokens": 9},
    })


async def marker(req):
    # Datalab Marker sync contract: multipart file in, JSON out. Failures are
    # HTTP 200 with success=false (Datalab convention); the proxy must not bill.
    post = await req.post()
    req.app["marker_forms"].append(
        {k: v for k, v in post.items() if isinstance(v, str)})
    # Proxy must strip `model` before forwarding -- Marker never sees it.
    if "model" in post:
        return aiohttp.web.json_response(
            {"success": False, "error": "unexpected model field"},
            status=200)

    if post.get("_fail"):
        return aiohttp.web.json_response(
            {"success": False, "error": "conversion failed"},
            status=200)

    if post.get("_nested"):
        # Deeper than the JSON parser's recursion limit.
        return aiohttp.web.Response(body=b"[" * 100000 + b"]" * 100000,
            content_type="application/json")

    file_field = post.get("file")
    if file_field is None:
        return aiohttp.web.json_response(
            {"success": False, "error": "no file provided"},
            status=200)

    # `_no_request_id` plays a backend that predates background conversions:
    # it ignores background=true and converts synchronously.
    if post.get("background") == "true" and not post.get("_no_request_id"):
        return _marker_queue(req, post)

    payload = {
        "success": True,
        "output": "# Converted\n\nhello",
        "images": {},
        "output_format": post.get("output_format", "markdown"),
    }
    if not post.get("_omit_page_count"):
        try:
            payload["page_count"] = int(post.get("_pages", "1"))
        except (TypeError, ValueError):
            payload["page_count"] = 1
    return aiohttp.web.json_response(payload)


# Keys the converted text is never stored under (file-processor contract).
MARKER_RESERVED_KEYS = {"status", "success", "images", "page_count",
    "output_format", "provenance", "delivery_claimed", "error_code", "error",
    "request_id"}


def _marker_queue(req, post):
    """Datalab async contract: queue the job, answer with a request id.

    Form controls: `_polls` = polls answered "processing" before the job is
    done; `_poll_fail` = the conversion fails; `_no_claim` = a backend that
    does not report delivery_claimed; `_pages` (JSON, so 0/true/1.5/"3"
    play a broken backend) / `_omit_page_count` as for the sync path;
    `_claim_raw` = JSON sent as delivery_claimed instead of the one-time bool;
    `_request_id_raw` = JSON sent as request_id (nothing is queued); `_text` =
    the converted text, `_text_size` = that many "x" instead.
    """
    if (raw := post.get("_request_id_raw")) is not None:
        return aiohttp.web.json_response({"success": True,
            "request_id": json.loads(raw)})
    rid = secrets.token_hex(16)
    req.app["marker_jobs"][rid] = {
        "remaining": int(post.get("_polls", "0")),
        "fail": bool(post.get("_poll_fail")),
        "no_claim": bool(post.get("_no_claim")),
        "claim_raw": post.get("_claim_raw"),
        "pages": None if post.get("_omit_page_count")
            else json.loads(post.get("_pages", "1")),
        "output_format": post.get("output_format", "markdown"),
        "text": "x" * int(post["_text_size"]) if "_text_size" in post
            else post.get("_text", "# Converted\n\nhello"),
        "claimed": False,
    }
    return aiohttp.web.json_response({
        "success": True,
        "request_id": rid,
        "request_check_url": "http://marker.invalid/api/v1/marker/%s" % rid,
    })


async def marker_poll(req):
    req.app["marker_polls"].append(dict(req.headers))
    control = req.app["marker_control"]
    if delay := control.get("poll_delay"):
        await asyncio.sleep(delay)
    if (status := control.get("poll_status")) is not None:
        return aiohttp.web.json_response({"error": "boom"}, status=status)
    if (raw := control.get("poll_raw")) is not None:
        return aiohttp.web.Response(body=raw, content_type="application/json")

    job = req.app["marker_jobs"].get(req.match_info["request_id"])
    if job is None:
        return aiohttp.web.json_response({"status": "failed",
            "success": False, "error": "Job not found",
            "error_code": "JOB_NOT_FOUND"})
    if job["remaining"] > 0:
        job["remaining"] -= 1
        return aiohttp.web.json_response(
            {"status": "processing", "success": True})
    if job["fail"]:
        return aiohttp.web.json_response({"status": "failed",
            "success": False, "error": "PDF is encrypted",
            "error_code": "CONVERSION_FAILED"})

    key = job["output_format"].lower()
    if key in MARKER_RESERVED_KEYS:
        key = "markdown"
    payload = {"status": "complete", "success": True,
        key: job["text"], "images": {"a.png": "aGk="},
        "output_format": job["output_format"],
        "provenance": {"ai_generated": True}}
    if job["pages"] is not None:
        payload["page_count"] = job["pages"]
    if req.headers.get("X-Delivery-Claim") == "true" and not job["no_claim"]:
        payload["delivery_claimed"] = not job["claimed"]
        job["claimed"] = True
        if job["claim_raw"] is not None:
            payload["delivery_claimed"] = json.loads(job["claim_raw"])
    return aiohttp.web.json_response(payload,
        headers={"X-AI-Processed": "true"})


@aiohttp.web.middleware
async def record_path(req, handler):
    # Lets tests prove a request the proxy rejected never reached a backend.
    req.app["paths"].append(req.path)
    return await handler(req)


def create_app():
    # Raised so the proxy can forward multi-MiB audio uploads to us in tests.
    app = aiohttp.web.Application(client_max_size=2 * 1024 ** 3,
        middlewares=[record_path])
    app["paths"] = []
    app["decision_calls"] = []
    app["decision_control"] = {}
    app["marker_forms"] = []
    app["marker_jobs"] = {}
    app["marker_polls"] = []
    app["marker_control"] = {}
    app.add_routes([
        aiohttp.web.post("/v1/systemone", decision),
        aiohttp.web.get("/health", health),
        aiohttp.web.post("/v1/chat/completions", chat),
        # The legacy route reuses the chat handler (chat-shaped body expected);
        # the proxy forwards path-preserving, which is all provenance tests need.
        aiohttp.web.post("/v1/completions", chat),
        aiohttp.web.post("/v1/embeddings", embeddings),
        aiohttp.web.post("/v1/audio/transcriptions", transcriptions),
        aiohttp.web.post("/v1/messages", messages),
        aiohttp.web.post("/v1/responses", responses),
        # Path-rewritten target for /v1/files/convert (not path-preserving).
        aiohttp.web.post("/api/v1/marker", marker),
        aiohttp.web.get("/api/v1/marker/{request_id}", marker_poll),
    ])
    return app


async def decision(req):
    body = await req.json()
    req.app["decision_calls"].append((body, dict(req.headers)))
    control = req.app["decision_control"]
    if event := control.get("started"):
        event.set()
    if event := control.get("release"):
        await event.wait()
    await asyncio.sleep(control.get("delay", 0))
    data = control.get("response", {
        "model": "Remek/basal-2.0-max-GGUF",
        "answers": {name: {"type": "choice", "choice": "yes",
            "probabilities": {"yes": 0.9, "no": 0.1}, "confidence": 0.9}
            for name in body["questions"]},
        "usage": {"input_tokens": 880, "output_tokens": 0}})
    raw = control.get("raw", json.dumps(data).encode())
    content_type = control.get("content_type", "application/json")
    status = control.get("status", 200)
    if control.get("chunked"):
        res = aiohttp.web.StreamResponse(status=status,
            headers={"Content-Type": content_type})
        await res.prepare(req)
        for start in range(0, len(raw), 17):
            await res.write(raw[start:start + 17])
        await res.write_eof()
        return res
    return aiohttp.web.Response(body=raw, status=status, content_type=content_type)
