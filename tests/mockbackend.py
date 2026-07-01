import asyncio
import json

import aiohttp
import aiohttp.web_exceptions


async def health(req):
    return aiohttp.web.Response()


async def chat(req):
    b = await req.json()
    if b["model"] != "mymodel":
        raise aiohttp.web_exceptions.HTTPBadRequest(body="bad model")

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
    await req.post()

    return aiohttp.web.json_response({
        "task": "transcribe",
        "language": "pl",
        "duration": 12.5,
        "text": "you said something",
        "segments": [],
    })


def create_app():
    app = aiohttp.web.Application()
    app.add_routes([
        aiohttp.web.get("/health", health),
        aiohttp.web.post("/v1/chat/completions", chat),
        aiohttp.web.post("/v1/embeddings", embeddings),
        aiohttp.web.post("/v1/audio/transcriptions", transcriptions),
    ])
    return app
