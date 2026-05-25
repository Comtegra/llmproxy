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
        res = aiohttp.web.StreamResponse(
            headers={"Content-Type": "text/event-stream"})
        res.enable_chunked_encoding()
        await res.prepare(req)
        chunk = {
            "choices": [{"delta": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        await res.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        await res.write(b"data: [DONE]\n\n")
        await res.write_eof()
        return res

    return aiohttp.web.json_response({
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        "choices": [{"message": {"content": "you said: %s" % msg}}]
    })


def create_app():
    app = aiohttp.web.Application()
    app.add_routes([
        aiohttp.web.get("/health", health),
        aiohttp.web.post("/v1/chat/completions", chat),
    ])
    return app
