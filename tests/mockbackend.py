import json

import aiohttp
import aiohttp.web_exceptions


async def health(req):
    return aiohttp.web.Response()


async def chat(req):
    b = await req.json()
    if b["model"] != "mymodel":
        raise aiohttp.web_exceptions.HTTPBadRequest(body="bad model")

    tok_prompt = b["messages"][0]["content"].split()
    tok_completion = ["you", "said:", *tok_prompt]

    if b.get("stream", False):
        res = aiohttp.web.StreamResponse()
        await res.prepare(req)
        for idx, chunk in enumerate(tok_completion):
            if idx > 0:
                chunk = " " + chunk
            data = json.dumps({"choices": [{"delta": {"content": chunk}}]})
            await res.write(b"data: %s\n\n" % data.encode())
        await res.write(b"data: [DONE]\n\n")

        await res.write_eof()

        return res

    return aiohttp.web.json_response({
        "usage": {
            "prompt_tokens": len(tok_prompt),
            "completion_tokens": len(tok_prompt),
        },
        "choices": [{"message": {"content": " ".join(tok_completion)}}]
    })


async def apply_template(req):
    b = await req.json()
    msgs = ("%s: %s" % (msg["role"], msg["content"]) for msg in b["messages"])

    return aiohttp.web.json_response({"prompt": "\n\n".join(msgs)})


async def tokenize(req):
    b = await req.json()
    tokens = [idx for idx, _ in enumerate(b["content"].split())]

    return aiohttp.web.json_response({"tokens": tokens})


def create_app():
    app = aiohttp.web.Application()
    app.add_routes([
        aiohttp.web.get("/health", health),
        aiohttp.web.post("/apply-template", apply_template),
        aiohttp.web.post("/tokenize", tokenize),
        aiohttp.web.post("/v1/chat/completions", chat),
    ])
    return app
