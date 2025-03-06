import aiohttp
import aiohttp.web_exceptions


async def health(req):
    return aiohttp.web.Response()


async def chat(req):
    b = await req.json()
    if b["model"] != "mymodel":
        raise aiohttp.web_exceptions.HTTPBadRequest(body="bad model")

    msg = b["messages"][0]["content"]

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
