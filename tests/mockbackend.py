import aiohttp
import aiohttp.web_exceptions


async def health(req):
    return aiohttp.web.Response()


async def chat(req):
    b = await req.json()
    if b["model"] != "mymodel":
        raise aiohttp.web_exceptions.HTTPBadRequest(body="bad model")

    if (err := b.get("_trigger_error")) is not None:
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
