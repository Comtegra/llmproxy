import datetime
import io
import json

import aiohttp
import aiohttp.web

from . import auth, proxy
from .db import DatabaseError, get_db


async def readuntil(stream, separator: bytes = b"\n") -> bytes:
    result = bytearray()
    while not result.endswith(separator):
        b = await stream.read(1)
        if not b:
            break
        result += b

    return result


async def handle_resp(f_req, b_res):
    app = f_req.app
    buf = io.StringIO()
    tokens = 0
    data = {}

    while True:
        if f_req.transport is None:
            b_res.close()
            app.logger.info("Client disconnected")
            break

        chunk = await readuntil(b_res.content, b"\n\n")
        if chunk == b"":
            break
        assert chunk.startswith(b"data: ")
        raw = chunk.removeprefix(b"data: ")
        if raw == b"[DONE]\n\n":
            continue
        data = json.loads(raw)
        assert len(data["choices"]) == 1
        content = data["choices"][0]["delta"].get("content")
        if content is not None:
            buf.write(content)
            app.logger.debug("Token: %r", content)
            tokens += 1

    hdrs = {
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
        "Access-Control-Allow-Origin": app["config"].get("http_origin", ""),
    }

    if data:
        data["object"] = "chat.completion"
        assert len(data["choices"]) == 1
        del data["choices"][0]["delta"]
        data["choices"][0]["message"] = {"role": "assistant",
            "content": buf.getvalue()}

    return aiohttp.web.json_response(data, headers=hdrs), tokens


async def handle_resp_stream(f_req, b_res):
    app = f_req.app

    hdrs = {
        "Content-Type": "text/event-stream",
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
        "Access-Control-Allow-Origin": app["config"].get("http_origin", ""),
    }

    f_res = aiohttp.web.StreamResponse(headers=hdrs)
    tokens = 0

    try:
        await f_res.prepare(f_req)

        while True:
            chunk = await readuntil(b_res.content, b"\n\n")
            if chunk == b"":
                break
            await f_res.write(chunk)
            app.logger.debug("Chunk: %r", chunk)
            tokens += 1

        await f_res.write_eof()

        # Last 2 chunks contain no tokens (usage -> [DONE])
        tokens -= 2
    except OSError as e:
        b_res.close()
        app.logger.info("Client disconnected: %s", e)

    return f_res, tokens


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def chat(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)

    if f_req.content_type != "application/json":
        raise aiohttp.web.HTTPUnsupportedMediaType()

    try:
        f_body = await f_req.json()
    except json.decoder.JSONDecodeError as e:
        raise aiohttp.web.HTTPBadRequest(text="JSON decode error: %s" % e)

    stream = f_body.get("stream", False)

    f_body["stream"] = True
    if "stream_options" not in f_body:
        f_body["stream_options"] = {}
    f_body["stream_options"]["include_usage"] = True

    app.logger.debug("Frontend request body:\n%s", f_body)

    b_name = f_body.get("model")
    b_cfg = proxy.get_backend_cfg(app, b_name)

    async with proxy.request(app, b_cfg, "apply-template", f_body) as prompt_res:
        prompt = (await prompt_res.json())["prompt"]
        app.logger.debug("Prompt: %s", prompt)

    tok_body = {"content": prompt, "add_special": True}
    async with proxy.request(app, b_cfg, "tokenize", tok_body) as tokens_res:
        tok_prompt = len((await tokens_res.json())["tokens"])

    async with proxy.request(app, b_cfg, "v1/chat/completions", f_body) as b_res:
        app.logger.debug("Backend request completed")

        if b_res.status != 200:
            app.logger.error("Backend \"%s\" error: %d %s", b_name,
                b_res.status, (await b_res.text()))
            raise aiohttp.web.HTTPBadGateway()

        if stream:
            f_res, tok_compl = await handle_resp_stream(f_req, b_res)
        else:
            f_res, tok_compl = await handle_resp(f_req, b_res)

        db = await get_db(app["config"]["db"]["uri"], f_req)
        try:
            await db.event_create(
                user=user,
                time=datetime.datetime.now(datetime.UTC),
                product="%s/%s/prompt" % (b_name, b_cfg["device"]),
                quantity=tok_prompt,
                request_id=f_req["request_id"],
            )
            await db.event_create(
                user=user,
                time=datetime.datetime.now(datetime.UTC),
                product="%s/%s/completion" % (b_name, b_cfg["device"]),
                quantity=tok_compl,
                request_id=f_req["request_id"],
            )
        except DatabaseError as e:
            app.logger.critical(e)
            raise aiohttp.web.GracefulExit() from e

        app.logger.info("Client used: P:%d C:%d tokens of %s",
            tok_prompt, tok_compl, b_name)

        return f_res


async def models(req):
    await auth.require_auth(req)

    data = {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": None, "owned_by": None,
                "device": meta.get("device")}
            for model, meta in req.app["config"].get("backends", {}).items()
        ],
    }

    return aiohttp.web.Response(text=json.dumps(data),
        content_type="application/json")
