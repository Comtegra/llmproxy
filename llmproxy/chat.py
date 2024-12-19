import datetime
import json
import logging

import aiohttp
import aiohttp.web

from . import auth, proxy
from .db import DatabaseError, get_db


async def handle_resp_stream(f_req, b_res):
    app = f_req.app
    headers = {"Content-Type":
        b_res.headers.get("Content-Type", "application/octet-stream")}
    f_res = aiohttp.web.StreamResponse(headers=headers)

    last = b""

    try:
        await f_res.prepare(f_req)
        while (c := await b_res.content.readuntil(b"\n\n")):
            logging.debug("Received chunk: %s", c)
            last = c
            await f_res.write(c)
        await f_res.write_eof()
    except OSError as e:
        app.logger.info("Client disconnected: %s", e)

    while (c := await b_res.content.readuntil(b"\n\n")):
        logging.debug("Received chunk after client disconnected: %s", c)
        last = c

    _, _, body_raw = last.partition(b" ")

    try:
        body = json.loads(body_raw)
    except json.decoder.JSONDecodeError:
        app.logger.error("Failed parsing usage information: %s", body_raw)
        raise

    return f_res, body["usage"]


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def chat(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)

    b_req, b_name, b_cfg = await proxy.request(f_req)
    async with b_req as b_res:
        app.logger.debug("Backend request completed")

        if b_res.status != 200:
            app.logger.error("Backend \"%s\" error: %d %s", b_name,
                b_res.status, (await b_res.text()))
            raise aiohttp.web.HTTPBadGateway()

        if b_res.headers.get("Transfer-Encoding", "") == "chunked":
            f_res, usage = await handle_resp_stream(f_req, b_res)
        else:
            body = await b_res.content.read()
            data = json.loads(body)
            f_hdrs = {"Content-Type":
                b_res.headers.get("Content-Type", "application/octet-stream")}
            f_res = aiohttp.web.Response(body=body, headers=f_hdrs)
            usage = data["usage"]

        db = await get_db(f_req)
        try:
            await db.put_event(
                user=user,
                time=datetime.datetime.now(datetime.UTC),
                product="%s/%s/prompt" % (b_name, b_cfg["device"]),
                quantity=usage["prompt_tokens"],
                request_id=f_req["request_id"],
            )
            await db.put_event(
                user=user,
                time=datetime.datetime.now(datetime.UTC),
                product="%s/%s/completion" % (b_name, b_cfg["device"]),
                quantity=usage["completion_tokens"],
                request_id=f_req["request_id"],
            )
        except DatabaseError as e:
            app.logger.critical(e)
            raise aiohttp.web.GracefulExit() from e

        app.logger.info("Client used: P:%d C:%d tokens of %s",
            usage["prompt_tokens"], usage["completion_tokens"],
            b_name)

        return f_res


async def models(req):
    await auth.require_auth(req)

    data = {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": None, "owned_by": None,
                "device": meta.get("device")}
            for model, meta in req.app["config"]["backends"].items()
        ],
    }

    return aiohttp.web.Response(text=json.dumps(data),
        content_type="application/json")
