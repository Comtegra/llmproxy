import datetime
import json
import logging

import aiohttp
import aiohttp.web
import pymongo.errors
import yarl

from . import auth


async def handle_resp_stream(f_req, b_res):
    app = f_req.app
    f_res = aiohttp.web.StreamResponse()

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

    try:
        f_body = await f_req.json()
    except json.decoder.JSONDecodeError as e:
        raise aiohttp.web.HTTPBadRequest(body="JSON decode error: %s" % e)

    try:
        b_cfg = app["config"]["backends"][f_body["model"]]
    except KeyError:
        raise aiohttp.web.HTTPUnauthorized(body="Incorrect model")

    b_url = yarl.URL(b_cfg["url"]) / str(f_req.rel_url)[1:]
    b_hdrs = {"Authorization": "Bearer %s" % b_cfg["token"]}

    try:
        async with app["client"].post(b_url, headers=b_hdrs, json=f_body) as b_res:
            if b_res.status != 200:
                app.logger.error("Backend \"%s\" error: %d %s", f_body["model"],
                    b_res.status, (await b_res.text()))
                raise aiohttp.web.HTTPBadGateway()

            if b_res.headers.get("Transfer-Encoding", "") == "chunked":
                f_res, usage = await handle_resp_stream(f_req, b_res)
            else:
                body = await b_res.content.read()
                data = json.loads(body)
                f_res = aiohttp.web.Response(body=body)
                usage = data["usage"]

            try:
                await app["db"].put_event(
                    user=user,
                    time=datetime.datetime.now(datetime.UTC),
                    model=f_body["model"],
                    device=b_cfg["device"],
                    prompt_n=usage["prompt_tokens"],
                    completion_n=usage["completion_tokens"],
                )
            except pymongo.errors.PyMongoError as e:
                app.logger.critical(e)
                raise aiohttp.web.GracefulExit() from e

            app.logger.info("Client used: P:%d C:%d tokens of %s",
                usage["prompt_tokens"], usage["completion_tokens"],
                f_body["model"])

            return f_res
    except aiohttp.ServerTimeoutError as e:
        app.logger.error("Backend timeout: %s", e)
        raise aiohttp.web.HTTPGatewayTimeout() from e
    except (aiohttp.ClientConnectorError, aiohttp.ServerConnectionError,
            aiohttp.ClientPayloadError, aiohttp.ClientResponseError,
            aiohttp.InvalidURL) as e:
        app.logger.error("Backend error: %s", e)
        raise aiohttp.web.HTTPBadGateway() from e
    except aiohttp.ClientError as e:
        app.logger.error("HTTP client error: %s", e)
        raise aiohttp.web.HTTPInternalServerError() from e


async def models(req):
    await auth.require_auth(req)

    data = {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": None, "owned_by": None}
            for model in req.app["config"]["backends"].keys()
        ],
    }

    return aiohttp.web.Response(text=json.dumps(data),
        content_type="application/json")
