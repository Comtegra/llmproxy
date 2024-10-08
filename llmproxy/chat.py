import datetime
import json
import logging

import aiohttp
import aiohttp.web
import pymongo.errors
import yarl

from . import auth


async def iter_chunks(stream):
    chunk = bytearray()
    async for data, end in stream.iter_chunks():
        logging.debug("Received chunk (end=%s): %s", end, data)
        chunk += data
        if end:
            if not chunk:
                logging.warning("Received empty chunk, assuming EOF")
                return
            yield bytes(chunk)
            chunk.clear()


async def handle_resp(f_req, b_res):
    body = await b_res.content.read()
    data = json.loads(body)
    f_res = aiohttp.web.Response(body=body)

    return f_res, data["usage"]


async def handle_resp_stream(f_req, b_res):
    app = f_req.app
    f_res = aiohttp.web.StreamResponse()

    last = {}

    try:
        await f_res.prepare(f_req)
        async for chunk in iter_chunks(b_res.content):
            last = chunk
            await f_res.write(chunk)
        await f_res.write_eof()
    except OSError as e:
        app.logger.info("Client disconnected: %s", e)

    async for chunk in iter_chunks(b_res.content):
        last = chunk

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
                f_res, usage = await handle_resp(f_req, b_res)

            try:
                await app["db"].put_event(
                    user=user,
                    time=datetime.datetime.now(datetime.UTC),
                    model=f_body["model"],
                    prompt_n=usage["prompt_tokens"],
                    completion_n=usage["completion_tokens"]
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
