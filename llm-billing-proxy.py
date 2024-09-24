import json
import logging
import tomllib

import aiohttp.web
import yarl

logger = logging.getLogger(__name__)

routes = aiohttp.web.RouteTableDef()


async def on_startup(app):
    timeout = aiohttp.ClientTimeout(connect=app["config"]["timeout_connect"])
    app["client"] = aiohttp.ClientSession(timeout=timeout)


async def on_cleanup(app):
    await app["client"].close()


async def iter_chunks(stream):
    chunk = bytearray()
    async for data, end in stream.iter_chunks():
        chunk += data
        if end:
            yield bytes(chunk)
            chunk.clear()


async def handle_resp(f_req, b_res):
    body = await b_res.content.read()
    data = json.loads(body)
    f_res = aiohttp.web.Response(body=body)

    return f_res, data["usage"]


async def handle_resp_stream(f_req, b_res):
    f_res = aiohttp.web.StreamResponse()

    last = {}

    try:
        await f_res.prepare(f_req)
        async for chunk in iter_chunks(b_res.content):
            last = chunk
            await f_res.write(chunk)
        await f_res.write_eof()
    except OSError as e:
        logger.info("Client disconnected: %s", e)

    async for chunk in iter_chunks(b_res.content):
        last = chunk

    _, _, body_raw = last.partition(b" ")
    body = json.loads(body_raw)

    return f_res, body["usage"]


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
@routes.post("/v1/chat/completions")
async def chat(f_req):
    app = f_req.app

    scheme, _, token = f_req.headers.get("Authorization", "").partition(" ")
    if scheme != "Bearer" or token not in app["config"]["api_keys"]:
        raise aiohttp.web.HTTPUnauthorized(body="Incorrect API key")

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
                logger.error("Backend \"%s\" error: %d %s", f_body["model"],
                    b_res.status, (await b_res.text()))
                raise aiohttp.web.HTTPBadGateway()

            if b_res.headers.get("Transfer-Encoding", "") == "chunked":
                f_res, usage = await handle_resp_stream(f_req, b_res)
            else:
                f_res, usage = await handle_resp(f_req, b_res)

            logger.info("Client used: %s", usage)

            return f_res
    except aiohttp.ServerTimeoutError as e:
        logger.error("Backend timeout: %s", e)
        raise aiohttp.web.HTTPGatewayTimeout() from e
    except (aiohttp.ClientConnectorError, aiohttp.ServerConnectionError,
            aiohttp.ClientPayloadError, aiohttp.ClientResponseError,
            aiohttp.InvalidURL) as e:
        logger.error("Backend error: %s", e)
        raise aiohttp.web.HTTPBadGateway() from e
    except aiohttp.ClientError as e:
        logger.error("HTTP client error: %s", e)
        raise aiohttp.web.HTTPInternalServerError() from e


if __name__ == "__main__":
    app = aiohttp.web.Application()

    with open("config.toml", "rb") as f:
        app["config"] = tomllib.load(f)

    log_fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=log_fmt, level=app["config"]["log_level"])

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.add_routes(routes)

    aiohttp.web.run_app(
        app=app,
        host=app["config"]["host"],
        port=app["config"]["port"],
        access_log=None,
    )
