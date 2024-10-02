import argparse
import asyncio
import datetime
import functools
import hashlib
import json
import logging
import pathlib
import signal
import sys
import tomllib

import aiohttp.web
import motor.motor_asyncio
import pymongo
import yarl

routes = aiohttp.web.RouteTableDef()


class Database:
    def __init__(self, cfg):
        self.db = motor.motor_asyncio.AsyncIOMotorClient(cfg["uri"],
            tz_aware=True, connect=True,
            server_api=pymongo.server_api.ServerApi('1'))

    def check(self):
        return self.db.admin.command("ping")

    def get_user(self, api_key):
        return self.db["cgc"]["api_keys"].find_one({
            "access_level": "COMPLETION",
            "secret": hashlib.sha256(api_key.encode()).hexdigest(),
            "$or": [
                {"date_expiry": None},
                {"date_expiry": {"$gt": datetime.datetime.now(datetime.UTC)}},
            ],
        })

    def put_event(self, user, time, model, prompt_n, completion_n):
        return self.db["billing"]["events_completion"].insert_one({
            "date_created": time,
            "user_id": user.get("user_id"),
            "api_key_id": str(user.get("_id", "")),
            "model": model,
            "prompt_n": prompt_n,
            "completion_n": completion_n,
        })


async def on_startup(app):
    app["db"] = Database(app["config"]["db"])
    await app["db"].check()

    timeout = aiohttp.ClientTimeout(connect=app["config"]["timeout_connect"])
    app["client"] = aiohttp.ClientSession(timeout=timeout)


async def on_cleanup(app):
    await app["client"].close()


def reload_config(app, config_path):
    try:
        with config_path.open("rb") as f:
            # Only backends are reloaded
            cfg = tomllib.load(f)
            app["config"]["backends"] = cfg["backends"]
            app.logger.info("Config reloaded. Configured backends: %s",
                " ".join(app["config"]["backends"]))
    except OSError as e:
        app.logger.error("Failed reloading config: %s", e)


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
    body = json.loads(body_raw)

    return f_res, body["usage"]


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
@routes.post("/v1/chat/completions")
async def chat(f_req):
    app = f_req.app

    scheme, _, token = f_req.headers.get("Authorization", "").partition(" ")
    if scheme != "Bearer":
        raise aiohttp.web.HTTPUnauthorized(body="Unsupported authorization scheme")

    try:
        user = await app["db"].get_user(token)
    except pymongo.errors.PyMongoError as e:
        app.logger.critical(e)
        raise aiohttp.web.GracefulExit() from e

    if scheme != "Bearer" or not user:
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


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=pathlib.Path, default="config.toml")

if __name__ == "__main__":
    args = parser.parse_args()

    app = aiohttp.web.Application()

    try:
        with args.config.open("rb") as f:
            app["config"] = tomllib.load(f)
    except OSError as e:
        print("Failed loading config:", e, file=sys.stderr)
        sys.exit(1)

    log_fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=log_fmt, level=app["config"]["log_level"])

    loop = asyncio.new_event_loop()
    loop.add_signal_handler(signal.SIGHUP,
        functools.partial(reload_config, app, args.config))

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.add_routes(routes)

    aiohttp.web.run_app(
        app=app,
        host=app["config"]["host"],
        port=app["config"]["port"],
        access_log_format="%a \"%r\" %s %Tfs",
        loop=loop,
    )
