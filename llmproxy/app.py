import argparse
import asyncio
import functools
import logging
import pathlib
import signal
import ssl
import sys
import uuid

import aiohttp.web
import yarl

from . import audio, chat, config, embeddings


async def check_backends(app):
    for name, cfg in app["config"]["backends"].items():
        try:
            ssl = None if cfg.get("verify_ssl", True) else False
            await app["client"].get(yarl.URL(cfg["url"]) / "health", ssl=ssl,
                raise_for_status=True)
            logging.info("Backend %s ready", name)
        except aiohttp.ClientError as e:
            logging.error("Backend %s not ready: %s", name, e)


async def on_startup(app):
    timeout = aiohttp.ClientTimeout(
        connect=app["config"]["timeout_connect"],
        sock_read=app["config"]["timeout_read"],
    )
    app["client"] = aiohttp.ClientSession(timeout=timeout)
    await check_backends(app)


async def on_cleanup(app):
    await app["client"].close()


@aiohttp.web.middleware
async def assign_request_id(req, handler):
    req["request_id"] = uuid.uuid4()
    return await handler(req)


@aiohttp.web.middleware
async def add_cors_headers(req, handler):
    try:
        res = await handler(req)
    except aiohttp.web.HTTPMethodNotAllowed as e:
        if e.method != "OPTIONS":
            raise
        res = aiohttp.web.Response()
    res.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    if o := req.app["config"].get("http_origin"):
        res.headers["Access-Control-Allow-Origin"] = o
    return res


@aiohttp.web.middleware
async def close_db(req, handler):
    try:
        res = await handler(req)
    finally:
        db = req.pop("db", None)
        if db is not None:
            await db.close()

    return res


def reload_config(app):
    try:
        cfg = config.load(app["config"]["_path"])
    except OSError as e:
        app.logger.error("Failed reloading config: %s", e)
        return

    # Only backends are reloaded
    app["config"]["backends"] = cfg["backends"]
    app.logger.info("Config reloaded. Configured backends: %s",
        " ".join(app["config"]["backends"]))


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=pathlib.Path)


def main():
    args = parser.parse_args()

    app = aiohttp.web.Application(
        middlewares=[assign_request_id, add_cors_headers, close_db])

    try:
        app["config"] = config.load(args.config)
    except OSError as e:
        print("Failed loading config:", e, file=sys.stderr)
        sys.exit(1)

    log_fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=log_fmt, level=app["config"]["log_level"])

    loop = asyncio.new_event_loop()

    if hasattr(signal, "SIGHUP"):
        loop.add_signal_handler(signal.SIGHUP,
            functools.partial(reload_config, app))

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.add_routes([
        aiohttp.web.post("/v1/chat/completions", chat.chat),
        aiohttp.web.get("/v1/models", chat.models),
        aiohttp.web.post("/v1/embeddings", embeddings.embeddings),
        aiohttp.web.post("/v1/audio/transcriptions", audio.transcriptions),
    ])

    ssl_ctx = None
    if (cert := app["config"].get("cert")) and (key := app["config"].get("key")):
        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(cert, key)

    aiohttp.web.run_app(
        app=app,
        host=app["config"]["host"],
        port=app["config"]["port"],
        ssl_context=ssl_ctx,
        access_log_format="%a \"%r\" %s %Tfs",
        loop=loop,
    )
