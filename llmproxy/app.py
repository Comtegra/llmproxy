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
from .db import get_db


async def check_db(app):
    db = await get_db(app["config"]["db"]["uri"])
    await db.close()

    logging.info("Database ready")


async def check_backends(app):
    for name, cfg in app["config"].get("backends", {}).items():
        try:
            ssl = None if cfg.get("verify_ssl", True) else False
            await app["client"].get(yarl.URL(cfg["url"]) / "health", ssl=ssl,
                raise_for_status=True)
            logging.info("Backend %s ready", name)
        except aiohttp.ClientError as e:
            logging.error("Backend %s not ready: %s", name, e)


async def _probe_backend_meta(app, name, cfg):
    if cfg.get("type") == "audio":
        return None

    headers = {"Authorization": "Bearer %s" % cfg["token"]}
    ssl = None if cfg.get("verify_ssl", True) else False

    try:
        async with app["client"].get(
            yarl.URL(cfg["url"]) / "v1/models",
            headers=headers, ssl=ssl, raise_for_status=True,
        ) as res:
            data = await res.json()

        if not isinstance(data, dict):
            logging.warning(
                "Backend %s: /v1/models response is not a JSON object", name)
            return None

        for entry in data.get("data", []):
            m = entry.get("max_model_len")
            if isinstance(m, int):
                return m
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError,
            AttributeError, TypeError) as e:
        logging.warning(
            "Backend %s: failed probing /v1/models: %s", name, e)
        return None

    logging.info(
        "Backend %s: no max_model_len in /v1/models response", name)
    return None


async def discover_backend_meta(app):
    backends = app["config"].get("backends", {})
    results = await asyncio.gather(
        *(_probe_backend_meta(app, n, c) for n, c in backends.items())
    )

    meta = {}
    for (name, cfg), discovered in zip(backends.items(), results):
        if discovered is None:
            continue
        meta[name] = {"context_length": discovered}
        logging.info(
            "Backend %s: discovered max_model_len=%d", name, discovered)

        if (toml_cl := cfg.get("context_length")) is not None \
                and toml_cl != discovered:
            logging.warning(
                "Backend %s: TOML context_length=%d differs from "
                "backend-reported max_model_len=%d",
                name, toml_cl, discovered)

    app["backend_meta"] = meta


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
    except ValueError as e:
        app.logger.error("Invalid reloaded config, keeping old: %s", e)
        return

    # Only backends are reloaded
    app["config"]["backends"] = cfg.get("backends", {})
    app.logger.info("Config reloaded. Configured backends: %s",
        " ".join(app["config"]["backends"]) or "none")


async def create_app(cfg):
    app = aiohttp.web.Application(
        middlewares=[assign_request_id, add_cors_headers, close_db])

    app.add_routes([
        aiohttp.web.post("/v1/chat/completions", chat.chat),
        aiohttp.web.post("/v1/completions", chat.chat),
        aiohttp.web.get("/v1/models", chat.models),
        aiohttp.web.post("/v1/embeddings", embeddings.embeddings),
        aiohttp.web.post("/v1/audio/transcriptions", audio.transcriptions),
    ])

    app["config"] = cfg

    await check_db(app)

    timeout = aiohttp.ClientTimeout(
        connect=app["config"]["timeout_connect"],
        sock_read=app["config"]["timeout_read"],
    )
    app["client"] = aiohttp.ClientSession(timeout=timeout)
    async def client_close(app):
        await app["client"].close()
    app.on_cleanup.append(client_close)

    await check_backends(app)
    await discover_backend_meta(app)

    return app


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=pathlib.Path)
parser.add_argument("--create-config", action="store_true")


def main():
    args = parser.parse_args()

    try:
        cfg = config.load(args.config, args.create_config)
    except (OSError, ValueError) as e:
        print("Failed loading config:", e, file=sys.stderr)
        sys.exit(1)

    log_fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=log_fmt, level=cfg["log_level"])

    loop = asyncio.new_event_loop()

    try:
        app = loop.run_until_complete(create_app(cfg))
    except ImportError as e:
        logging.critical("Failed to import module \"%s\"", e.name)
        sys.exit(1)

    if hasattr(signal, "SIGHUP"):
        loop.add_signal_handler(signal.SIGHUP,
            functools.partial(reload_config, app))

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
