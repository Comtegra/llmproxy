import argparse
import asyncio
import functools
import logging
import pathlib
import signal
import ssl
import sys
import tomllib

import aiohttp.web
import yarl

from . import chat, db


async def check_backends(app):
    for name, cfg in app["config"]["backends"].items():
        async with app["client"].get(yarl.URL(cfg["url"]) / "health") as res:
            if res.ok:
                logging.info("Backend %s ready", name)
            else:
                logging.error("Backend %s not ready: %d %s",
                    name, res.status, res.reason)


async def on_startup(app):
    app["db"] = db.Database(app["config"]["db"])
    await app["db"].check()

    timeout = aiohttp.ClientTimeout(connect=app["config"]["timeout_connect"])
    app["client"] = aiohttp.ClientSession(timeout=timeout)
    await check_backends(app)


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


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=pathlib.Path, default="config.toml")


def main():
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
    app.add_routes([aiohttp.web.post("/v1/chat/completions", chat.chat)])

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
