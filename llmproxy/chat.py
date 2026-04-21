import datetime
import decimal
import json
import logging

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


async def handle_resp_stream(f_req, b_res):
    app = f_req.app
    headers = {"Content-Type":
        b_res.headers.get("Content-Type", "application/octet-stream")}
    f_res = aiohttp.web.StreamResponse(headers=headers)

    f_res.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    if o := app["config"].get("http_origin"):
        f_res.headers["Access-Control-Allow-Origin"] = o

    last = b""

    try:
        await f_res.prepare(f_req)
        while (c := await readuntil(b_res.content, b"\n\n")):
            if c != b"data: [DONE]\n\n":
                logging.debug("Received chunk: %s", c)
                last = c
            await f_res.write(c)
        await f_res.write_eof()
    except OSError as e:
        app.logger.info("Client disconnected: %s", e)

    while (c := await readuntil(b_res.content, b"\n\n")):
        if c != b"data: [DONE]\n\n":
            logging.debug("Received chunk after client disconnected: %s", c)
            last = c

    _, _, body_raw = last.partition(b" ")

    try:
        body = json.loads(body_raw, parse_float=decimal.Decimal)
    except json.decoder.JSONDecodeError:
        app.logger.error("Failed parsing usage information: %s", body_raw)
        raise

    return f_res, body["usage"]


def force_include_usage(body):
    if body.get("stream"):
        body["stream_options"] = {"include_usage": True}


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def chat(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)

    b_req, b_name, b_cfg = await proxy.request(f_req, force_include_usage)
    async with b_req as b_res:
        app.logger.debug("Backend request completed")

        await proxy.check_response(app, b_name, b_res)

        if b_res.headers.get("Transfer-Encoding", "") == "chunked":
            f_res, usage = await handle_resp_stream(f_req, b_res)
        else:
            body = await b_res.content.read()
            data = json.loads(body, parse_float=decimal.Decimal)
            f_hdrs = {"Content-Type":
                b_res.headers.get("Content-Type", "application/octet-stream")}
            f_res = aiohttp.web.Response(body=body, headers=f_hdrs)
            usage = data["usage"]

        db = await get_db(app["config"]["db"]["uri"], f_req)
        res = {
            "%s/%s/prompt" % (b_name, b_cfg["device"]):
                usage["prompt_tokens"],
            "%s/%s/completion" % (b_name, b_cfg["device"]):
                usage["completion_tokens"],
        }
        try:
            await db.billing_record_add(
                user=user,
                time=datetime.datetime.now(datetime.UTC),
                resources=res,
                request_id=f_req["request_id"],
            )
        except DatabaseError as e:
            app.logger.critical(e)
            raise aiohttp.web.GracefulExit() from e

        app.logger.info("Client used: P:%d C:%d tokens of %s",
            usage["prompt_tokens"], usage["completion_tokens"],
            b_name)

        return f_res


def _card_url(cfg):
    if (u := cfg.get("model_url")) is not None:
        return u
    return "https://huggingface.co/%s" % cfg["model"]


def _context_length(cfg, meta):
    if (cl := cfg.get("context_length")) is not None:
        return cl
    return meta.get("context_length")


def _model_entry(alias, cfg, meta):
    entry = {
        "id": alias,
        "object": "model",
        "created": None,
        "owned_by": None,
        "type": cfg["type"],
        "source_model": cfg["model"],
        "card_url": _card_url(cfg),
    }
    if (q := cfg.get("quantization")) is not None:
        entry["quantization"] = q
    if cfg["type"] in ("chat", "embedding"):
        if (cl := _context_length(cfg, meta)) is not None:
            entry["context_length"] = cl
    return entry


async def models(req):
    await auth.require_auth(req)

    backends = req.app["config"].get("backends", {})
    backend_meta = req.app.get("backend_meta", {})

    data = {
        "object": "list",
        "data": [
            _model_entry(alias, cfg, backend_meta.get(alias, {}))
            for alias, cfg in backends.items()
        ],
    }

    return aiohttp.web.Response(text=json.dumps(data),
        content_type="application/json")
