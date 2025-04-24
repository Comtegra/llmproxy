import datetime
import json

import aiohttp

from . import auth, proxy
from .db import DatabaseError, get_db


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def embeddings(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)

    if f_req.content_type != "application/json":
        raise aiohttp.web.HTTPUnsupportedMediaType()

    try:
        f_body = await f_req.json()
    except json.decoder.JSONDecodeError as e:
        raise aiohttp.web.HTTPBadRequest(text="JSON decode error: %s" % e)

    app.logger.debug("Frontend request body:\n%s", f_body)

    b_name = f_body.get("model")
    b_cfg = proxy.get_backend_cfg(app, b_name)

    async with proxy.request(app, b_cfg, "v1/embeddings", f_body) as b_res:
        app.logger.debug("Backend request completed")

        if b_res.status != 200:
            app.logger.error("Backend \"%s\" error: %d %s", b_name,
                b_res.status, (await b_res.text()))
            raise aiohttp.web.HTTPBadGateway()

        body = await b_res.content.read()
        data = json.loads(body)
        f_hdrs = {"Content-Type":
            b_res.headers.get("Content-Type", "application/octet-stream")}
        f_res = aiohttp.web.Response(body=body, headers=f_hdrs)
        usage = data["usage"]

        db = await get_db(app["config"]["db"]["uri"], f_req)
        try:
            await db.event_create(
                user=user,
                time=datetime.datetime.now(datetime.UTC),
                product="%s/%s/embedding" % (b_name, b_cfg["device"]),
                quantity=usage["prompt_tokens"],
                request_id=f_req["request_id"],
            )
        except DatabaseError as e:
            app.logger.critical(e)
            raise aiohttp.web.GracefulExit() from e

        app.logger.info("Client used: P:%d C:%d tokens of %s",
            usage["prompt_tokens"], 0,
            b_name)

        return f_res
