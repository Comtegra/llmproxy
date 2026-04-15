import datetime
import decimal
import json

import aiohttp

from . import auth, proxy
from .db import DatabaseError, get_db


def force_verbose(body):
    if body.get("response_format") not in (None, "json", "verbose_json"):
        raise aiohttp.web.HTTPUnprocessableEntity(
            text="response_format must be one of: json, verbose_json")
    body["response_format"] = "verbose_json"


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def transcriptions(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)

    b_req, b_name, b_cfg = await proxy.request(f_req, force_verbose)
    async with b_req as b_res:
        app.logger.debug("Backend request completed")

        await proxy.check_response(app, b_name, b_res)

        body = await b_res.content.read()
        data = json.loads(body, parse_float=decimal.Decimal)
        f_hdrs = {"Content-Type":
            b_res.headers.get("Content-Type", "application/octet-stream")}
        f_res = aiohttp.web.Response(body=body, headers=f_hdrs)

        db = await get_db(app["config"]["db"]["uri"], f_req)
        res = {"%s/%s/transcription" % (b_name, b_cfg["device"]):
            data["duration"]}
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

        app.logger.info("Client used: %d s of %s", data["duration"], b_name)

        return f_res
