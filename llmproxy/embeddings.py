import decimal
import json

import aiohttp

from . import auth, billing, proxy


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def embeddings(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)

    async with proxy.request(f_req) as (b_res, b_name, b_cfg):
        app.logger.debug("Backend request completed")

        await proxy.check_response(app, b_name, b_res,
            request_id=f_req["request_id"])

        body = await b_res.content.read()
        data = json.loads(body, parse_float=decimal.Decimal)
        f_hdrs = {"Content-Type":
            b_res.headers.get("Content-Type", "application/octet-stream")}
        f_res = aiohttp.web.Response(body=body, headers=f_hdrs)
        usage = data["usage"]

        await billing.record(f_req, user, {
            "%s/%s/embedding" % (b_name, b_cfg["device"]):
                usage["prompt_tokens"],
        })

        app.logger.info("Client used: P:%d C:%d tokens of %s",
            usage["prompt_tokens"], 0, b_name)

        return f_res
