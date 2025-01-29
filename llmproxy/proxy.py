import json

import aiohttp
import yarl


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def request(f_req):
    app = f_req.app

    try:
        f_body = await f_req.json()
    except json.decoder.JSONDecodeError as e:
        raise aiohttp.web.HTTPBadRequest(body="JSON decode error: %s" % e)

    app.logger.debug("Frontend request body:\n%s", f_body)

    try:
        b_name = f_body["model"]
        b_cfg = app["config"]["backends"][b_name]
    except KeyError:
        raise aiohttp.web.HTTPUnauthorized(body="Incorrect model")

    b_url = yarl.URL(b_cfg["url"]) / str(f_req.rel_url)[1:]
    b_hdrs = {"Authorization": "Bearer %s" % b_cfg["token"]}

    b_body = f_body.copy()
    if (m := b_cfg.get("model")) is not None:
        b_body["model"] = m

    try:
        app.logger.debug("Sending backend request")
        return (app["client"].post(b_url, headers=b_hdrs, json=b_body),
            b_name, b_cfg)
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
