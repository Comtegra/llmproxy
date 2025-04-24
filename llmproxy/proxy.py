import json

import aiohttp
import yarl


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def request(f_req, b_name, url, body):
    app = f_req.app

    try:
        b_cfg = app["config"].get("backends", {})[b_name]
    except KeyError:
        raise aiohttp.web.HTTPUnauthorized(text="Incorrect model")

    b_url = yarl.URL(b_cfg["url"]) / url
    b_hdrs = {"Authorization": "Bearer %s" % b_cfg["token"]}

    if (m := b_cfg.get("model")) is not None and "model" in body:
        body = {**body, "model": m}

    if f_req.content_type == "application/json":
        body = json.dumps(body)
        b_hdrs["Content-Type"] = "application/json"
    elif f_req.content_type == "multipart/form-data":
        # Manually add fields to FormData as aiohttp can't serialize FileField
        d = aiohttp.FormData()
        for key, value in body.items():
            if isinstance(value, aiohttp.web.FileField):
                d.add_field(key, value.file, content_type=value.content_type,
                    filename=value.filename)
            else:
                d.add_field(key, value)
        body = d

    try:
        app.logger.debug("Sending backend request")
        ssl = None if b_cfg.get("verify_ssl", True) else False
        return (app["client"].post(b_url, headers=b_hdrs, data=body, ssl=ssl),
            b_cfg)
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
