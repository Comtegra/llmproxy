import json

import aiohttp
import yarl


def get_backend_cfg(app, name):
    try:
        return app["config"].get("backends", {})[name]
    except KeyError:
        raise aiohttp.web.HTTPUnauthorized(text="Incorrect model")


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def request(f_req, backend_cfg, url, body):
    app = f_req.app

    b_url = yarl.URL(backend_cfg["url"]) / url
    b_hdrs = {"Authorization": "Bearer %s" % backend_cfg["token"]}

    if (m := backend_cfg.get("model")) is not None and "model" in body:
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
        ssl = None if backend_cfg.get("verify_ssl", True) else False
        return app["client"].post(b_url, headers=b_hdrs, data=body, ssl=ssl)
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
