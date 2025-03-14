import json

import aiohttp
import yarl


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def request(f_req, body_transform=None):
    app = f_req.app

    if f_req.content_type == "application/json":
        try:
            f_body = await f_req.json()
        except json.decoder.JSONDecodeError as e:
            raise aiohttp.web.HTTPBadRequest(text="JSON decode error: %s" % e)
    elif f_req.content_type == "multipart/form-data":
        f_body = await f_req.post()
    else:
        raise aiohttp.web.HTTPUnsupportedMediaType()

    app.logger.debug("Frontend request body:\n%s", f_body)

    try:
        b_name = f_body["model"]
        b_cfg = app["config"].get("backends", {})[b_name]
    except KeyError:
        raise aiohttp.web.HTTPUnauthorized(text="Incorrect model")

    b_url = yarl.URL(b_cfg["url"]) / str(f_req.rel_url)[1:]
    b_hdrs = {"Authorization": "Bearer %s" % b_cfg["token"]}

    b_body = f_body.copy()
    if (m := b_cfg.get("model")) is not None:
        b_body["model"] = m

    if body_transform is not None:
        body_transform(b_body)

    if f_req.content_type == "application/json":
        b_body = json.dumps(b_body)
        b_hdrs["Content-Type"] = "application/json"
    elif f_req.content_type == "multipart/form-data":
        # Manually add fields to FormData as aiohttp can't serialize FileField
        d = aiohttp.FormData()
        for key, value in b_body.items():
            if isinstance(value, aiohttp.web.FileField):
                d.add_field(key, value.file, content_type=value.content_type,
                    filename=value.filename)
            else:
                d.add_field(key, value)
        b_body = d

    try:
        app.logger.debug("Sending backend request")
        ssl = None if b_cfg.get("verify_ssl", True) else False
        return (app["client"].post(b_url, headers=b_hdrs, data=b_body, ssl=ssl),
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
