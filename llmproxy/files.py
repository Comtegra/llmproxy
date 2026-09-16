import json

import aiohttp

from . import auth, billing, metrics, provenance, proxy


# Marker Datalab-compat path (sync PDF -> Markdown). Not path-preserving:
# the public route is /v1/files/convert.
MARKER_PATH = "api/v1/marker"


def prepare_marker_body(body):
    """Validate upload fields and strip llmproxy-only form keys.

    Marker expects ``file`` (and optional ``output_format``); ``model`` is only
    used by the proxy for backend selection / billing.
    """
    file_field = body.get("file")
    if not isinstance(file_field, aiohttp.web.FileField):
        raise aiohttp.web.HTTPUnprocessableEntity(text="file is required")

    body.pop("model", None)
    if "output_format" not in body:
        body["output_format"] = "markdown"


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def convert(f_req):
    """POST /v1/files/convert -- sync PDF upload to Markdown via marker."""
    app = f_req.app

    user = await auth.require_auth(f_req)

    async with proxy.request(
            f_req, prepare_marker_body, user=user, path=MARKER_PATH) as (
                b_res, b_name, b_cfg):
        app.logger.debug("Backend request completed")

        await proxy.check_response(app, b_name, b_res,
            request_id=f_req["request_id"])

        body = await b_res.content.read()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            app.logger.error(
                "File-convert backend %s returned non-JSON: request_id=%s",
                b_name, f_req["request_id"])
            raise aiohttp.web.HTTPBadGateway(
                text="File conversion backend returned invalid JSON")

        # Datalab Marker returns HTTP 200 even on conversion failure, with
        # success=false. Do not bill those.
        if not isinstance(data, dict) or data.get("success") is not True:
            app.logger.error(
                "File-convert backend %s reported failure: request_id=%s "
                "body_preview=%s",
                b_name, f_req["request_id"],
                body[:1024].decode("utf-8", errors="replace"))
            exc = aiohttp.web.HTTPUnprocessableEntity(
                content_type="application/json")
            exc.body = body
            raise exc

        f_hdrs = {"Content-Type":
            b_res.headers.get("Content-Type", "application/json")}
        # Header-only marking: forward Marker's JSON (may already include its
        # own provenance object) byte-for-byte, same carve-out pattern as audio.
        f_hdrs.update(provenance.headers(app["config"]))
        # Forward Marker AI-processed headers when present.
        for h in ("X-AI-Processed", "X-AI-Generated"):
            if h in b_res.headers and h not in f_hdrs:
                f_hdrs[h] = b_res.headers[h]

        f_res = aiohttp.web.Response(body=body, headers=f_hdrs)

        await billing.record(f_req, user, {
            "%s/%s/conversion" % (b_name, b_cfg["device"]): 1,
        })

        app.logger.info("Client used: 1 conversion of %s", b_name)
        metrics.FILE_CONVERSIONS_TOTAL.labels(b_name).inc()

        return f_res
