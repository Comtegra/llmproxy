import decimal
import json
import logging

import aiohttp
import aiohttp.web

from . import auth, billing, metrics, proxy, streaming


class _ChatStreamUsage:
    """on_chunk accumulator for OpenAI chat streams: keep the last non-[DONE]
    SSE block and expose its ``usage`` object. OpenAI reports usage in the final
    chunk when ``stream_options.include_usage`` is set, so keeping the last block
    also captures a usage chunk that arrives separately after ``finish_reason``.
    """

    def __init__(self):
        self._last = b""

    def __call__(self, chunk):
        if chunk != b"data: [DONE]\n\n":
            self._last = chunk

    def usage(self):
        _, _, raw = self._last.partition(b" ")
        try:
            data = json.loads(raw, parse_float=decimal.Decimal)
        except json.JSONDecodeError:
            logging.error("Failed parsing usage information: %s", raw)
            raise
        return data["usage"]


def force_include_usage(body):
    if body.get("stream"):
        body["stream_options"] = {"include_usage": True}


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def chat(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)

    async with proxy.request(f_req, force_include_usage) as (
            b_res, b_name, b_cfg):
        app.logger.debug("Backend request completed")

        await proxy.check_response(app, b_name, b_res,
            request_id=f_req["request_id"])

        if "text/event-stream" in b_res.headers.get("Content-Type", ""):
            usage_acc = _ChatStreamUsage()
            f_res = await streaming.stream_through(f_req, b_res, usage_acc)
            usage = usage_acc.usage()
        else:
            body = await b_res.content.read()
            data = json.loads(body, parse_float=decimal.Decimal)
            f_hdrs = {"Content-Type":
                b_res.headers.get("Content-Type", "application/octet-stream")}
            f_res = aiohttp.web.Response(body=body, headers=f_hdrs)
            usage = data["usage"]

        await billing.record(f_req, user, {
            "%s/%s/prompt" % (b_name, b_cfg["device"]):
                usage["prompt_tokens"],
            "%s/%s/completion" % (b_name, b_cfg["device"]):
                usage["completion_tokens"],
        })

        app.logger.info("Client used: P:%d C:%d tokens of %s",
            usage["prompt_tokens"], usage["completion_tokens"],
            b_name)

        metrics.TOKENS_TOTAL.labels(b_name, "prompt").inc(
            usage["prompt_tokens"])
        metrics.TOKENS_TOTAL.labels(b_name, "completion").inc(
            usage["completion_tokens"])

        return f_res


async def models(req):
    await auth.require_auth(req)

    models = []
    for model, meta in req.app["config"].get("backends", {}).items():
        item = {
            "id": model,
            "object": "model",
            "created": None,
            "owned_by": None,
            "device": meta.get("device"),
        }
        if "max_model_len" in meta:
            item["max_model_len"] = meta["max_model_len"]
        models.append(item)

    data = {
        "object": "list",
        "data": models,
    }

    return aiohttp.web.Response(text=json.dumps(data),
        content_type="application/json")
