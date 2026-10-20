import decimal
import json
import logging

import aiohttp
import aiohttp.web

from . import (
    auth, billing, budget, decision_adapters, decision_openai, errors,
    metrics, provenance, proxy, streaming,
)


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
    try:
        return await _chat(f_req)
    except aiohttp.web.HTTPUnauthorized as exc:
        if f_req.path != "/v1/chat/completions":
            raise
        raise errors.json_error(f_req, aiohttp.web.HTTPUnauthorized, "Invalid or missing API key.",
            openai_type="authentication_error", anthropic_type="authentication_error") from exc


async def _chat(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)
    async with proxy.request(f_req, user=user, backend_type="chat",
            body_transform=force_include_usage) as (b_res, b_name, b_cfg):
        app.logger.debug("Backend request completed")

        await proxy.check_response(app, b_name, b_res,
            request_id=f_req["request_id"])

        is_stream = "text/event-stream" in b_res.headers.get("Content-Type", "")
        if is_stream:
            usage_acc = _ChatStreamUsage()
            f_res = await streaming.stream_through(f_req, b_res,
                budget.observer(f_req, usage_acc),
                extra_headers={**provenance.headers(app["config"]),
                    **budget.headers(f_req, stream=True)})
            usage = usage_acc.usage()
        else:
            body = await b_res.content.read()
            data = json.loads(body, parse_float=decimal.Decimal)
            f_hdrs = {"Content-Type":
                b_res.headers.get("Content-Type", "application/octet-stream")}
            body, p_hdrs = budget.mark(f_req, body, data, b_name)
            f_hdrs.update(p_hdrs)
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

        metrics.observe_text_tokens(b_name,
            usage["prompt_tokens"], usage["completion_tokens"])
        budget.record(f_req, b_name, is_stream)

        return f_res


def model_items(cfg):
    items = []
    for model, meta in cfg.get("backends", {}).items():
        item = {
            "id": model,
            "object": "model",
            "created": None,
            "owned_by": None,
            # Which endpoints the model serves (config.BACKEND_TYPES), so a
            # client can pick the right one instead of hitting a 404.
            "type": meta["type"],
            "device": meta.get("device"),
            # The real model id / repo name from the config (the upstream
            # model card), so clients can see what we actually serve.
            "model_repo": meta.get("model"),
        }
        if "max_model_len" in meta:
            item["max_model_len"] = meta["max_model_len"]
        if meta["type"] == "decision":
            item["endpoints"] = ["/v1/systemone"]
            adapter = decision_adapters.get_adapter(meta)
            item["decision_capabilities"] = {
                "question_types": sorted(adapter.capabilities.question_types),
            }
            if decision_openai.supports(meta):
                item["endpoints"].append(decision_openai.PATH)
                item["decision_capabilities"].update(
                    openai_question_types=list(decision_openai.QUESTION_TYPES),
                    openai_input="text_or_user_messages", openai_streaming=False)
        item.update(budget.catalog(cfg, meta))
        items.append(item)
    return items


async def models(req):
    await auth.require_auth(req)
    items = model_items(req.app["config"])

    data = {
        "object": "list",
        "data": items,
    }

    return aiohttp.web.Response(text=json.dumps(data),
        content_type="application/json")


async def model(req):
    await auth.require_auth(req)
    for item in model_items(req.app["config"]):
        if item["id"] == req.match_info["id"]:
            return aiohttp.web.json_response(item)
    raise proxy._model_error(req, "The model does not exist.", "model_not_found")
