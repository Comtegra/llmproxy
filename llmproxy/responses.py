"""OpenAI Responses API (`POST /v1/responses`) — stateless forward + usage.

Driver: Codex CLI (speaks only Responses, sends the full history in `input`,
`store:false`). We forward to vLLM's native `/v1/responses` and meter usage
ourselves. Usage is reported ONLY in the terminal `response.completed` /
`response.incomplete` event (streaming) or top-level `usage` (non-stream),
under `input_tokens`/`output_tokens` — NOT chat's `usage.prompt_tokens`. We map
those to the existing `model/device/prompt|completion` products (no new rate).

Stateful use (server-side `store` + `previous_response_id` / `conversation` /
`background`) needs a cross-replica conversation store we do not have, so it is
rejected fail-loud with a 400 rather than silently dropped (a stateful client
would otherwise get a context-less answer and only notice on the 2nd turn).
"""

import decimal
import json

import aiohttp
import aiohttp.web

from . import auth, billing, proxy, streaming


# Terminal Responses events carrying final usage. `response.incomplete`
# (max_output_tokens / content filter) and `response.failed` (a run that
# generated billable output before failing) carry usage and are billed too.
RESPONSES_TERMINAL_EVENTS = ("response.completed", "response.incomplete",
    "response.failed")


def usage_from_event(obj):
    """Extract `response.usage` from a terminal Responses event, else None."""
    if not isinstance(obj, dict):
        return None
    if obj.get("type") not in RESPONSES_TERMINAL_EVENTS:
        return None

    response = obj.get("response")
    if not isinstance(response, dict):
        return None

    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    return usage


def to_billing_tokens(usage):
    """Map a Responses usage object to (prompt_tokens, completion_tokens).

    `output_tokens` already includes reasoning tokens (they are a subset of
    `output_tokens_details.reasoning_tokens`), so completion = output_tokens
    with no addition; cached/reasoning are returned for information only.
    """
    in_details = usage.get("input_tokens_details") or {}
    out_details = usage.get("output_tokens_details") or {}
    return {
        # Required (fail loud on absence, like messages/chat); a present 0 is
        # legitimate (empty output / fully-cached input).
        "prompt_tokens": usage["input_tokens"],
        "completion_tokens": usage["output_tokens"],
        "cached_tokens": in_details.get("cached_tokens", 0),
        "reasoning_tokens": out_details.get("reasoning_tokens", 0),
    }


class StatefulNotSupported(ValueError):
    """Request relies on server-side state this stateless proxy has no store
    for. The handler maps this to 400 (not 500)."""


# Responses fields that require a server-side conversation store.
STATEFUL_FIELDS = ("previous_response_id", "conversation")


def force_stateless(body):
    """body_transform for /v1/responses (fail-loud).

    - `store:true` (OpenAI default) with no stateful field -> silently `false`
      (harmless: the client resends the history, only losing retrieval-by-id).
    - `previous_response_id` / `conversation` / `background:true` ->
      StatefulNotSupported (-> 400), NEVER a silent drop.
    """
    for field in STATEFUL_FIELDS:
        if body.get(field) is not None:
            raise StatefulNotSupported(
                "Stateful Responses ('%s') is not supported on this endpoint. "
                "Send the full conversation in 'input', or use a stateful "
                "backend." % field)
    if body.get("background"):
        raise StatefulNotSupported(
            "background:true (async Responses) is not supported: no server-side "
            "store to poll by id.")
    body["store"] = False


class _ResponsesStreamUsage:
    """on_chunk accumulator: capture usage from the terminal Responses event.

    Keyed by event `type` (not stream position); the Responses stream has no
    `data: [DONE]` sentinel, the terminal event itself is the end.
    """

    def __init__(self):
        self._usage = None

    def __call__(self, chunk):
        found = usage_from_event(streaming.parse_sse_event(chunk))
        if found is not None:
            self._usage = found

    def usage(self):
        if self._usage is None:
            raise ValueError("no terminal usage event in Responses stream")
        return self._usage


def _bill_map(billed, b_name, device):
    return {
        "%s/%s/prompt" % (b_name, device): billed["prompt_tokens"],
        "%s/%s/completion" % (b_name, device): billed["completion_tokens"],
    }


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def responses(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)

    try:
        async with proxy.request(f_req, force_stateless) as (
                b_res, b_name, b_cfg):
            app.logger.debug("Backend request completed")

            await proxy.check_response(app, b_name, b_res,
                request_id=f_req["request_id"])

            if "text/event-stream" in b_res.headers.get("Content-Type", ""):
                usage_acc = _ResponsesStreamUsage()
                f_res = await streaming.stream_through(f_req, b_res, usage_acc)
                usage = usage_acc.usage()
            else:
                body = await b_res.content.read()
                data = json.loads(body, parse_float=decimal.Decimal)
                f_hdrs = {"Content-Type":
                    b_res.headers.get("Content-Type",
                        "application/octet-stream")}
                f_res = aiohttp.web.Response(body=body, headers=f_hdrs)
                usage = data["usage"]

            billed = to_billing_tokens(usage)
            await billing.record(f_req, user,
                _bill_map(billed, b_name, b_cfg["device"]))

            app.logger.info("Client used (responses): P:%d C:%d of %s",
                billed["prompt_tokens"], billed["completion_tokens"], b_name)

            return f_res
    except StatefulNotSupported as e:
        raise aiohttp.web.HTTPBadRequest(
            text=json.dumps({"error": {
                "message": str(e),
                "type": "invalid_request_error",
                "code": "stateful_not_supported",
            }}),
            content_type="application/json") from e
