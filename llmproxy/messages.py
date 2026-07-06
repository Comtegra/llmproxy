"""Anthropic Messages API (`POST /v1/messages`) — native forward + usage.

Gateway to the agentic ecosystem (Claude Code, the Anthropic SDK). vLLM serves
`/v1/messages` natively; we forward and meter usage ourselves.

Anthropic SPLITS usage across streaming events: `input_tokens` (+ cache
creation/read) arrives in `message_start`, and the CUMULATIVE `output_tokens`
in the final `message_delta`. So we take input from `message_start` and output
from the LAST `message_delta` — never sum output (it is cumulative) and never
double-count input. Mapped to the existing `model/device/prompt|completion`
products (cache tokens folded into prompt); no new rate.
"""

import decimal
import json

import aiohttp
import aiohttp.web

from . import auth, billing, metrics, proxy, streaming


def _input_tokens(usage):
    """Anthropic input billed as prompt = input + cache creation + cache read.

    input_tokens is REQUIRED (fail loud on absence, like chat/responses; a
    present 0 is legitimate for a fully cache-read turn); cache_* are optional
    (may be absent, e.g. vLLM #45079).
    """
    return (usage["input_tokens"]
        + (usage.get("cache_creation_input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0))


class _MessagesStreamUsage:
    """on_chunk accumulator for Anthropic streams: input from `message_start`,
    cumulative output from the last `message_delta`.

    Fail-loud: if a truncated stream never delivers either, usage() raises
    rather than silently billing 0 — matching chat/responses, which raise on
    missing usage instead of under-billing a paying customer.
    """

    def __init__(self):
        self._start_usage = None
        self._output = None

    def __call__(self, chunk):
        obj = streaming.parse_sse_event(chunk)
        if not isinstance(obj, dict):
            return
        event_type = obj.get("type")
        if event_type == "message_start":
            self._start_usage = (obj.get("message") or {}).get("usage") or {}
        elif event_type == "message_delta":
            usage = obj.get("usage") or {}
            if usage.get("output_tokens") is not None:
                self._output = usage["output_tokens"]  # cumulative; keep last

    def usage(self):
        if self._start_usage is None or self._output is None:
            raise ValueError("missing Anthropic usage in stream "
                "(no message_start or final message_delta)")
        # _input_tokens requires input_tokens -> fail loud (not silent 0) if a
        # present message_start usage omits it. Computed here (after the stream
        # forwards) rather than in __call__, so the client still gets the body.
        return {"prompt_tokens": _input_tokens(self._start_usage),
            "completion_tokens": self._output}


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
async def messages(f_req):
    app = f_req.app

    user = await auth.require_auth(f_req)

    async with proxy.request(f_req) as (b_res, b_name, b_cfg):
        app.logger.debug("Backend request completed")

        await proxy.check_response(app, b_name, b_res,
            request_id=f_req["request_id"])

        if "text/event-stream" in b_res.headers.get("Content-Type", ""):
            usage_acc = _MessagesStreamUsage()
            f_res = await streaming.stream_through(f_req, b_res, usage_acc)
            usage = usage_acc.usage()
        else:
            body = await b_res.content.read()
            data = json.loads(body, parse_float=decimal.Decimal)
            f_hdrs = {"Content-Type":
                b_res.headers.get("Content-Type", "application/octet-stream")}
            f_res = aiohttp.web.Response(body=body, headers=f_hdrs)
            # Fail loud on a usage-less 200 (like chat.py's data["usage"]),
            # never silently under-bill to zero.
            u = data["usage"]
            usage = {"prompt_tokens": _input_tokens(u),
                "completion_tokens": u["output_tokens"]}

        await billing.record(f_req, user, {
            "%s/%s/prompt" % (b_name, b_cfg["device"]):
                usage["prompt_tokens"],
            "%s/%s/completion" % (b_name, b_cfg["device"]):
                usage["completion_tokens"],
        })

        app.logger.info("Client used (messages): P:%d C:%d of %s",
            usage["prompt_tokens"], usage["completion_tokens"], b_name)

        metrics.observe_text_tokens(b_name,
            usage["prompt_tokens"], usage["completion_tokens"])

        return f_res
