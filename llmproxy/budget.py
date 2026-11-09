"""Opt-in output-budget policy and best-effort diagnostics.

Metadata describes what the proxy forwarded, never an unobserved backend
default. SSE is observed on the read side only; billing keeps its own parser.
"""

import copy
import logging

from . import metrics, provenance, streaming
from .budget_headers import EXPOSE_HEADERS


PATHS = {
    "/v1/chat/completions": "chat",
    "/v1/completions": "completions",
    "/v1/responses": "responses",
    "/v1/messages": "messages",
}


def active(cfg, backend):
    return (cfg.get("budget", {}).get("enabled", False)
        and "output_budget" in backend)


def catalog(cfg, backend):
    if not active(cfg, backend):
        return {}
    policy = backend["output_budget"]
    enforce = policy.get("mode", "observe") == "enforce"
    reasoning = backend.get("reasoning", {})
    return {"extensions": ["budget.v1"], "output_budget": {
        "mode": policy.get("mode", "observe"),
        "default_max_tokens": policy.get("default_max_tokens")
            if enforce and policy.get("apply_default", False) else None,
        "max_output_tokens_ceiling": policy.get("ceiling"),
        "clamp_enabled": enforce and policy.get("clamp", False),
        "backend_default_max_tokens": policy.get("backend_default_max_tokens"),
        "backend_default_source": "operator_config",
        "effective_max_output_tokens": None,
        "reasoning": {"native_budget": reasoning.get("native_budget", False),
            "endpoints": reasoning.get("endpoints", ["chat"]),
            "reserve_content_tokens": reasoning.get("reserve_content_tokens"),
            "effort_budgets": reasoning.get("effort_budgets", {}),
            "dialect": reasoning.get("dialect", "native")},
    }}


def positive_int(value):
    return type(value) is int and value > 0


def effort_for_budget(budget, effort_budgets):
    """Dialect ladder: the highest configured effort level whose cap fits
    the computed token budget (smaller budget -> lower effort). The floor
    is the smallest configured effort, so a budget below every cap maps to
    the lowest effort instead of skipping levels."""
    ordered = sorted(effort_budgets.items(), key=lambda kv: kv[1])
    chosen = ordered[0][0]
    for name, cap in ordered:
        if cap <= budget:
            chosen = name
    return chosen


class BudgetError(ValueError):
    pass


class Contract:
    """One request's immutable config snapshot and bounded observations."""

    def __init__(self, cfg, backend, protocol, body):
        self.options = copy.deepcopy(cfg.get("budget", {}))
        self.policy = copy.deepcopy(backend["output_budget"])
        self.reasoning = copy.deepcopy(backend.get("reasoning", {}))
        self.protocol = protocol
        self.stops = {}
        self.raw_usage = None
        self.reasoning_observed = False
        self.metadata = {}
        self._apply(body)

    def _apply(self, body):
        protocol = self.protocol
        keys = {"chat": ("max_completion_tokens", "max_tokens"),
            "completions": ("max_tokens",), "messages": ("max_tokens",),
            "responses": ("max_output_tokens",)}[protocol]
        enforce = self.policy.get("mode", "observe") == "enforce"
        present = [key for key in keys if key in body]
        # Conflicts/invalid values stay backend-owned in observation mode.
        # Enforcement must not guess which contradictory limit to honor.
        if enforce and len(present) > 1:
            raise BudgetError("Send only one of max_tokens and max_completion_tokens.")
        key = present[0] if present else keys[0]
        requested = body.get(key)
        if enforce and present and not positive_int(requested):
            raise BudgetError("%s must be a positive integer." % key)
        forwarded = requested if positive_int(requested) and len(present) == 1 else None
        source = "client" if forwarded is not None else "unknown"
        clamped = False
        if enforce:
            if not present and self.policy.get("apply_default", False):
                forwarded = self.policy["default_max_tokens"]
                source = "proxy"
                body[key] = forwarded
            if forwarded is not None and self.policy.get("clamp", False):
                ceiling = self.policy["ceiling"]
                clamped = forwarded > ceiling
                forwarded = min(forwarded, ceiling)
                body[key] = forwarded
        self.metadata = {"mode": self.policy.get("mode", "observe"),
            "parameter": key, "requested_max_output_tokens": requested if positive_int(requested) else None,
            "forwarded_max_output_tokens": forwarded,
            "effective_max_output_tokens": None, "source": source,
            "clamped": clamped}
        if enforce:
            self._apply_reasoning(body, forwarded)

    def _apply_reasoning(self, body, forwarded):
        cfg = self.reasoning
        if (not cfg.get("native_budget", False)
                or self.protocol != "chat"):
            return
        effort = body.get("reasoning_effort")
        if effort is not None and (type(effort) is not str or effort not in
                ("none", "minimal", "low", "medium", "high", "xhigh")):
            raise BudgetError("reasoning_effort must be a supported effort string.")
        thinking = body.get("thinking")
        template = body.get("chat_template_kwargs")
        if cfg.get("dialect", "native") == "effort_enum" and effort is None:
            kwargs_effort = template.get("reasoning_effort") \
                if isinstance(template, dict) else None
            if isinstance(kwargs_effort, str) and kwargs_effort:
                # A client speaking the dialect directly through template
                # kwargs is honored, not silently overwritten.
                effort = kwargs_effort
                if effort not in ("none", "minimal", "low", "medium", "high", "xhigh"):
                    raise BudgetError("reasoning_effort must be a supported effort string.")
        if (effort == "none"
                or isinstance(thinking, dict) and thinking.get("type") == "disabled"
                or isinstance(template, dict) and template.get("enable_thinking") is False):
            return
        native = body.get("thinking_token_budget")
        if "thinking_token_budget" in body and not positive_int(native):
            raise BudgetError("thinking_token_budget must be a positive integer.")
        requested = native
        source = "client" if native is not None else "proxy"
        # Native client budget takes precedence over protocol-specific effort.
        if native is None:
            native = cfg.get("effort_budgets", {}).get(effort or cfg.get("default_effort"))
        reserve = cfg.get("reserve_content_tokens")
        if reserve is not None and forwarded is not None:
            available = forwarded - reserve
            if available <= 0:
                raise BudgetError("Output budget must exceed reserve_content_tokens "
                    "when native reasoning budgeting is enabled.")
            native = min(native, available) if native is not None else available
        if native is not None and cfg.get("dialect", "native") == "effort_enum":
            # Dialect injection: the budget that the native computation
            # selected maps onto the effort ladder. Both channels are
            # written consistently (recipe accepts either; vLLM precedence
            # then cannot disagree). thinking_token_budget is NEVER written
            # — the client's own field stays verbatim and the backend sees
            # only the enum.
            chosen = effort_for_budget(native, cfg.get("effort_budgets", {}))
            body["reasoning_effort"] = chosen
            body.setdefault("chat_template_kwargs", {})["reasoning_effort"] = chosen
            self.metadata["reasoning"] = {"dialect": "effort_enum",
                "parameter": "reasoning_effort",
                "requested_tokens": requested, "requested_effort": effort,
                "forwarded_tokens": native, "forwarded_effort": chosen,
                "source": source, "reserve_content_tokens": reserve}
            return
        if native is not None:
            body["thinking_token_budget"] = native
            self.metadata["reasoning"] = {"parameter": "thinking_token_budget",
                "requested_tokens": requested, "forwarded_tokens": native,
                "source": source, "reserve_content_tokens": reserve}

    def observe(self, data):
        if not isinstance(data, dict):
            return
        if self.protocol in ("chat", "completions"):
            for choice in data.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message") or choice.get("delta") or {}
                if isinstance(message, dict) and (message.get("reasoning") or message.get("reasoning_content")):
                    self.reasoning_observed = True
                index = choice.get("index", 0)
                # Bound memory even for a misbehaving backend.
                if (type(index) is int and 0 <= index < 128
                        and isinstance(choice.get("finish_reason"), str)
                        and choice["finish_reason"]):
                    self.stops[index] = choice["finish_reason"]
            usage = data.get("usage")
        elif self.protocol == "responses":
            if data.get("type") == "response.reasoning_text.delta" and data.get("delta"):
                self.reasoning_observed = True
            obj = data.get("response", data)
            if not isinstance(obj, dict):
                return
            for item in obj.get("output") or []:
                if isinstance(item, dict) and item.get("type") == "reasoning" and (item.get("content") or item.get("summary")):
                    self.reasoning_observed = True
            if obj.get("status") in ("completed", "incomplete", "failed"):
                detail = obj.get("incomplete_details") or {}
                self.stops[0] = detail.get("reason", obj["status"])
            usage = obj.get("usage")
        else:
            obj = data.get("message", data)
            if not isinstance(obj, dict):
                return
            delta = obj.get("delta") or {}
            if isinstance(delta, dict) and delta.get("thinking"):
                self.reasoning_observed = True
            for item in obj.get("content") or []:
                if isinstance(item, dict) and item.get("type") == "thinking" and item.get("thinking"):
                    self.reasoning_observed = True
            stop = obj.get("stop_reason") or delta.get("stop_reason")
            if isinstance(stop, str) and stop:
                self.stops[0] = stop
            usage = obj.get("usage")
        if isinstance(usage, dict):
            # Retain only counts used by this contract, not large backend
            # per-turn arrays, cached details or any generated text.
            observed = self.raw_usage if self.raw_usage is not None else {}
            for key in ("completion_tokens", "output_tokens"):
                if key in usage:
                    value = usage[key]
                    observed[key] = value if type(value) is int else None
            for key in ("completion_tokens_details", "output_tokens_details"):
                if key in usage:
                    details = usage[key]
                    if not isinstance(details, dict):
                        observed[key] = {}
                        continue
                    observed[key] = {name: details.get(name) if type(details.get(name)) is int else None
                        for name in ("reasoning_tokens", "thinking_tokens") if name in details}
            self.raw_usage = observed

    def on_chunk(self, chunk):
        try:
            self.observe(streaming.parse_sse_event(chunk))
        except (TypeError, ValueError, AttributeError):
            # Diagnostic errors must never interfere with forwarding/billing.
            logging.warning("Unable to parse budget stream diagnostics")

    def diagnostic(self):
        usage = self.raw_usage or {}
        chat = self.protocol in ("chat", "completions")
        total = usage.get("completion_tokens" if chat else "output_tokens")
        details = usage.get("completion_tokens_details" if chat else "output_tokens_details")
        details = details if isinstance(details, dict) else {}
        reasoning = details.get("reasoning_tokens", details.get("thinking_tokens"))
        valid = (type(total) is int and type(reasoning) is int
            and 0 <= reasoning <= total)
        inconsistent = valid and reasoning == 0 and self.reasoning_observed
        valid = valid and not inconsistent
        reasons = ("length", "max_tokens", "max_output_tokens")
        truncated = any(stop in reasons for stop in self.stops.values())
        return {"occurred": truncated if self.stops else None,
            "reason": "unknown" if truncated or not self.stops else None,
            "backend_stop_reasons": [{"index": index, "reason": reason}
                for index, reason in sorted(self.stops.items())],
            "details_available": valid,
            "details_status": "present" if valid else "inconsistent" if inconsistent else "missing_or_invalid",
            "reasoning_tokens": reasoning if valid else None,
            "non_reasoning_output_tokens": total - reasoning if valid else None}

    def headers(self, *, stream=False):
        if not self.options.get("echo_headers", True):
            return {}
        result = {"X-LLMProxy-Extensions": "budget.v1",
            "X-LLMProxy-Budget-Source": self.metadata["source"],
            "X-LLMProxy-Budget-Clamped": str(self.metadata["clamped"]).lower(),
            "X-LLMProxy-Truncation": "pending" if stream else
                ("unknown" if self.diagnostic()["occurred"] is not False else "none")}
        forwarded = self.metadata["forwarded_max_output_tokens"]
        if forwarded is not None:
            result["X-LLMProxy-Max-Output-Tokens"] = str(forwarded)
        return result

    def fields(self):
        if not self.options.get("echo_body", True):
            return {}
        return {"llmproxy": {"version": 1, "budget": self.metadata,
            "truncation": self.diagnostic()}}

    def record(self, model, request_id, is_stream):
        try:
            diagnostic = self.diagnostic()
            metrics.BUDGET_SOURCE_TOTAL.labels(model, self.metadata["source"],
                str(self.metadata["clamped"]).lower()).inc()
            if diagnostic["occurred"]:
                metrics.TRUNCATION_TOTAL.labels(model, "unknown", str(is_stream).lower()).inc()
            if diagnostic["details_available"]:
                metrics.REASONING_TOKENS_TOTAL.labels(model).inc(diagnostic["reasoning_tokens"])
                metrics.REASONING_DETAILS_TOTAL.labels(model, "present").inc()
            else:
                metrics.REASONING_DETAILS_TOTAL.labels(model, diagnostic["details_status"]).inc()
            logging.info("Output budget request_id=%s model=%s stream=%s budget=%s truncation=%s",
                request_id, model, is_stream, self.metadata, diagnostic)
        except Exception:
            logging.exception("Budget telemetry failed (request_id=%s)", request_id)


def prepare(req, backend, body):
    protocol = PATHS.get(req.path)
    if protocol and active(req.app["config"], backend):
        req["budget_contract"] = Contract(req.app["config"], backend, protocol, body)


def headers(req, *, stream=False):
    contract = req.get("budget_contract")
    return contract.headers(stream=stream) if contract else {}


def observer(req, usage_acc):
    contract = req.get("budget_contract")
    if contract is None:
        return usage_acc

    def on_chunk(chunk):
        usage_acc(chunk)  # Billing's fail-loud parser remains authoritative.
        contract.on_chunk(chunk)
    return on_chunk


def mark(req, body, data, model):
    contract = req.get("budget_contract")
    fields = {}
    if contract:
        try:
            contract.observe(data)
            fields = contract.fields()
        except (TypeError, ValueError, AttributeError):
            logging.warning("Unable to parse budget response diagnostics")
    result, hdrs = provenance.mark(req.app["config"], body, data, model,
        req["request_id"], extra_fields=fields)
    hdrs.update(headers(req))
    return result, hdrs


def record(req, model, is_stream):
    contract = req.get("budget_contract")
    if contract:
        contract.record(model, req["request_id"], is_stream)
