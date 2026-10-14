"""Authenticated native Basal decisions; inference and calibration stay upstream."""

import json
import math

import aiohttp.web

from . import auth, billing, config, errors, metrics, proxy


QUESTION_TYPES = frozenset(("choice", "noul", "score", "multi", "act", "tool"))
CHAT_FIELDS = frozenset(("messages", "stream_options", "max_tokens",
    "max_completion_tokens", "max_output_tokens", "temperature", "top_p",
    "frequency_penalty", "presence_penalty", "n", "tools", "tool_choice",
    "response_format", "orders"))
MAX_TOKENS = 2 ** 63 - 1


def prepare(body, backend):
    """Validate only routing/workload structure, leaving semantics to Basal."""
    if "state" not in body:
        raise ValueError("state is required")
    if body.get("stream") is not None and body["stream"] is not False:
        raise ValueError("streaming is not supported")
    if CHAT_FIELDS.intersection(body):
        raise ValueError("unsupported decision fields: " + ", ".join(sorted(CHAT_FIELDS.intersection(body))))
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    limits = config.DECISION_LIMITS | backend.get("decision_limits", {})
    if len(questions) > limits["max_questions"]:
        raise ValueError("max_questions exceeded")
    expected = {}
    branches = 0
    for name, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError("each question must be an object")
        kind = question.get("type", "choice")
        if not isinstance(kind, str) or kind not in QUESTION_TYPES:
            raise ValueError("unsupported question type")
        multi = kind == "multi" or (kind == "tool" and bool(question.get("multi")))
        if multi:
            options = question.get("tools" if kind == "tool" else "criteria")
            if not isinstance(options, (dict, list)) or not options:
                raise ValueError("multi questions need a non-empty options collection")
            count = len(options)
            if kind == "tool":
                # Basal's tool-multi default is no extras (single tool defaults
                # to three). Do not change the definitions or build schemas.
                extra = question.get("extra", [])
                if not isinstance(extra, list):
                    raise ValueError("tool extra must be a list")
                count += len(extra)
        else:
            count = 1
        branches += count
        expected[name] = (kind, multi)
    # These are logical branches as in usage.branches. Option orders are an
    # upstream deployment setting, not a supported per-request override.
    if branches > limits["max_branches"]:
        raise ValueError("max_branches exceeded")
    body.pop("model", None)
    return expected, branches


def _number(value, probability=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (probability and not 0 <= value <= 1)):
        raise ValueError("invalid_answers")


def validate_response(data, expected):
    if not isinstance(data, dict) or not isinstance(data.get("model"), str):
        raise ValueError("invalid_response")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("invalid_usage")
    for field, lower in (("input_tokens", 1), ("output_tokens", 0)):
        value = usage.get(field)
        if type(value) is not int or not lower <= value <= MAX_TOKENS:
            raise ValueError("invalid_usage")
    answers = data.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(expected):
        raise ValueError("invalid_answers")
    for name, (kind, multi) in expected.items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError("invalid_answers")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or not probabilities:
            raise ValueError("invalid_answers")
        for value in probabilities.values():
            _number(value, probability=True)
        _number(answer.get("set_confidence" if multi else "confidence"), probability=True)
        if multi:
            selected = answer.get("tools" if kind == "tool" else "selected")
            if (not isinstance(selected, list) or any(not isinstance(key, str)
                    or key not in probabilities for key in selected)):
                raise ValueError("invalid_answers")
        elif kind in ("choice", "act", "tool"):
            field = {"choice": "choice", "act": "action", "tool": "tool"}[kind]
            value = answer.get(field)
            if kind == "tool" and value is None and answer.get("deferred") is True:
                continue
            if not isinstance(value, str):
                raise ValueError("invalid_answers")
            if kind != "act" and value not in probabilities:
                raise ValueError("invalid_answers")
            if kind == "act":
                costs = answer.get("expected_costs")
                if not isinstance(costs, dict) or value not in costs:
                    raise ValueError("invalid_answers")
                for cost in costs.values():
                    _number(cost)
        elif kind == "noul":
            _number(answer.get("noul"), probability=True)
        elif kind == "score":
            _number(answer.get("score"))
            if not 0 <= answer["score"] <= len(probabilities) - 1:
                raise ValueError("invalid_answers")
        for field in ("threshold", "expected_score"):
            if field in answer:
                _number(answer[field], probability=True)
        if "ranking" in answer:
            if not isinstance(answer["ranking"], list):
                raise ValueError("invalid_answers")
            for item in answer["ranking"]:
                if not isinstance(item, dict):
                    raise ValueError("invalid_answers")
                _number(item.get("probability"), probability=True)
    return usage


def _reject_constant(value):
    raise ValueError("invalid_json")


def _bad_gateway(request):
    return errors.json_error(request, aiohttp.web.HTTPBadGateway,
        "Invalid decision backend response.", openai_type="server_error",
        anthropic_type="api_error", code="invalid_backend_response")


async def decide(request):
    user = await auth.require_auth(request)
    expected = {}
    branches = 0

    def transform(body, backend):
        nonlocal expected, branches
        try:
            expected, branches = prepare(body, backend)
        except ValueError as exc:
            raise proxy._bad_request(request, str(exc)) from exc

    async with proxy.request(request, user=user, backend_type="decision",
            path="v1/systemone", backend_body_transform=transform) as (response, alias, backend):
        limit = backend.get("decision_limits", {}).get("max_response_bytes",
            config.DECISION_LIMITS["max_response_bytes"])
        try:
            await proxy.check_response(request.app, alias, response,
                request_id=request["request_id"], max_bytes=limit, log_body=False)
            if response.content_type != "application/json":
                raise ValueError("invalid_response")
            raw = await proxy.read_limited(response, limit)
            try:
                data = json.loads(raw, parse_constant=_reject_constant)
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise ValueError("invalid_json") from exc
            usage = validate_response(data, expected)
        except (ValueError, OverflowError, RecursionError) as exc:
            reason = str(exc) if str(exc) in {"invalid_usage", "invalid_answers",
                "invalid_json", "response_too_large", "invalid_response"} else "invalid_response"
            request.app.logger.error("Decision backend rejected: request_id=%s model=%s reason=%s",
                request["request_id"], alias, reason)
            metrics.DECISION_ERRORS_TOTAL.labels(alias, reason).inc()
            raise _bad_gateway(request) from exc
        await billing.record(request, user, {
            "%s/%s/prompt" % (alias, backend["device"]): usage["input_tokens"],
            "%s/%s/completion" % (alias, backend["device"]): usage["output_tokens"],
        })
        try:
            metrics.observe_text_tokens(alias, usage["input_tokens"], usage["output_tokens"])
            metrics.DECISION_QUESTIONS_TOTAL.labels(alias).inc(len(expected))
            metrics.DECISION_BRANCHES_TOTAL.labels(alias).inc(branches)
        except Exception:
            # Billing has committed. Observability must not turn it into a
            # failed request which a client might retry and pay for twice.
            request.app.logger.error("Decision metrics failed: request_id=%s model=%s",
                request["request_id"], alias)
        request.app.logger.info("Decision billed: request_id=%s model=%s questions=%d branches=%d input=%d output=%d",
            request["request_id"], alias, len(expected), branches,
            usage["input_tokens"], usage["output_tokens"])
        return aiohttp.web.Response(body=raw, content_type="application/json")
