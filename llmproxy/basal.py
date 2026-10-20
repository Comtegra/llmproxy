"""Basal wire protocol: native question validation, workload and calibrated answers."""

import math
from dataclasses import dataclass

from .decision_contract import DecisionCapabilities, DecisionResult, DecisionUsage


QUESTION_TYPES = frozenset(("choice", "noul", "score", "multi", "act", "tool"))
CHAT_FIELDS = frozenset(("messages", "stream_options", "max_tokens",
    "max_completion_tokens", "max_output_tokens", "temperature", "top_p",
    "frequency_penalty", "presence_penalty", "n", "tools", "tool_choice",
    "response_format", "orders"))
MAX_TOKENS = 2 ** 63 - 1


def prepare(body, limits):
    """Validate only routing/workload structure, leaving semantics to Basal."""
    body = body.copy()
    if "state" not in body:
        raise ValueError("state is required")
    if body.get("stream") is not None and body["stream"] is not False:
        raise ValueError("streaming is not supported")
    if CHAT_FIELDS.intersection(body):
        raise ValueError("unsupported decision fields: " + ", ".join(sorted(CHAT_FIELDS.intersection(body))))
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
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
    return BasalRequest(body, expected, branches)


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


@dataclass(frozen=True)
class BasalRequest:
    body: dict
    expected: dict
    branches: int

    @property
    def question_count(self):
        return len(self.expected)

    def decode(self, data):
        usage = validate_response(data, self.expected)
        return DecisionResult(data["model"], data["answers"],
            DecisionUsage(usage["input_tokens"], usage["output_tokens"]), data)


class BasalAdapter:
    path = "v1/systemone"
    capabilities = DecisionCapabilities(QUESTION_TYPES)

    def prepare(self, body, limits):
        return prepare(body, limits)


ADAPTER = BasalAdapter()
