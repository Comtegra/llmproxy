"""Supported OpenAI Decisions subset translated to the Basal wire protocol.

Text and user messages containing text are translated to native Basal state.
This frontend does not provide image inference or equivalent model calibration.
"""

import json
import math
from dataclasses import dataclass

import aiohttp.web

from . import auth, config, decision_adapters, decisions, errors, proxy


PATH = "/v1/decisions"
QUESTION_TYPES = ("predicate", "choice", "score")


def supports(backend):
    return backend.get("decision_adapter", "basal") == "basal"


def _fields(request, value, allowed, param):
    if not isinstance(value, dict):
        raise errors.bad_request(request, "Expected an object.", param)
    if set(value) - set(allowed):
        raise errors.bad_request(request, "Unsupported Decisions parameter.", param)


def _text(request, value, param):
    if not isinstance(value, str):
        raise errors.bad_request(request, "Expected a text string.", param)
    return value


def _non_empty(request, text):
    # Whitespace is no evidence: the backend would still be called, and
    # billed, for a probability about nothing.
    if not text.strip():
        raise errors.bad_request(request, "input must contain text.", "input")
    return text


def input_text(request, value):
    """Join shared user evidence; do not invent role or multimodal semantics."""
    if isinstance(value, str):
        return _non_empty(request, value)
    if not isinstance(value, list) or not value:
        raise errors.bad_request(request, "input must be text or a non-empty array of user messages.", "input")
    messages = []
    for i, message in enumerate(value):
        param = "input[%d]" % i
        _fields(request, message, ("type", "role", "content"), param)
        if message.get("type", "message") != "message" or message.get("role") != "user":
            raise errors.bad_request(request, "Only user messages are supported.", param)
        content = message.get("content")
        if isinstance(content, str):
            messages.append(content)
            continue
        if not isinstance(content, list) or not content:
            raise errors.bad_request(request, "content must be text or a non-empty array of input_text parts.",
                param + ".content")
        parts = []
        for j, part in enumerate(content):
            part_param = "%s.content[%d]" % (param, j)
            _fields(request, part, ("type", "text"), part_param)
            if part.get("type") != "input_text":
                raise errors.bad_request(request,
                    "This frontend accepts text only; images, audio and files are not supported.", part_param)
            parts.append(_text(request, part.get("text"), part_param + ".text"))
        messages.append("\n".join(parts))
    return _non_empty(request, "\n\n".join(messages))


@dataclass(frozen=True)
class Question:
    kind: str
    name: str | None
    values: tuple = ()


def prepare(request, body, backend):
    """Return native validation state and request-local response metadata."""
    if not supports(backend):
        raise errors.bad_request(request, "This frontend requires the Basal adapter.", "model")
    _fields(request, body, ("model", "input", "questions"), "body")
    state = input_text(request, body.get("input"))
    questions = body.get("questions")
    limit = backend.get("decision_limits", {}).get("max_questions", config.DECISION_LIMITS["max_questions"])
    if not isinstance(questions, list) or not 1 <= len(questions) <= min(limit, 400):
        raise errors.bad_request(request, "questions must be a non-empty array within max_questions.", "questions")
    native, specs = {}, []
    for i, question in enumerate(questions):
        param = "questions[%d]" % i
        if not isinstance(question, dict):
            raise errors.bad_request(request, "Expected a question object.", param)
        kind = question.get("type")
        if kind not in QUESTION_TYPES:
            raise errors.bad_request(request, "Supported types: predicate, choice, score.", param + ".type")
        extra = {"choice": "choices", "score": "levels"}.get(kind)
        _fields(request, question, ("type", "name", "instructions") + ((extra,) if extra else ()), param)
        instructions = _text(request, question.get("instructions"), param + ".instructions")
        name = question.get("name")
        if "name" in question:
            _text(request, name, param + ".name")
        item = {"type": "noul" if kind == "predicate" else kind, "instructions": instructions}
        values, captions = [], []
        if extra:
            options = question.get(extra)
            if not isinstance(options, list) or not 2 <= len(options) <= 255:
                raise errors.bad_request(request, "Provide between 2 and 255 options.", param + "." + extra)
            for j, option in enumerate(options):
                option_param = "%s.%s[%d]" % (param, extra, j)
                field = "value" if kind == "choice" else "label"
                _fields(request, option, (field, "description"), option_param)
                value = option.get(field)
                if type(value) not in ((str, bool) if kind == "choice" else (str,)):
                    raise errors.bad_request(request, "Invalid option value type.", option_param + "." + field)
                if any(type(value) is type(previous) and value == previous for previous in values):
                    raise errors.bad_request(request, "Options must be unique.", param + "." + extra)
                description = option.get("description", "")
                _text(request, description, option_param + ".description")
                # JSON notation distinguishes boolean true from the string "true".
                caption = json.dumps(value, ensure_ascii=False) if kind == "choice" else value
                captions.append(caption + (": " + description if description else ""))
                values.append(value)
            if kind == "choice":
                item.update(criteria={"o%d" % j: text for j, text in enumerate(captions)}, option_keys="hide")
            else:
                item["levels"] = captions
        native["q%d" % i] = item
        specs.append(Question(kind, name, tuple(values)))
    try:
        prepared = decisions.prepare({"state": state, "questions": native}, backend)
    except ValueError as exc:
        raise errors.bad_request(request, str(exc), "questions") from exc
    return prepared, tuple(specs)


def render(request, alias, result, specs):
    answers = []
    for i, spec in enumerate(specs):
        native = result.answers["q%d" % i]
        probabilities = native["probabilities"]
        keys = (["true", "false"] if spec.kind == "predicate" else
            ["o%d" % j if spec.kind == "choice" else str(j) for j in range(len(spec.values))])
        # The native contract validates numbers; this frontend also requires the
        # exact requested option set, so no category can disappear or be invented.
        if set(probabilities) != set(keys) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4):
            raise ValueError("invalid_answers")
        answer = {"type": spec.kind, "name": spec.name}
        if spec.kind == "predicate":
            if not math.isclose(native["noul"], probabilities["true"], abs_tol=1e-6):
                raise ValueError("invalid_answers")
            answer["probability"] = native["noul"]
        elif spec.kind == "choice":
            answer.update(choice=spec.values[keys.index(native["choice"])], confidence=native["confidence"],
                probabilities=[{"value": value, "probability": probabilities[key]}
                    for key, value in zip(keys, spec.values)])
        else:
            expected_score = sum(j * probabilities[key] for j, key in enumerate(keys))
            if not math.isclose(native["score"], expected_score, abs_tol=1e-4):
                raise ValueError("invalid_answers")
            answer.update(score=native["score"], confidence=native["confidence"],
                probabilities=[{"label": label, "value": j, "probability": probabilities[key]}
                    for j, (key, label) in enumerate(zip(keys, spec.values))])
        answers.append(answer)
    return aiohttp.web.json_response({"model": alias, "answers": answers,
        "usage": {"input_tokens": result.usage.input_tokens,
            # These counters describe separately accounted tokens, not physical
            # cache hits. Basal supplies only aggregate input/output usage.
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": result.usage.output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": result.usage.input_tokens + result.usage.output_tokens}})


async def decide(request):
    try:
        user = await auth.require_auth(request)
    except aiohttp.web.HTTPUnauthorized as exc:
        raise errors.json_error(request, aiohttp.web.HTTPUnauthorized, "Invalid or missing API key.",
            openai_type="authentication_error", anthropic_type="authentication_error") from exc
    prepared, specs = None, None

    def transform(body, backend):
        nonlocal prepared, specs
        prepared, specs = prepare(request, body, backend)
        body.clear()
        body.update(prepared.body)

    def response(request, alias, result):
        return render(request, alias, result, specs)

    try:
        async with proxy.request(request, user=user, backend_type="decision",
                path=decision_adapters.backend_path, backend_body_transform=transform) as (upstream, alias, backend):
            return await decisions.respond(request, user, upstream, alias, backend, prepared,
                render=response, backend_error=decisions.backend_error)
    except aiohttp.web.HTTPException as exc:
        if exc.content_type != "application/json":
            raise decisions.backend_error(request, exc) from exc
        raise
