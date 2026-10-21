"""Shared decision service: transport, validated billing and native HTTP responses."""

import json

import aiohttp.web

from . import auth, billing, config, decision_adapters, errors, metrics, proxy


def _reject_constant(value):
    raise ValueError("invalid_json")


def _bad_gateway(request):
    return errors.json_error(request, aiohttp.web.HTTPBadGateway,
        "Invalid decision backend response.", openai_type="server_error",
        anthropic_type="api_error", code="invalid_backend_response")


async def decide(request):
    user = await auth.require_auth(request)
    prepared = None

    def transform(body, backend):
        nonlocal prepared
        try:
            prepared = prepare(body, backend)
            body.clear()
            body.update(prepared.body)
        except ValueError as exc:
            raise errors.bad_request(request, str(exc)) from exc

    async with proxy.request(request, user=user, backend_type="decision",
            path=decision_adapters.backend_path, backend_body_transform=transform) as (response, alias, backend):
        return await respond(request, user, response, alias, backend, prepared)


def prepare(body, backend):
    limits = config.DECISION_LIMITS | backend.get("decision_limits", {})
    return decision_adapters.get_adapter(backend).prepare(body, limits)


async def respond(request, user, response, alias, backend, prepared, *, render=None, backend_error=None):
    """Validate and render before committing one billing record, for either frontend."""
    limit = backend.get("decision_limits", {}).get("max_response_bytes",
        config.DECISION_LIMITS["max_response_bytes"])
    try:
        try:
            await proxy.check_response(request.app, alias, response,
                request_id=request["request_id"], max_bytes=limit, log_body=False)
        except aiohttp.web.HTTPException as exc:
            if backend_error:
                raise backend_error(request, exc) from exc
            raise
        if response.content_type != "application/json":
            raise ValueError("invalid_response")
        raw = await proxy.read_limited(response, limit)
        try:
            data = json.loads(raw, parse_constant=_reject_constant)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("invalid_json") from exc
        result = prepared.decode(data)
        usage = result.usage
    except (ValueError, OverflowError, RecursionError) as exc:
        reason = str(exc) if str(exc) in {"invalid_usage", "invalid_answers",
            "invalid_json", "response_too_large", "invalid_response"} else "invalid_response"
        request.app.logger.error("Decision backend rejected: request_id=%s model=%s reason=%s",
            request["request_id"], alias, reason)
        metrics.DECISION_ERRORS_TOTAL.labels(alias, reason).inc()
        raise _bad_gateway(request) from exc
    try:
        frontend = (render(request, alias, result) if render else
            aiohttp.web.Response(body=raw, content_type="application/json"))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise _bad_gateway(request) from exc
    await billing.record(request, user, {
        "%s/%s/prompt" % (alias, backend["device"]): usage.input_tokens,
        "%s/%s/completion" % (alias, backend["device"]): usage.output_tokens,
    })
    try:
        metrics.observe_text_tokens(alias, usage.input_tokens, usage.output_tokens)
        metrics.DECISION_QUESTIONS_TOTAL.labels(alias).inc(prepared.question_count)
        metrics.DECISION_BRANCHES_TOTAL.labels(alias).inc(prepared.branches)
    except Exception:
        # Billing has committed. Observability must not turn it into a
        # failed request which a client might retry and pay for twice.
        request.app.logger.error("Decision metrics failed: request_id=%s model=%s",
            request["request_id"], alias)
    request.app.logger.info("Decision billed: request_id=%s model=%s questions=%d branches=%d input=%d output=%d",
        request["request_id"], alias, prepared.question_count, prepared.branches,
        usage.input_tokens, usage.output_tokens)
    return frontend


def backend_error(request, exc):
    """Normalize errors without returning upstream documents or credentials."""
    status = exc.status
    if status in (401, 403):
        # Authentication already succeeded at the proxy. These statuses now
        # describe a misconfigured backend credential, not the client's key.
        status = 502
    code = None
    if status in (400, 422):
        context = proxy.looks_like_context_length_error(exc.body or b"")
        code = "context_length_exceeded" if context else "invalid_decision_request"
        message = "Context length exceeded." if context else "Decision backend rejected the request."
        kind = "invalid_request_error"
    else:
        kind = "rate_limit_error" if status == 429 else "server_error"
        message = "Decision backend request failed."
    error = errors.json_error(request, aiohttp.web.HTTPBadGateway, message,
        openai_type=kind, anthropic_type="api_error", code=code,
        headers={"Retry-After": exc.headers["Retry-After"]} if "Retry-After" in exc.headers else None)
    error.set_status(status)
    return error
