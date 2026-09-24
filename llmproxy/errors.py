"""JSON error responses in the wire format of the API the client speaks.

``/v1/messages`` is the Anthropic Messages API, whose SDKs parse
``{"type": "error", "error": {"type", "message"}}``; every other endpoint is
OpenAI-compatible (``{"error": {"message", "type", "param", "code"}}``).
Choosing the flavour here keeps that path check in one place.
"""

import json


def json_error(f_req, exc_class, message, *, openai_type, anthropic_type,
        code=None, param=None, headers=None):
    """Build (and return, not raise) ``exc_class`` with a JSON error body.

    ``openai_type``/``code``/``param`` fill the OpenAI error object;
    ``anthropic_type`` is the Anthropic error type for the same condition.
    """
    if f_req.rel_url.path == "/v1/messages":
        body = {"type": "error",
            "error": {"type": anthropic_type, "message": message}}
    else:
        body = {"error": {"message": message, "type": openai_type,
            "param": param, "code": code}}

    return exc_class(text=json.dumps(body), content_type="application/json",
        headers=headers)
