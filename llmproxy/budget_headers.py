"""Shared CORS header names (no policy imports in the stream pump)."""

EXPOSE_HEADERS = ("X-Request-ID, X-AI-Generated, X-LLMProxy-Extensions, "
    "X-LLMProxy-Max-Output-Tokens, X-LLMProxy-Budget-Source, "
    "X-LLMProxy-Budget-Clamped, X-LLMProxy-Truncation")
