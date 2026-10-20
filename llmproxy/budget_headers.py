"""Shared CORS header names (no policy imports in the stream pump)."""

# Location and Retry-After: where a background file conversion is polled and
# when (also on 429s), readable by browser clients too.
EXPOSE_HEADERS = ("X-Request-ID, X-AI-Generated, X-LLMProxy-Extensions, "
    "X-LLMProxy-Max-Output-Tokens, X-LLMProxy-Budget-Source, "
    "X-LLMProxy-Budget-Clamped, X-LLMProxy-Truncation, Location, Retry-After")
