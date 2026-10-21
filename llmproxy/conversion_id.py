"""Opaque, owner-bound ids for background file conversions.

The proxy keeps no state for a background conversion -- there is no table for
it in the billing database. Everything a poll needs to reach the job (the
backend alias and the backend's own request id) travels inside the id, and an
HMAC tag binds the id to the API key that submitted it:

    fc_<b64url(json [backend, request_id])>.<b64url(tag)>

so another key, a forged id or a single flipped character resolves to "not
found" without the backend ever being asked. The tag key is derived from the
backend's token: it is already a secret shared only with that backend, stays
the same across restarts and replicas, and needs no new configuration.
Rotating the token invalidates the ids of conversions still open.

Pure functions only (no aiohttp), so the edge cases are cheap to test.
"""

import base64
import collections
import hashlib
import hmac
import json
import re


PREFIX = "fc_"

# Generous for a backend alias plus a request id, and small enough that a
# client cannot make the proxy decode arbitrary amounts of data.
MAX_LENGTH = 512

# The backend's request id for a background job: 32 random hex digits. It
# becomes a URL path segment on the poll, so nothing that could escape the
# segment ("/", "..", "%", "?") may get through; and an older file-processor
# answers with a 16-hex content hash instead, which is not a per-job id it can
# claim deliveries for.
REQUEST_ID_RE = re.compile(r"[0-9a-f]{32}")

_KEY_CONTEXT = b"llmproxy/files-convert/job-id/v1"
_TAG_SIZE = 16
_ID_RE = re.compile(r"fc_([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)")

ParsedId = collections.namedtuple("ParsedId",
    ("backend", "request_id", "payload", "tag"))


def _b64encode(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text):
    """Strict base64url without padding; None unless ``text`` is canonical.

    The canonical check means every id has exactly one spelling, so changing
    any character of a valid id always makes it invalid (otherwise unused low
    bits of the last character could be flipped without changing the bytes).
    """
    try:
        data = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except ValueError:
        return None
    if _b64encode(data) != text:
        return None
    return data


def _tag(token, payload, owner):
    key = hmac.new(token.encode(), _KEY_CONTEXT, hashlib.sha256).digest()
    # NUL separators keep the fields unambiguous: the JSON payload never
    # contains a raw NUL, and the owner comes last.
    msg = b"\0".join((_KEY_CONTEXT, payload, str(owner).encode()))
    return hmac.new(key, msg, hashlib.sha256).digest()[:_TAG_SIZE]


def make(token, owner, backend, request_id):
    """Build the id for ``request_id`` on ``backend``, owned by ``owner``.

    ``owner`` is the submitting API key's id; ``token`` the backend's token.
    """
    if not REQUEST_ID_RE.fullmatch(request_id):
        raise ValueError("request_id is not 32 hex digits")
    payload = json.dumps([backend, request_id],
        separators=(",", ":")).encode()
    return "%s%s.%s" % (PREFIX, _b64encode(payload),
        _b64encode(_tag(token, payload, owner)))


def parse(job_id):
    """Split an id into its parts, or None if it is malformed.

    NOT authenticated: the result only says which backend's token to check
    the tag against (verify). Never act on it before verify passes.
    """
    if len(job_id) > MAX_LENGTH:
        return None
    m = _ID_RE.fullmatch(job_id)
    if m is None:
        return None
    payload, tag = _b64decode(m[1]), _b64decode(m[2])
    if payload is None or tag is None or len(tag) != _TAG_SIZE:
        return None
    try:
        fields = json.loads(payload)
    except ValueError:
        return None
    if (not isinstance(fields, list) or len(fields) != 2
            or not all(isinstance(f, str) for f in fields)
            or not fields[0]
            or not REQUEST_ID_RE.fullmatch(fields[1])):
        return None
    return ParsedId(fields[0], fields[1], payload, tag)


def verify(token, owner, parsed):
    """Whether ``parsed`` was made with ``token`` for ``owner``."""
    if not isinstance(token, str) or not token:
        return False
    return hmac.compare_digest(parsed.tag, _tag(token, parsed.payload, owner))
