import collections
import datetime
import hashlib
import time

import aiohttp.web

from . import metrics
from .db import DatabaseError, get_db

# Default TTL (seconds) for the in-process auth cache. 0 disables it.
DEFAULT_CACHE_TTL = 5

# Negative results (unknown key) are cached too, otherwise a flood of invalid
# keys still costs one database round trip each. Their TTL is capped
# independently of auth_cache_ttl so that a key created moments after someone
# probed for it does not stay rejected for a long configured TTL.
NEGATIVE_CACHE_TTL = 5

# The cache is an LRU because digests of *unknown* keys are attacker-controlled:
# without a cap, negative caching hands over unbounded memory growth.
CACHE_MAX_ENTRIES = 8192

# digest -> (row | None, expires_at | None, monotonic deadline).
# row is None for a negative entry; expires_at is kept alongside the row rather
# than inside it so the dict handed to callers is identical on hit and miss.
_cache = collections.OrderedDict()

# Sentinel: this row's expires could not be parsed, so it must never be cached
# (we cannot re-evaluate expiry in-process and would risk honouring a stale key).
_UNPARSEABLE = object()


def flush_cache():
    """Drop every cached lookup.

    Wired to SIGHUP via reload_config: the TTL bounds how long a REVOKED key
    keeps working, and this is the operator's lever to make that immediate
    without restarting the proxy."""
    _cache.clear()


def _normalize_expires(value):
    """Coerce a row's ``expires`` into an aware UTC datetime, or None.

    SQLite hands back an ISO string (datetimes are stored via the isoformat
    adapter registered in db.py); Mongo hands back a datetime that is already
    aware because the client is built with tz_aware=True. Normalizing once, at
    insertion, keeps the per-request check to a single comparison."""
    if value is None:
        return None

    if isinstance(value, str):
        try:
            value = datetime.datetime.fromisoformat(value)
        except ValueError:
            return _UNPARSEABLE

    if not isinstance(value, datetime.datetime):
        return _UNPARSEABLE

    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC)

    return value


def _cache_get(digest, now):
    entry = _cache.get(digest)
    if entry is None:
        return None

    if entry[2] <= now:
        del _cache[digest]
        return None

    _cache.move_to_end(digest)
    return entry


def _cache_put(digest, row, expires_at, ttl, now):
    _cache[digest] = (row, expires_at, now + ttl)
    _cache.move_to_end(digest)
    while len(_cache) > CACHE_MAX_ENTRIES:
        _cache.popitem(last=False)


async def require_auth(req):
    scheme, _, token = req.headers.get("Authorization", "").partition(" ")
    if scheme != "Bearer":
        raise aiohttp.web.HTTPUnauthorized(text="Unsupported authorization scheme")

    digest = hashlib.sha256(token.encode()).hexdigest()

    ttl = req.app["config"].get("auth_cache_ttl", DEFAULT_CACHE_TTL)
    now = time.monotonic()

    if ttl > 0 and (entry := _cache_get(digest, now)) is not None:
        metrics.AUTH_CACHE_HITS_TOTAL.inc()
        row, expires_at, _ = entry
        if row is None:
            raise aiohttp.web.HTTPUnauthorized(text="Incorrect API key")
        # Expiry is re-evaluated here rather than trusted from the cached
        # lookup. user_list filters expired keys in the query, so a cached row
        # would otherwise keep an expired key alive for the whole TTL — which
        # matters most for exactly the short-lived keys expiry exists for.
        if expires_at is not None \
                and expires_at <= datetime.datetime.now(datetime.UTC):
            del _cache[digest]
            raise aiohttp.web.HTTPUnauthorized(text="Incorrect API key")
        # A copy, so a handler mutating the user dict cannot poison the cache.
        return dict(row)

    metrics.AUTH_CACHE_MISSES_TOTAL.inc()

    # Only on a miss do we need a database connection at all; on the hit path
    # above the request never opens one.
    db = await get_db(req.app["config"]["db"]["uri"], req)

    try:
        rows = await db.user_list(digest)
    except DatabaseError as e:
        req.app.logger.critical(e)
        raise aiohttp.web.GracefulExit() from e

    if not rows:
        if ttl > 0:
            _cache_put(digest, None, None, min(ttl, NEGATIVE_CACHE_TTL), now)
        raise aiohttp.web.HTTPUnauthorized(text="Incorrect API key")

    row = rows[0]

    if ttl > 0:
        expires_at = _normalize_expires(row.get("expires"))
        if expires_at is not _UNPARSEABLE:
            _cache_put(digest, dict(row), expires_at, ttl, now)

    return row
