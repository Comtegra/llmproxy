import hashlib

import aiohttp.web

from .db import DatabaseError, get_db


async def require_auth(req):
    db = await get_db(req.app["config"]["db"]["uri"], req)

    scheme, _, token = req.headers.get("Authorization", "").partition(" ")
    if scheme != "Bearer":
        raise aiohttp.web.HTTPUnauthorized(text="Unsupported authorization scheme")

    digest = hashlib.sha256(token.encode()).hexdigest()

    try:
        rows = await db.user_list(digest)
    except DatabaseError as e:
        req.app.logger.critical(e)
        raise aiohttp.web.GracefulExit() from e

    if not rows:
        raise aiohttp.web.HTTPUnauthorized(text="Incorrect API key")

    return rows[0]
