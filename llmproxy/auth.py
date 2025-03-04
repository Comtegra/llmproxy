import hashlib

import aiohttp.web
import pymongo.errors

from .db import get_db


async def require_auth(req):
    db = await get_db(req.app["config"]["db"]["uri"], req)

    scheme, _, token = req.headers.get("Authorization", "").partition(" ")
    if scheme != "Bearer":
        raise aiohttp.web.HTTPUnauthorized(body="Unsupported authorization scheme")

    digest = hashlib.sha256(token.encode()).hexdigest()

    try:
        rows = await db.user_list(digest)
    except pymongo.errors.PyMongoError as e:
        req.app.logger.critical(e)
        raise aiohttp.web.GracefulExit() from e

    if not rows:
        raise aiohttp.web.HTTPUnauthorized(body="Incorrect API key")

    return rows[0]
