import aiohttp.web
import pymongo.errors


async def require_auth(req):
    scheme, _, token = req.headers.get("Authorization", "").partition(" ")
    if scheme != "Bearer":
        raise aiohttp.web.HTTPUnauthorized(body="Unsupported authorization scheme")

    try:
        user = await req.app["db"].get_user(token)
    except pymongo.errors.PyMongoError as e:
        req.app.logger.critical(e)
        raise aiohttp.web.GracefulExit() from e

    if scheme != "Bearer" or not user:
        raise aiohttp.web.HTTPUnauthorized(body="Incorrect API key")

    return user
