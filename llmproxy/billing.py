import datetime

import aiohttp.web

from .db import DatabaseError, get_db


async def record(f_req, user, resources):
    """Write one billing record for a successful request.

    Centralizes the get_db + billing_record_add + DatabaseError->GracefulExit
    block that every billed endpoint shares, so usage extraction is the only
    per-endpoint concern. On DB failure we kill the worker (GracefulExit)
    rather than silently drop revenue.
    """
    app = f_req.app

    db = await get_db(app["config"]["db"]["uri"], f_req)
    try:
        await db.billing_record_add(
            user=user,
            time=datetime.datetime.now(datetime.UTC),
            resources=resources,
            request_id=f_req["request_id"],
        )
    except DatabaseError as e:
        app.logger.critical(e)
        raise aiohttp.web.GracefulExit() from e
