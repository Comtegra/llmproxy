import datetime
import hashlib
import logging
import sqlite3
import uuid


logger = logging.getLogger(__name__)

sqlite3.register_adapter(datetime.datetime, lambda d: d.isoformat())


async def get_db(uri, req=None):
    if req is not None and "db" in req:
        return req["db"]

    if uri.startswith("mongodb://"):
        db = await MongoDatabase.create(uri)
    elif uri.startswith("sqlite://"):
        db = await SqliteDatabase.create(uri)
    else:
        raise ValueError("Unrecognized URI scheme: \"%s\"" % uri)

    if req is not None:
        req["db"] = db

    return db


class DatabaseError(Exception):
    pass


class MongoDatabase:
    @classmethod
    async def create(cls, uri):
        import motor.motor_asyncio
        import pymongo.server_api

        self = cls()

        self.db = motor.motor_asyncio.AsyncIOMotorClient(uri,
            tz_aware=True, connect=True,
            server_api=pymongo.server_api.ServerApi('1'))

        await self.db.admin.command("ping")
        logger.debug("Connected to database")

        return self

    async def close(self):
        self.db.close()
        logger.debug("Closed database connection")

    async def user_create(self, secret_hash, expires=None, comment=None):
        raise NotImplementedError("not implemented for MongoDB")

    async def user_list(self, secret_hash="", include_expired=False):
        if include_expired:
            raise NotImplementedError("not implemented for MongoDB")

        try:
            res = await self.db["cgc"]["api_keys"].find({
                "access_level": "LLM",
                "secret": secret_hash,
                "$or": [
                    {"date_expiry": None},
                    {"date_expiry": {"$gt": datetime.datetime.now(datetime.UTC)}},
                ],
            }).to_list(None)
        except pymongo.errors.PyMongoError as e:
            raise DatabaseError(e) from e

        return [{"id": x["_id"],"_user_id": x["user_id"], "secret": x["secret"],
            "expires": x.get("date_expiry", None), "comment": x.get("comment")}
            for x in res]

    async def user_update(self, user, **kwargs):
        raise NotImplementedError("not implemented for MongoDB")

    async def event_create(self, user, time, product, quantity, request_id):
        common = {"date_created": time, "user_id": user.get("_user_id"),
            "api_key_id": str(user.get("id", "")), "request_id": str(request_id)}

        try:
            await self.db["billing"]["events_oneoff"].insert_one({
                **common,
                "product": product,
                "quantity": quantity,
            })
        except pymongo.errors.PyMongoError as e:
            raise DatabaseError(e) from e


class SqliteDatabase:
    @classmethod
    async def create(cls, uri):
        import aiosqlite

        path = uri.removeprefix("sqlite://")
        assert path != uri

        self = cls()
        self.db = await aiosqlite.connect(path)
        logger.debug("Connected to database")

        self.db.row_factory = self.dict_factory

        return self

    @staticmethod
    def dict_factory(cursor, row):
        fields = [column[0] for column in cursor.description]
        return {key: value for key, value in zip(fields, row)}

    async def close(self):
        await self.db.close()
        logger.debug("Closed database connection")

    async def user_create(self, secret_hash, expires=None, comment=None):
        id_ = str(uuid.uuid4())

        try:
            await self.db.execute("""
                INSERT INTO api_key (id, secret, type, expires, comment)
                VALUES (?, ?, 'LLM', ?, ?)
                """, (id_, secret_hash, expires, comment))
        except sqlite3.DatabaseError as e:
            raise DatabaseError(e) from e

        await self.db.commit()

        return id_

    async def user_list(self, secret_hash="", include_expired=False):
        try:
            cur = await self.db.execute("""
                SELECT id, secret, expires,
                    CASE WHEN CURRENT_TIMESTAMP >= expires THEN 'expired'
                        ELSE 'active' END AS status,
                    IFNULL(comment, '') AS comment
                FROM api_key
                WHERE type = 'LLM' AND secret LIKE ? || '%'
                    AND (? OR expires > datetime() OR expires IS NULL)
                """, (secret_hash, include_expired))
        except sqlite3.DatabaseError as e:
            raise DatabaseError(e) from e

        rows = await cur.fetchall()
        await cur.close()

        return rows

    async def user_update(self, user, **kwargs):
        assert kwargs.keys() <= {"expires", "comment"}
        try:
            if "expires" in kwargs:
                await self.db.execute("""
                    UPDATE api_key SET expires = ? WHERE id = ?
                    """, (kwargs["expires"], user["id"]))
            if "comment" in kwargs:
                await self.db.execute("""
                    UPDATE api_key SET comment = ? WHERE id = ?
                    """, (kwargs["comment"], user["id"]))
        except sqlite3.DatabaseError as e:
            raise DatabaseError(e) from e

        await self.db.commit()

    async def event_create(self, user, time, product, quantity, request_id):
        try:
            await self.db.execute("""
                INSERT INTO event_oneoff (created, api_key, product, quantity, rid)
                VALUES (?, ?, ?, ?, ?)
                """, (time, user["id"], product, quantity, str(request_id)))
        except sqlite3.DatabaseError as e:
            raise DatabaseError(e) from e

        await self.db.commit()
