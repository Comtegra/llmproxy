import datetime
import decimal
import hashlib
import importlib.resources
import logging
import sqlite3
import uuid

logger = logging.getLogger(__name__)

sqlite3.register_adapter(datetime.datetime, lambda d: d.isoformat())
sqlite3.register_adapter(decimal.Decimal, float)


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
        import bson
        import motor.motor_asyncio
        import pymongo.errors
        import pymongo.server_api

        class DecimalCodec(bson.codec_options.TypeCodec):
            python_type = decimal.Decimal
            bson_type = bson.decimal128.Decimal128

            def transform_python(self, value):
                # If the decimal can be represented by an exact integer, use it
                with decimal.localcontext() as ctx:
                    ctx.clear_flags()
                    i = value.to_integral_exact(context=ctx)
                    if not ctx.flags[decimal.Inexact]:
                        return int(i)

                return bson.decimal128.Decimal128(value)

            def transform_bson(self, value):
                return value.to_decimal()

        self = cls()

        self.db = motor.motor_asyncio.AsyncIOMotorClient(uri,
            tz_aware=True, connect=True,
            server_api=pymongo.server_api.ServerApi('1'))

        treg = bson.codec_options.TypeRegistry([DecimalCodec()])
        self.copt = bson.codec_options.CodecOptions(type_registry=treg)

        await self.db.admin.command("ping")
        logger.debug("Connected to database")

        return self

    async def close(self):
        self.db.close()
        logger.debug("Closed database connection")

    async def user_create(self, secret_hash, expires=None, comment=None):
        raise NotImplementedError("not implemented for MongoDB")

    async def user_list(self, secret_hash="", include_expired=False):
        import pymongo.errors

        if secret_hash == "":
            raise NotImplementedError("not implemented for MongoDB")
        if include_expired:
            raise NotImplementedError("not implemented for MongoDB")

        try:
            key = await self.db["cgc"]["api_keys"].find_one({
                "access_level": "LLM",
                "secret": secret_hash,
                "$or": [
                    {"date_expiry": None},
                    {"date_expiry": {"$gt": datetime.datetime.now(datetime.UTC)}},
                ],
            })
            if key is None:
                return []

            user = await self.db["cgc"]["rest_users"].find_one(
                {"_id": key["user_id"]})
            if user is None:
                return []
        except pymongo.errors.PyMongoError as e:
            raise DatabaseError(e) from e

        return [{
            "id": key["_id"],
            "_user_id": key["user_id"],
            "_namespace": user["namespace"],
            "_org_id": str(user.get("org_id", "")),
            "_tier": user.get("subscription_level", ""),
            "secret": key["secret"],
            "expires": key.get("date_expiry", None),
            "comment": key.get("comment"),
        }]

    async def user_update(self, user, **kwargs):
        raise NotImplementedError("not implemented for MongoDB")

    async def billing_record_add(self, user, time, resources, request_id):
        import pymongo.errors

        col = self.db["cgc"].get_collection("billing_record",
            codec_options=self.copt)

        try:
            await col.insert_one({
                "cluster_id": None,
                "namespace": user["_namespace"],
                "name": "",
                "revision": 1,
                "created_at": time,
                "deleted_at": None,
                "resources": resources,
                "type": "oneoff",
                "k8s_uid": None,
                "user_id": user["_user_id"],
                "api_key_id": user["id"],
                "org_id": user["_org_id"],
                "tier": user["_tier"],
                "revision_synced": None,
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

        cur = await self.db.execute("PRAGMA user_version")
        ver, = await cur.fetchone()
        if ver == 0:
            schema = importlib.resources.files("llmproxy") \
                .joinpath("schema.sql").read_text()
            await self.db.executescript(schema)
            await self.db.execute("PRAGMA user_version = 1")
            logger.info("Initialized database to version 1")

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

    async def billing_record_add(self, user, time, resources, request_id):
        try:
            for name, quant in resources.items():
                await self.db.execute("""
                    INSERT INTO event_oneoff
                        (created, api_key, product, quantity, rid)
                    VALUES (?, ?, ?, ?, ?)
                    """, (time, user["id"], name, quant, str(request_id)))
        except sqlite3.DatabaseError as e:
            raise DatabaseError(e) from e

        await self.db.commit()
