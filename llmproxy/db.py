import asyncio
import datetime
import hashlib
import sqlite3

import aiosqlite
import motor.motor_asyncio
import pymongo.server_api

async def get_db(req):
    if "db" not in req:
        uri = req.app["config"]["db"]["uri"]

        if uri.startswith("mongodb://"):
            req["db"] = await MongoDatabase.create(uri, req.app.logger)
        elif uri.startswith("sqlite://"):
            req["db"] = await SqliteDatabase.create(uri, req.app.logger)
        else:
            raise ValueError("Unrecognized URI scheme: \"%s\"" % uri)

    return req["db"]


class DatabaseError(Exception):
    pass


class MongoDatabase:
    @classmethod
    async def create(cls, uri, logger):
        self = cls()

        self.logger = logger
        self.db = motor.motor_asyncio.AsyncIOMotorClient(uri,
            tz_aware=True, connect=True,
            server_api=pymongo.server_api.ServerApi('1'))

        await self.db.admin.command("ping")
        self.logger.debug("Connected to database")

        return self

    async def close(self):
        self.db.close()
        self.logger.debug("Closed database connection")

    async def get_user(self, api_key):
        try:
            return await self.db["cgc"]["api_keys"].find_one({
                "access_level": "LLM",
                "secret": hashlib.sha256(api_key.encode()).hexdigest(),
                "$or": [
                    {"date_expiry": None},
                    {"date_expiry": {"$gt": datetime.datetime.now(datetime.UTC)}},
                ],
            })
        except pymongo.errors.PyMongoError as e:
            raise DatabaseError(e) from e

    async def put_event(self, user, time, model, device, prompt_n, completion_n, request_id):
        common = {"date_created": time, "user_id": user.get("user_id"),
            "api_key_id": str(user.get("_id", "")), "request_id": str(request_id)}

        try:
            await self.db["billing"]["events_oneoff"].insert_one({
                **common,
                "product": "%s/%s/prompt" % (model, device),
                "quantity": prompt_n,
            })

            await self.db["billing"]["events_oneoff"].insert_one({
                **common,
                "product": "%s/%s/completion" % (model, device),
                "quantity": completion_n,
            })
        except pymongo.errors.PyMongoError as e:
            raise DatabaseError(e) from e


class SqliteDatabase:
    @classmethod
    async def create(cls, uri, logger):
        path = uri.removeprefix("sqlite://")
        assert path != uri

        self = cls()
        self.logger = logger
        self.db = await aiosqlite.connect(path)
        self.logger.debug("Connected to database")

        return self

    async def close(self):
        await self.db.close()
        self.logger.debug("Closed database connection")

    async def get_user(self, api_key):
        digest = hashlib.sha256(api_key.encode()).hexdigest()

        try:
            cur = await self.db.execute("""
                SELECT id FROM api_key WHERE type = 'LLM' AND secret = ?
                    AND (expires > datetime() OR expires IS NULL)
                """, (digest,))
        except sqlite3.DatabaseError as e:
            raise DatabaseError(e) from e

        row = await cur.fetchone()
        if row is not None:
            row = row[0]

        await cur.close()

        return row

    async def put_event(self, user, time, model, device, prompt_n, completion_n, request_id):
        product_prompt = "%s/%s/prompt" % (model, device)
        product_completion = "%s/%s/completion" % (model, device)

        try:
            await self.db.execute("""
                INSERT INTO event_oneoff (created, api_key, product, quantity, rid)
                VALUES (?, ?, ?, ?, ?)
                """, (time, user, product_prompt, prompt_n, str(request_id)))

            await self.db.execute("""
                INSERT INTO event_oneoff (created, api_key, product, quantity, rid)
                VALUES (?, ?, ?, ?, ?)
                """, (time, user, product_completion, completion_n, str(request_id)))
        except sqlite3.DatabaseError as e:
            raise DatabaseError(e) from e

        await self.db.commit()
