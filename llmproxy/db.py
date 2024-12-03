import asyncio
import datetime
import hashlib

import motor.motor_asyncio
import pymongo.server_api

async def get_db(req):
    if "db" not in req:
        uri = req.app["config"]["db"]["uri"]
        req["db"] = await Database.create(uri, req.app.logger)

    return req["db"]


class Database:
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
        return await self.db["cgc"]["api_keys"].find_one({
            "access_level": "LLM",
            "secret": hashlib.sha256(api_key.encode()).hexdigest(),
            "$or": [
                {"date_expiry": None},
                {"date_expiry": {"$gt": datetime.datetime.now(datetime.UTC)}},
            ],
        })

    async def put_event(self, user, time, model, device, prompt_n, completion_n, request_id):
        common = {"date_created": time, "user_id": user.get("user_id"),
            "api_key_id": str(user.get("_id", "")), "request_id": str(request_id)}

        prompt = self.db["billing"]["events_oneoff"].insert_one({
            **common,
            "product": "%s/%s/prompt" % (model, device),
            "quantity": prompt_n,
        })

        completion = self.db["billing"]["events_oneoff"].insert_one({
            **common,
            "product": "%s/%s/completion" % (model, device),
            "quantity": completion_n,
        })

        return await asyncio.gather(prompt, completion)
