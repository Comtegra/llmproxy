import datetime
import hashlib

import motor.motor_asyncio
import pymongo.server_api


class Database:
    def __init__(self, cfg):
        self.db = motor.motor_asyncio.AsyncIOMotorClient(cfg["uri"],
            tz_aware=True, connect=True,
            server_api=pymongo.server_api.ServerApi('1'))

    def check(self):
        return self.db.admin.command("ping")

    def get_user(self, api_key):
        return self.db["cgc"]["api_keys"].find_one({
            "access_level": "COMPLETION",
            "secret": hashlib.sha256(api_key.encode()).hexdigest(),
            "$or": [
                {"date_expiry": None},
                {"date_expiry": {"$gt": datetime.datetime.now(datetime.UTC)}},
            ],
        })

    def put_event(self, user, time, model, prompt_n, completion_n):
        return self.db["billing"]["events_completion"].insert_one({
            "date_created": time,
            "user_id": user.get("user_id"),
            "api_key_id": str(user.get("_id", "")),
            "model": model,
            "prompt_n": prompt_n,
            "completion_n": completion_n,
        })
