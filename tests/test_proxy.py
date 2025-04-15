import hashlib
import importlib
import importlib.resources
import os
import sqlite3
import tempfile
import warnings

import aiohttp
import aiohttp.test_utils

from llmproxy.app import create_app
from llmproxy.db import get_db

from . import mockbackend


class LLMProxyAppTestCase(aiohttp.test_utils.AioHTTPTestCase):
    async def asyncSetUp(self):
        # Don't care about type checkers
        warnings.simplefilter("ignore", category=aiohttp.web.NotAppKeyWarning)

        self.backend = aiohttp.test_utils.TestServer(mockbackend.create_app())
        await self.backend.start_server()

        await super().asyncSetUp()

    async def asyncTearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)
        await super().asyncTearDown()

    async def get_application(self):
        self.db_fd, self.db_path = tempfile.mkstemp()

        app = await create_app({
            "timeout_connect": 1,
            "timeout_read": 1,
            "db": {"uri": "sqlite://%s" % self.db_path},
            "backends": {"mymodel":
                {"url": "http://%s:%d" % (self.backend.host, self.backend.port),
                    "token": "mybackendtoken", "device": "none"},
            },
        })

        # Insert test user
        secret = hashlib.sha256("mytoken".encode()).hexdigest()
        db = await get_db(app["config"]["db"]["uri"])
        await db.db.execute("""
            INSERT INTO api_key (id, secret, type) VALUES ('myuser', ?, 'LLM')
            """, (secret,))
        await db.db.commit()
        await db.close()

        return app

    async def get_events(self):
        db = await get_db(self.app["config"]["db"]["uri"])
        cur = await db.db.execute("SELECT product, quantity FROM event_oneoff")
        rows = await cur.fetchall()
        await db.close()

        return rows


class TestChat(LLMProxyAppTestCase):
    async def test_simple(self):
        body = {"model": "mymodel", "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)
            data = await res.json()

        self.assertEqual(data["choices"][0]["message"]["content"], "you said: hi")

        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 1},
            {"product": "mymodel/none/completion", "quantity": 2},
        ])

    async def test_unknown_token(self):
        body = {"model": "mymodel", "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer badtoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 401)

        self.assertListEqual(await self.get_events(), [])

    async def test_blank_token(self):
        body = {"model": "mymodel", "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer "}, json=body)

        async with req as res:
            self.assertEqual(res.status, 401)

        self.assertListEqual(await self.get_events(), [])
