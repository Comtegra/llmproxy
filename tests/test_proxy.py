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
                    "token": "mybackendtoken", "device": "none",
                    "model": "test-org/mymodel", "type": "chat",
                    "quantization": "test-quant"},
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

    async def test_4xx_forwarded(self):
        body = {"model": "mymodel", "_trigger_error": 400,
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 400)
            data = await res.json()
            self.assertEqual(data["error"]["message"], "Input too long")

        self.assertListEqual(await self.get_events(), [])

    async def test_422_forwarded(self):
        body = {"model": "mymodel", "_trigger_error": 422,
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 422)
            data = await res.json()
            self.assertEqual(data["error"]["message"], "Context length exceeded")

        self.assertListEqual(await self.get_events(), [])

    async def test_unexpected_2xx_masked(self):
        body = {"model": "mymodel", "_trigger_error": 201,
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 502)

        self.assertListEqual(await self.get_events(), [])

    async def test_5xx_masked(self):
        body = {"model": "mymodel", "_trigger_error": 500,
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 502)

        self.assertListEqual(await self.get_events(), [])


class TestModels(LLMProxyAppTestCase):
    async def test_list_uses_discovery(self):
        req = self.client.request("GET", "/v1/models",
            headers={"Authorization": "Bearer mytoken"})
        async with req as res:
            self.assertEqual(res.status, 200)
            data = await res.json()

        self.assertEqual(data["object"], "list")
        self.assertEqual(len(data["data"]), 1)
        m = data["data"][0]
        self.assertEqual(m["id"], "mymodel")
        self.assertEqual(m["object"], "model")
        self.assertIsNone(m["owned_by"])
        self.assertEqual(m["type"], "chat")
        self.assertEqual(m["source_model"], "test-org/mymodel")
        self.assertEqual(m["card_url"],
            "https://huggingface.co/test-org/mymodel")
        self.assertEqual(m["quantization"], "test-quant")
        self.assertEqual(m["context_length"], 4096)
        self.assertNotIn("device", m)

    async def test_unauthorized(self):
        req = self.client.request("GET", "/v1/models",
            headers={"Authorization": "Bearer badtoken"})
        async with req as res:
            self.assertEqual(res.status, 401)


class TestModelsTOMLOverride(LLMProxyAppTestCase):
    async def get_application(self):
        app = await super().get_application()
        app["config"]["backends"]["mymodel"]["context_length"] = 2048
        return app

    async def test_toml_wins(self):
        req = self.client.request("GET", "/v1/models",
            headers={"Authorization": "Bearer mytoken"})
        async with req as res:
            data = await res.json()
        self.assertEqual(data["data"][0]["context_length"], 2048)


class TestModelsNoQuantization(LLMProxyAppTestCase):
    async def get_application(self):
        app = await super().get_application()
        del app["config"]["backends"]["mymodel"]["quantization"]
        return app

    async def test_field_absent(self):
        req = self.client.request("GET", "/v1/models",
            headers={"Authorization": "Bearer mytoken"})
        async with req as res:
            data = await res.json()
        self.assertNotIn("quantization", data["data"][0])


class TestModelsModelUrlOverride(LLMProxyAppTestCase):
    async def get_application(self):
        app = await super().get_application()
        app["config"]["backends"]["mymodel"]["model_url"] = \
            "https://example.com/custom-card"
        return app

    async def test_model_url_used(self):
        req = self.client.request("GET", "/v1/models",
            headers={"Authorization": "Bearer mytoken"})
        async with req as res:
            data = await res.json()
        self.assertEqual(data["data"][0]["card_url"],
            "https://example.com/custom-card")


class TestModelsDiscoveryFailure(LLMProxyAppTestCase):
    async def get_application(self):
        self.backend.app["v1_models_error"] = 500

        self.db_fd, self.db_path = tempfile.mkstemp()

        app = await create_app({
            "timeout_connect": 1,
            "timeout_read": 1,
            "db": {"uri": "sqlite://%s" % self.db_path},
            "backends": {"mymodel":
                {"url": "http://%s:%d" % (self.backend.host, self.backend.port),
                    "token": "mybackendtoken", "device": "none",
                    "model": "test-org/mymodel", "type": "chat"},
            },
        })

        secret = hashlib.sha256("mytoken".encode()).hexdigest()
        db = await get_db(app["config"]["db"]["uri"])
        await db.db.execute("""
            INSERT INTO api_key (id, secret, type) VALUES ('myuser', ?, 'LLM')
            """, (secret,))
        await db.db.commit()
        await db.close()

        return app

    async def test_probe_failure_graceful(self):
        req = self.client.request("GET", "/v1/models",
            headers={"Authorization": "Bearer mytoken"})
        async with req as res:
            self.assertEqual(res.status, 200)
            data = await res.json()

        self.assertEqual(len(data["data"]), 1)
        m = data["data"][0]
        self.assertEqual(m["id"], "mymodel")
        self.assertNotIn("context_length", m)


class TestModelsAudio(LLMProxyAppTestCase):
    async def get_application(self):
        self.db_fd, self.db_path = tempfile.mkstemp()

        app = await create_app({
            "timeout_connect": 1,
            "timeout_read": 1,
            "db": {"uri": "sqlite://%s" % self.db_path},
            "backends": {"mymodel":
                {"url": "http://%s:%d" % (self.backend.host, self.backend.port),
                    "token": "mybackendtoken", "device": "none",
                    "model": "test-org/mymodel", "type": "audio"},
            },
        })

        secret = hashlib.sha256("mytoken".encode()).hexdigest()
        db = await get_db(app["config"]["db"]["uri"])
        await db.db.execute("""
            INSERT INTO api_key (id, secret, type) VALUES ('myuser', ?, 'LLM')
            """, (secret,))
        await db.db.commit()
        await db.close()

        return app

    async def test_probe_skipped(self):
        self.assertEqual(self.backend.app["hits"]["v1_models"], 0)

    async def test_no_context_length_in_response(self):
        req = self.client.request("GET", "/v1/models",
            headers={"Authorization": "Bearer mytoken"})
        async with req as res:
            data = await res.json()
        self.assertNotIn("context_length", data["data"][0])
        self.assertEqual(data["data"][0]["type"], "audio")
