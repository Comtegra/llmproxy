import hashlib
import importlib
import importlib.resources
import os
import sqlite3
import tempfile
import unittest
import warnings

import aiohttp
import aiohttp.test_utils

from llmproxy import config
from llmproxy.app import create_app, reload_config
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
            "max_json_body": 1024 * 1024,
            "db": {"uri": "sqlite://%s" % self.db_path},
            "backends": {
                "mymodel": {
                    "url": "http://%s:%d" % (
                        self.backend.host, self.backend.port),
                    "token": "mybackendtoken",
                    "device": "none",
                    "max_model_len": 12345,
                },
                "nolimit": {
                    "url": "http://%s:%d" % (
                        self.backend.host, self.backend.port),
                    "token": "secret-backend-token",
                    "device": "none",
                    "model": "mymodel",
                    "verify_ssl": False,
                },
                "slowok": {
                    "url": "http://%s:%d" % (
                        self.backend.host, self.backend.port),
                    "token": "mybackendtoken",
                    "device": "none",
                    "model": "mymodel",
                    "timeout": 5,
                },
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
    async def test_models_include_public_metadata_only(self):
        req = self.client.request("GET", "/v1/models",
            headers={"Authorization": "Bearer mytoken"})

        async with req as res:
            self.assertEqual(res.status, 200)
            data = await res.json()

        models = {m["id"]: m for m in data["data"]}
        self.assertEqual(models["mymodel"]["max_model_len"], 12345)
        self.assertNotIn("max_model_len", models["nolimit"])
        for model in models.values():
            self.assertLessEqual(
                set(model),
                {"id", "object", "created", "owned_by", "device",
                    "max_model_len"},
            )

    async def test_simple(self):
        body = {"model": "mymodel", "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)
            self.assertIn("X-Request-ID", res.headers)
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
            self.assertIn("X-Request-ID", res.headers)
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
            self.assertIn("X-Request-ID", res.headers)

        self.assertListEqual(await self.get_events(), [])

    async def test_context_length_5xx_mapped_to_422(self):
        body = {"model": "mymodel", "_trigger_error": "context500",
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 422)
            self.assertIn("X-Request-ID", res.headers)
            data = await res.json()
            self.assertEqual(data["error"]["code"], "context_length_exceeded")

        self.assertListEqual(await self.get_events(), [])

    async def test_slow_backend_returns_504(self):
        body = {"model": "mymodel", "_trigger_error": "slow",
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 504)
            self.assertIn("X-Request-ID", res.headers)

        self.assertListEqual(await self.get_events(), [])

    async def test_streaming_response_includes_request_id(self):
        body = {"model": "mymodel", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)
            self.assertIn("X-Request-ID", res.headers)
            body = await res.text()
            self.assertIn("data: [DONE]", body)

        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 1},
            {"product": "mymodel/none/completion", "quantity": 2},
        ])

    async def test_embeddings_billing(self):
        body = {"model": "mymodel", "input": "hello"}
        req = self.client.request("POST", "/v1/embeddings",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)
            data = await res.json()

        self.assertEqual(data["usage"]["prompt_tokens"], 7)
        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/embedding", "quantity": 7},
        ])

    async def test_audio_transcription_billing(self):
        form = aiohttp.FormData()
        form.add_field("model", "mymodel")
        form.add_field("file", b"RIFFfake-audio", filename="a.wav",
            content_type="audio/wav")
        req = self.client.request("POST", "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer mytoken"}, data=form)

        async with req as res:
            self.assertEqual(res.status, 200)
            data = await res.json()

        # Billed per second of audio; fractional durations must survive.
        self.assertEqual(data["duration"], 12.5)
        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/transcription", "quantity": 12.5},
        ])

    async def test_streaming_usage_in_trailing_chunk(self):
        # Realistic vLLM: usage arrives in a separate trailing chunk, not the
        # content chunk. The proxy must keep the last non-[DONE] chunk.
        body = {"model": "mymodel", "stream": True,
            "_stream_mode": "split_usage",
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)
            text = await res.text()
            self.assertIn("data: [DONE]", text)

        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 1},
            {"product": "mymodel/none/completion", "quantity": 2},
        ])

    async def test_per_backend_timeout_allows_slow_backend(self):
        # Global sock_read=1 would 504 a 2s backend; a backend with a long
        # per-backend timeout waits and still bills.
        body = {"model": "slowok", "_trigger_error": "slow",
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)

        self.assertListEqual(await self.get_events(), [
            {"product": "slowok/none/prompt", "quantity": 1},
            {"product": "slowok/none/completion", "quantity": 2},
        ])

    async def test_large_audio_upload_accepted(self):
        # A >1 MiB upload must not be rejected: aiohttp's default client_max_size
        # is 1 MiB, and the app raises it so audio works.
        big = b"\x00" * (2 * 1024 * 1024)
        form = aiohttp.FormData()
        form.add_field("model", "mymodel")
        form.add_field("file", big, filename="a.wav",
            content_type="audio/wav")
        req = self.client.request("POST", "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer mytoken"}, data=form)

        async with req as res:
            self.assertEqual(res.status, 200)

        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/transcription", "quantity": 12.5},
        ])

    async def test_unbillable_duration_fails_loud(self):
        # Missing, zero, negative, NaN and Infinity durations are all
        # unbillable -> 502 + no billing (never a corrupt or zero billing row).
        cases = [("_omit_duration", "1"), ("_duration", "zero"),
            ("_duration", "neg"), ("_duration", "nan"), ("_duration", "inf"),
            ("_duration", "true")]
        for field, value in cases:
            with self.subTest(case=value):
                form = aiohttp.FormData()
                form.add_field("model", "mymodel")
                form.add_field(field, value)
                form.add_field("file", b"RIFFfake", filename="a.wav",
                    content_type="audio/wav")
                req = self.client.request("POST", "/v1/audio/transcriptions",
                    headers={"Authorization": "Bearer mytoken"}, data=form)

                async with req as res:
                    self.assertEqual(res.status, 502)

                self.assertListEqual(await self.get_events(), [])

    async def test_oversized_json_body_rejected(self):
        # A JSON body over max_json_body is rejected with 413 before it is
        # buffered, protecting the text endpoints from a memory-exhaustion DoS.
        big = {"model": "mymodel",
            "messages": [{"role": "user", "content": "x" * (2 * 1024 * 1024)}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=big)

        async with req as res:
            self.assertEqual(res.status, 413)

        self.assertListEqual(await self.get_events(), [])

    async def test_oversized_chunked_body_rejected(self):
        # The cap must also hold for a chunked body (no Content-Length), which
        # would slip past a Content-Length-only check.
        async def gen():
            for _ in range(2 * 1024):  # ~2 MiB in 1 KiB chunks
                yield b"x" * 1024

        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken",
                "Content-Type": "application/json"}, data=gen())

        async with req as res:
            self.assertEqual(res.status, 413)

        self.assertListEqual(await self.get_events(), [])


class TestConfigValidation(unittest.IsolatedAsyncioTestCase):
    def test_validate_accepts_positive_integer_max_model_len(self):
        config.validate({"backends": {"mymodel": {"max_model_len": 131072}}})

    def test_validate_rejects_invalid_max_model_len(self):
        for value in (0, -1, "131072", True):
            with self.subTest(value=value):
                with self.assertRaises(config.ConfigError):
                    config.validate({
                        "backends": {"mymodel": {"max_model_len": value}},
                    })

    def test_validate_rejects_invalid_body_limits(self):
        for key in ("client_max_size", "max_json_body"):
            for value in (0, -1, "100", True):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(config.ConfigError):
                        config.validate({key: value})

    def test_validate_rejects_invalid_backend_timeout(self):
        for value in (0, -1, "5", True):
            with self.subTest(value=value):
                with self.assertRaises(config.ConfigError):
                    config.validate({"backends": {"m": {"timeout": value}}})

    def test_validate_accepts_valid_timeout_and_client_max_size(self):
        config.validate({
            "client_max_size": 2147483648,
            "backends": {"m": {"timeout": 1800}, "n": {"timeout": 0.5}},
        })

    def test_load_invalid_toml_raises_config_error(self):
        fd, path = tempfile.mkstemp(suffix=".toml")
        with os.fdopen(fd, "w") as f:
            f.write("[backends\n")
        try:
            with self.assertRaises(config.ConfigError):
                config.load(path)
        finally:
            os.unlink(path)

    async def test_create_app_validates_in_memory_config(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            with self.assertRaises(config.ConfigError):
                await create_app({
                    "timeout_connect": 1,
                    "timeout_read": 1,
                    "db": {"uri": "sqlite://%s" % path},
                    "backends": {"mymodel": {"max_model_len": 0}},
                })
        finally:
            os.unlink(path)

    def test_reload_config_keeps_previous_backends_on_invalid_config(self):
        app = aiohttp.web.Application()
        app["config"] = {"_path": "dummy.toml", "backends": {"old": {}}}

        old_load = config.load

        def load_invalid(path):
            raise config.ConfigError("bad config")

        config.load = load_invalid
        try:
            reload_config(app)
        finally:
            config.load = old_load

        self.assertEqual(app["config"]["backends"], {"old": {}})
