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
import prometheus_client

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


def _parse_metrics(text):
    """Parse Prometheus exposition format into {metric_name: [lines]}."""
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        metrics.setdefault(name, []).append(line)
    return metrics


class TestMetrics(LLMProxyAppTestCase):
    async def test_metrics_endpoint_exists(self):
        """The /metrics endpoint returns 200 and Prometheus content type."""
        async with self.client.request("GET", "/metrics") as res:
            self.assertEqual(res.status, 200)
            self.assertIn("text/plain", res.headers.get("Content-Type", ""))

    async def test_metrics_contain_request_counter(self):
        """After a request the metrics expose llmproxy_requests_total."""
        # Make a chat request first.
        body = {"model": "mymodel",
            "messages": [{"role": "user", "content": "hi"}]}
        async with self.client.request("POST", "/v1/chat/completions",
                headers={"Authorization": "Bearer mytoken"}, json=body) as res:
            self.assertEqual(res.status, 200)

        # Scrape metrics.
        async with self.client.request("GET", "/metrics") as res:
            self.assertEqual(res.status, 200)
            text = await res.text()

        parsed = _parse_metrics(text)
        self.assertIn("llmproxy_requests_total", parsed)

        # There should be a series for the POST /v1/chat/completions 200.
        found = any(
            'method="POST"' in line
            and 'path="/v1/chat/completions"' in line
            and 'status="200"' in line
            for line in parsed["llmproxy_requests_total"]
        )
        self.assertTrue(found,
            "llmproxy_requests_total missing POST /v1/chat/completions 200")

    async def test_metrics_contain_error_counter_on_401(self):
        """A 401 request is counted with status=401."""
        body = {"model": "mymodel",
            "messages": [{"role": "user", "content": "hi"}]}
        async with self.client.request("POST", "/v1/chat/completions",
                headers={"Authorization": "Bearer badtoken"}, json=body) as res:
            self.assertEqual(res.status, 401)

        async with self.client.request("GET", "/metrics") as res:
            text = await res.text()

        parsed = _parse_metrics(text)
        self.assertIn("llmproxy_requests_total", parsed)
        found = any(
            'status="401"' in line
            for line in parsed["llmproxy_requests_total"]
        )
        self.assertTrue(found, "llmproxy_requests_total missing status=401")

    async def test_metrics_contain_backend_counter(self):
        """Backend metrics are exposed after a successful request."""
        body = {"model": "mymodel",
            "messages": [{"role": "user", "content": "hi"}]}
        async with self.client.request("POST", "/v1/chat/completions",
                headers={"Authorization": "Bearer mytoken"}, json=body) as res:
            self.assertEqual(res.status, 200)

        async with self.client.request("GET", "/metrics") as res:
            text = await res.text()

        parsed = _parse_metrics(text)
        self.assertIn("llmproxy_backend_requests_total", parsed)
        found = any(
            'model="mymodel"' in line
            and 'status="200"' in line
            for line in parsed["llmproxy_backend_requests_total"]
        )
        self.assertTrue(found,
            "llmproxy_backend_requests_total missing mymodel 200")

    async def test_metrics_contain_token_counter(self):
        """Token metrics are exposed after a successful chat request."""
        body = {"model": "mymodel",
            "messages": [{"role": "user", "content": "hi"}]}
        async with self.client.request("POST", "/v1/chat/completions",
                headers={"Authorization": "Bearer mytoken"}, json=body) as res:
            self.assertEqual(res.status, 200)

        async with self.client.request("GET", "/metrics") as res:
            text = await res.text()

        parsed = _parse_metrics(text)
        self.assertIn("llmproxy_tokens_total", parsed)
        found_prompt = any(
            'model="mymodel"' in line
            and 'type="prompt"' in line
            for line in parsed["llmproxy_tokens_total"]
        )
        found_completion = any(
            'model="mymodel"' in line
            and 'type="completion"' in line
            for line in parsed["llmproxy_tokens_total"]
        )
        self.assertTrue(found_prompt, "missing prompt token metric")
        self.assertTrue(found_completion, "missing completion token metric")

    async def test_metrics_contain_duration_histogram(self):
        """Request duration histogram is exposed."""
        body = {"model": "mymodel",
            "messages": [{"role": "user", "content": "hi"}]}
        async with self.client.request("POST", "/v1/chat/completions",
                headers={"Authorization": "Bearer mytoken"}, json=body) as res:
            self.assertEqual(res.status, 200)

        async with self.client.request("GET", "/metrics") as res:
            text = await res.text()

        # Histograms produce _bucket, _sum, and _count series.
        self.assertIn("llmproxy_request_duration_seconds_bucket", text)
        self.assertIn("llmproxy_request_duration_seconds_count", text)
        self.assertIn("llmproxy_backend_duration_seconds_count", text)

    async def test_metrics_contain_active_requests_gauge(self):
        """The active requests gauge is present (value 0 when idle)."""
        async with self.client.request("GET", "/metrics") as res:
            text = await res.text()

        self.assertIn("llmproxy_active_requests", text)

    async def test_metrics_endpoint_itself_is_counted(self):
        """The /metrics scrape is also counted by the middleware."""
        async with self.client.request("GET", "/metrics") as res:
            self.assertEqual(res.status, 200)

        async with self.client.request("GET", "/metrics") as res:
            text = await res.text()

        parsed = _parse_metrics(text)
        found = any(
            'method="GET"' in line
            and 'path="/metrics"' in line
            for line in parsed.get("llmproxy_requests_total", [])
        )
        self.assertTrue(found, "metrics endpoint not counted")
