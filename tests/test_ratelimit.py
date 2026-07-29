"""Unit tests for the in-memory rate limiter (resolver, rpm window,
concurrency, 429 body flavours). No aiohttp server — `slot` is driven directly
via a tiny fake request."""

import asyncio
import types
import unittest

import aiohttp.web

from llmproxy import ratelimit


class _FakeReq:
    """Stand-in for an aiohttp request: just the two attributes slot touches."""

    def __init__(self, path="/v1/chat/completions", config=None):
        self.rel_url = types.SimpleNamespace(path=path)
        self.app = {"config": config or {}}


class _Clock:
    """Controllable monotonic clock: slot/_reject read time.monotonic()."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


class TestResolve(unittest.TestCase):
    def setUp(self):
        ratelimit.flush()

    def test_no_config_returns_empty(self):
        self.assertEqual(ratelimit.resolve({"id": "u"}, "m", {}), {})

    def test_global_only(self):
        cfg = {"rate_limit": {"rpm": 60}}
        self.assertEqual(ratelimit.resolve({"id": "u"}, "m", cfg),
            {"rpm": (60, ("u",))})

    def test_per_model_overrides_global(self):
        cfg = {"rate_limit": {"rpm": 60},
            "backends": {"m": {"rate_limit": {"rpm": 30}}}}
        self.assertEqual(ratelimit.resolve({"id": "u"}, "m", cfg),
            {"rpm": (30, ("u", "m"))})

    def test_zero_means_unlimited(self):
        # per-model 0 exempts even when a global is set.
        cfg = {"rate_limit": {"rpm": 60},
            "backends": {"m": {"rate_limit": {"rpm": 0}}}}
        self.assertEqual(ratelimit.resolve({"id": "u"}, "m", cfg), {})
        # global 0 alone also exempts.
        self.assertEqual(ratelimit.resolve({"id": "u"}, "m",
            {"rate_limit": {"rpm": 0}}), {})

    def test_absent_falls_through(self):
        # per-model present only for rpm; concurrency falls through to global.
        cfg = {"rate_limit": {"concurrency": 5},
            "backends": {"m": {"rate_limit": {"rpm": 30}}}}
        self.assertEqual(ratelimit.resolve({"id": "u"}, "m", cfg),
            {"rpm": (30, ("u", "m")), "concurrency": (5, ("u",))})

    def test_user_none_uses_anon_bucket(self):
        cfg = {"rate_limit": {"rpm": 60}}
        self.assertEqual(ratelimit.resolve(None, "m", cfg),
            {"rpm": (60, (ratelimit._ANON,))})


class TestRpmWindow(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ratelimit.flush()
        self._clock = _Clock()
        self._orig = ratelimit.time
        ratelimit.time = types.SimpleNamespace(monotonic=self._clock.monotonic)

    def tearDown(self):
        ratelimit.time = self._orig
        ratelimit.flush()

    def _slot(self, config, user=None, path="/v1/chat/completions"):
        return ratelimit.slot(_FakeReq(path, config), user or {"id": "u"},
            "mymodel", {})

    async def test_rejects_over_limit_and_sets_retry_after(self):
        cfg = {"rate_limit": {"rpm": 2}}
        self._clock.t = 0
        async with self._slot(cfg):
            pass
        self._clock.t = 1
        async with self._slot(cfg):
            pass
        # Third within the window -> 429. oldest=0, now=2 -> retry=ceil(58)=58.
        self._clock.t = 2
        with self.assertRaises(aiohttp.web.HTTPTooManyRequests) as cm:
            async with self._slot(cfg):
                pass
        self.assertEqual(cm.exception.status, 429)
        self.assertEqual(cm.exception.headers["Retry-After"], "58")

    async def test_eviction_lets_window_refill(self):
        cfg = {"rate_limit": {"rpm": 2}}
        for t in (0, 1):
            self._clock.t = t
            async with self._slot(cfg):
                pass
        # After the window empties, a new request is admitted.
        self._clock.t = 61
        async with self._slot(cfg):
            pass  # should not raise
        # And another immediately is fine (deque was reset to 1 entry).
        self._clock.t = 61.5
        async with self._slot(cfg):
            pass


class TestConcurrency(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ratelimit.flush()

    def tearDown(self):
        ratelimit.flush()

    async def _slot(self, config, user=None, path="/v1/chat/completions"):
        return ratelimit.slot(_FakeReq(path, config), user or {"id": "u"},
            "mymodel", {})

    async def test_second_concurrent_rejected_no_retry_after(self):
        cfg = {"rate_limit": {"concurrency": 1}}
        async with await self._slot(cfg) as _:
            # Slot held: a second one on the same bucket is rejected.
            with self.assertRaises(aiohttp.web.HTTPTooManyRequests) as cm:
                async with await self._slot(cfg):
                    pass
            self.assertEqual(cm.exception.status, 429)
            self.assertNotIn("Retry-After", cm.exception.headers)

    async def test_release_on_exception(self):
        cfg = {"rate_limit": {"concurrency": 1}}
        with self.assertRaises(ValueError):
            async with await self._slot(cfg):
                raise ValueError("boom")
        # Counter released -> a fresh slot succeeds.
        async with await self._slot(cfg):
            pass

    async def test_release_on_normal_exit(self):
        cfg = {"rate_limit": {"concurrency": 1}}
        async with await self._slot(cfg):
            pass
        async with await self._slot(cfg):
            pass  # second sequential one must succeed


class TestRejectBody(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ratelimit.flush()

    def tearDown(self):
        ratelimit.flush()

    async def _reject(self, path, limit=1):
        cfg = {"rate_limit": {"rpm": limit}}
        req = _FakeReq(path, cfg)
        # Prime one entry then over-limit on the second.
        async with ratelimit.slot(req, {"id": "u"}, "mymodel", {}):
            pass
        with self.assertRaises(aiohttp.web.HTTPTooManyRequests) as cm:
            async with ratelimit.slot(req, {"id": "u"}, "mymodel", {}):
                pass
        return cm.exception

    async def test_openai_flavour(self):
        exc = await self._reject("/v1/chat/completions")
        import json
        body = json.loads(exc.text)
        self.assertEqual(body["error"]["type"], "rate_limit_exceeded")
        self.assertEqual(body["error"]["code"], "rate_limit_exceeded")
        self.assertIn("mymodel", body["error"]["message"])

    async def test_anthropic_flavour(self):
        exc = await self._reject("/v1/messages")
        import json
        body = json.loads(exc.text)
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "rate_limit_error")
        self.assertIn("mymodel", body["error"]["message"])


if __name__ == "__main__":
    unittest.main()
