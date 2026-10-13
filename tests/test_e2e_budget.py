"""Opt-in real proxy -> vLLM -> SQLite billing tests.

Set BUDGET_VLLM_URL, BUDGET_VLLM_TOKEN and BUDGET_VLLM_MODEL.
Requests generate at most 256 tokens each; never modify the live backend.
Native reasoning control is tested only on Chat Completions. vLLM 0.26.0
does not wire thinking_token_budget into Responses/Messages sampling.
"""

import asyncio
import hashlib
import json
import os
import tempfile
import time
import unittest
import warnings

import aiohttp.test_utils
import aiohttp.web

from llmproxy import auth, budget, streaming
from llmproxy.app import create_app
from llmproxy.db import get_db
from tests.test_budget import AUTH, BillingTestServer


URL = os.environ.get("BUDGET_VLLM_URL")
TOKEN = os.environ.get("BUDGET_VLLM_TOKEN")
MODEL = os.environ.get("BUDGET_VLLM_MODEL")
PROMPT = "What is 17 * 19? Answer with the number only."


@unittest.skipUnless(URL and TOKEN and MODEL,
    "e2e: set BUDGET_VLLM_URL, BUDGET_VLLM_TOKEN and BUDGET_VLLM_MODEL")
class TestLiveBudget(aiohttp.test_utils.AioHTTPTestCase):
    async def asyncSetUp(self):
        warnings.simplefilter("ignore", aiohttp.web.NotAppKeyWarning)
        auth.flush_cache()
        await super().asyncSetUp()

    async def get_server(self, app):
        return BillingTestServer(app)

    async def get_application(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        app = await create_app({"timeout_connect": 15, "timeout_read": 90,
            "db": {"uri": "sqlite://%s" % self.db_path},
            "budget": {"enabled": True}, "provenance": {"enabled": False},
            "backends": {"live-qwen": {"url": URL, "token": TOKEN, "model": MODEL,
                "device": "a100", "type": "chat", "max_model_len": 65536,
                "output_budget": {"mode": "enforce", "default_max_tokens": 256,
                    "ceiling": 256, "apply_default": True, "clamp": True},
                "reasoning": {"native_budget": True, "reserve_content_tokens": 224,
                    "effort_budgets": {"low": 16}}}}})
        db = await get_db(app["config"]["db"]["uri"])
        await db.db.execute("INSERT INTO api_key (id,secret,type) VALUES (?,?,?)",
            ("budget-live-user", hashlib.sha256(b"budget-test-token").hexdigest(), "LLM"))
        await db.db.commit()
        await db.close()
        return app

    async def asyncTearDown(self):
        await super().asyncTearDown()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    async def events(self, count):
        deadline = time.monotonic() + 5
        while True:
            db = await get_db(self.app["config"]["db"]["uri"])
            cur = await db.db.execute("SELECT product,quantity,rid FROM event_oneoff")
            rows = await cur.fetchall()
            await db.close()
            if len(rows) >= count or time.monotonic() >= deadline:
                return rows
            await asyncio.sleep(.01)

    async def test_real_backend_four_protocols_and_streaming(self):
        expected_rows = 0
        for path, protocol in budget.PATHS.items():
            for is_stream in (False, True):
                with self.subTest(path=path, stream=is_stream):
                    payload = {"model": "live-qwen", "stream": is_stream, "temperature": 0}
                    key = {"chat": "max_completion_tokens", "completions": "max_tokens",
                        "responses": "max_output_tokens", "messages": "max_tokens"}[protocol]
                    payload[key] = 999999  # clamp -> 256, not a context-sized allowance
                    if protocol == "completions":
                        payload["prompt"] = "17 * 19 ="
                    elif protocol == "responses":
                        payload["input"] = PROMPT
                    else:
                        payload["messages"] = [{"role": "user", "content": PROMPT}]
                    if protocol == "chat":
                        payload["reasoning_effort"] = "low"
                    async with self.client.post(path, headers=AUTH, json=payload) as res:
                        self.assertEqual(res.status, 200, (await res.text())[:500] if res.status != 200 else "")
                        self.assertEqual(res.headers["X-LLMProxy-Max-Output-Tokens"], "256")
                        self.assertEqual(res.headers["X-LLMProxy-Budget-Clamped"], "true")
                        rid = res.headers["X-Request-ID"]
                        raw = await res.read()
                        if is_stream:
                            self.assertEqual(res.headers["X-LLMProxy-Truncation"], "pending")
                            self.assertNotIn(b'"llmproxy"', raw)
                            self.assertNotIn(b"X-LLMProxy", raw)
                            events = [streaming.parse_sse_event(block + b"\n\n")
                                for block in raw.split(b"\n\n") if block]
                            if protocol == "chat":
                                text = "".join((choice.get("delta") or {}).get("content") or ""
                                    for event in events if isinstance(event, dict)
                                    for choice in event.get("choices", []))
                                self.assertIn("323", text)
                        else:
                            data = json.loads(raw)
                            meta = data["llmproxy"]
                            self.assertIsNone(meta["budget"]["effective_max_output_tokens"])
                            if protocol == "chat":
                                self.assertEqual(meta["budget"]["reasoning"]["forwarded_tokens"], 16)
                                self.assertIn("323", data["choices"][0]["message"]["content"])
                                self.assertEqual(data["choices"][0]["finish_reason"], "stop")
                            else:
                                self.assertNotIn("reasoning", meta["budget"])
                        expected_rows += 2
                        rows = await self.events(expected_rows)
                        current = [row for row in rows if row["rid"] == rid]
                        self.assertEqual(len(current), 2)
                        completion = next(row for row in current if row["product"].endswith("/completion"))
                        self.assertLessEqual(completion["quantity"], 256)
                        self.assertTrue(all(row["product"].startswith("live-qwen/a100/") for row in current))
                        print("live-budget:", protocol, "stream=" + str(is_stream),
                            "completion_tokens=" + str(completion["quantity"]))

    async def test_native_budget_changes_reasoning_only_failure_into_content(self):
        cfg = self.app["config"]["backends"]["live-qwen"]
        payload = {"model": "live-qwen", "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0, "seed": 42}
        cfg["reasoning"]["native_budget"] = False
        async with self.client.post("/v1/chat/completions", headers=AUTH, json=payload) as res:
            self.assertEqual(res.status, 200)
            baseline = await res.json()
        cfg["reasoning"]["native_budget"] = True
        payload["reasoning_effort"] = "low"
        async with self.client.post("/v1/chat/completions", headers=AUTH, json=payload) as res:
            self.assertEqual(res.status, 200)
            limited = await res.json()
        before = baseline["choices"][0]
        after = limited["choices"][0]
        self.assertEqual(before["finish_reason"], "length")
        self.assertFalse(before["message"].get("content"))
        self.assertEqual(after["finish_reason"], "stop")
        self.assertIn("323", after["message"]["content"])
        self.assertEqual(limited["llmproxy"]["budget"]["reasoning"]["forwarded_tokens"], 16)
        self.assertLess(limited["usage"]["completion_tokens"], baseline["usage"]["completion_tokens"])
        self.assertEqual(len(await self.events(4)), 4)
        print("live-budget: baseline reasoning-only length=", baseline["usage"]["completion_tokens"],
            "; native budget produced final content in", limited["usage"]["completion_tokens"], "tokens")
