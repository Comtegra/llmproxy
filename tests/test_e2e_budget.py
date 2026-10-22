"""Opt-in real proxy -> vLLM -> SQLite billing tests.

Set BUDGET_VLLM_URL, BUDGET_VLLM_TOKEN and BUDGET_VLLM_MODEL.
Requests generate at most 256 tokens each; never modify the live backend.
Native reasoning control is tested only on Chat Completions. vLLM does
not wire thinking_token_budget into Responses/Messages sampling.

BUDGET_VLLM_FIXTURE selects the live backend profile (default "qwen"):

* "qwen" -- the original reasoning-enabled Qwen profile: the historical
  target whose unbounded default thinking burns a whole 256-token
  budget on the math prompt.
* "glm53" -- a GLM-5.3-style profile with a small reserve/effort
  calibration for bounded live runs; set BUDGET_VLLM_MODEL=glm-53-nvfp4
  (the served name). Adjust the fixture fields to match the target
  backend before running live.

Usage-equality on reasoning tokens is asserted ONLY when the limit binds
(below the prompt's natural thinking); otherwise only <= / shape checks.

Live-host note: this in-process live e2e needs direct backend
reachability via BUDGET_VLLM_URL. Where the backend's only public
surface is an auth-gated proxy, run the deployed-proxy live-check
matrices against that surface instead; the unit fixtures keep covering
byte-identity here. The env-gating below is unchanged: with no
BUDGET_VLLM_* env the suite skips explicitly and stays part of the
offline run.
"""

import asyncio
import hashlib
import json
import os
import tempfile
import time
import unittest
import warnings
from unittest import mock

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
PRIMES_PROMPT = ("Sum all prime numbers below 100. Verify by listing each "
    "prime and the running total. Answer with only the final number.")


class BudgetFixture:
    """One live backend profile: config values, prompts and calibrated caps.

    ceiling == default_max_tokens == 256 keeps the "<= 256 generated
    tokens" cost guarantee for every request this suite makes.
    """

    name = None            # backend/model name routed through the proxy
    device = None          # billing product device segment
    max_model_len = None
    default_max_tokens = None
    ceiling = None
    reserve = None
    effort_budgets = {}
    math_prompt = PROMPT
    primes_prompt = None   # small-cap scenario prompt (bounded-reasoning check)
    small_cap = None       # output cap for the primes prompt scenario

    def backends(self, url, token, model):
        return {self.name: {"url": url, "token": token, "model": model,
            "device": self.device, "type": "chat",
            "max_model_len": self.max_model_len,
            "output_budget": {"mode": "enforce",
                "default_max_tokens": self.default_max_tokens,
                "ceiling": self.ceiling, "apply_default": True,
                "clamp": True},
            "reasoning": {"native_budget": True,
                "reserve_content_tokens": self.reserve,
                "effort_budgets": dict(self.effort_budgets)}}}


class QwenFixture(BudgetFixture):
    """Exactly the historical live target (regression anchor): every value
    matches the pre-fixture config byte for byte."""

    name = "live-qwen"
    device = "a100"
    max_model_len = 65536
    default_max_tokens = 256
    ceiling = 256
    reserve = 224
    effort_budgets = {"low": 16}


class GLM53Fixture(BudgetFixture):
    """A GLM-5.3-style profile with a small reserve/effort calibration:
    reserve 64 and effort low=64 make the injected thinking_token_budget
    bind in the small-cap scenario (min(64, 128-64) == 64), and small_cap
    128 is that scenario's output cap. All values are the profile's
    calibration, not universal constants: adjust them to match the
    target backend before running live."""

    name = "live-glm53"
    device = "b200"
    max_model_len = 497728
    default_max_tokens = 256
    ceiling = 256
    reserve = 64
    effort_budgets = {"low": 64}
    math_prompt = "What is 17 * 19? Answer with the number only."
    primes_prompt = PRIMES_PROMPT
    small_cap = 128


FIXTURES = {"qwen": QwenFixture, "glm53": GLM53Fixture}


def select_fixture(name=None):
    """Resolve the live fixture (env BUDGET_VLLM_FIXTURE, default "qwen").
    Unknown names fail loudly instead of silently running the wrong suite."""
    if name is None:
        name = os.environ.get("BUDGET_VLLM_FIXTURE") or "qwen"
    try:
        return FIXTURES[name.lower()]()
    except KeyError:
        raise ValueError("unknown BUDGET_VLLM_FIXTURE %r; choose one of: %s"
            % (name, ", ".join(sorted(FIXTURES)))) from None


FIXTURE = select_fixture()


class TestBudgetFixtureConfig(unittest.TestCase):
    """Offline regression on the fixture wiring itself: the default (qwen)
    profile must keep producing the exact legacy live config, and the glm53
    profile must mirror the plan's calibrated values."""

    def test_qwen_fixture_is_the_legacy_config(self):
        fixture = select_fixture("qwen")
        self.assertEqual(fixture.backends("https://vllm.example", "token", "m"),
            {"live-qwen": {"url": "https://vllm.example", "token": "token",
                "model": "m", "device": "a100", "type": "chat",
                "max_model_len": 65536,
                "output_budget": {"mode": "enforce",
                    "default_max_tokens": 256, "ceiling": 256,
                    "apply_default": True, "clamp": True},
                "reasoning": {"native_budget": True,
                    "reserve_content_tokens": 224,
                    "effort_budgets": {"low": 16}}}})

    def test_default_fixture_is_qwen(self):
        env = dict(os.environ)
        env.pop("BUDGET_VLLM_FIXTURE", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsInstance(select_fixture(), QwenFixture)

    def test_unknown_fixture_fails_loud(self):
        with self.assertRaises(ValueError):
            select_fixture("nope")

    def test_glm53_fixture_values(self):
        fixture = select_fixture("glm53")
        self.assertEqual(fixture.name, "live-glm53")
        self.assertEqual(fixture.device, "b200")
        self.assertEqual(fixture.max_model_len, 497728)
        self.assertEqual(fixture.default_max_tokens, 256)
        self.assertEqual(fixture.ceiling, 256)
        self.assertEqual(fixture.reserve, 64)
        self.assertEqual(fixture.effort_budgets, {"low": 64})
        self.assertEqual(fixture.math_prompt,
            "What is 17 * 19? Answer with the number only.")
        self.assertEqual(fixture.primes_prompt,
            "Sum all prime numbers below 100. Verify by listing each prime "
            "and the running total. Answer with only the final number.")
        self.assertEqual(fixture.small_cap, 128)

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
            "backends": FIXTURE.backends(URL, TOKEN, MODEL)})
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
        """All four protocols x json/SSE with the selected fixture: a
        999999 limit clamps to the 256 ceiling, exact echo headers, chat
        effort low forwards the fixture's low budget (proxy-side exact
        assert), the math answer completes, and every rid bills exactly
        twice. Reasoning usage equality is never asserted here: the low
        effort limit does not bind GLM-5.3's natural 17*19 thinking."""
        expected_rows = 0
        for path, protocol in budget.PATHS.items():
            for is_stream in (False, True):
                with self.subTest(path=path, stream=is_stream):
                    payload = {"model": FIXTURE.name, "stream": is_stream, "temperature": 0}
                    key = {"chat": "max_completion_tokens", "completions": "max_tokens",
                        "responses": "max_output_tokens", "messages": "max_tokens"}[protocol]
                    payload[key] = 999999  # clamp -> 256, not a context-sized allowance
                    if protocol == "completions":
                        payload["prompt"] = "17 * 19 ="
                    elif protocol == "responses":
                        payload["input"] = FIXTURE.math_prompt
                    else:
                        payload["messages"] = [{"role": "user", "content": FIXTURE.math_prompt}]
                    if protocol == "chat":
                        payload["reasoning_effort"] = "low"
                    async with self.client.post(path, headers=AUTH, json=payload) as res:
                        self.assertEqual(res.status, 200, (await res.text())[:500] if res.status != 200 else "")
                        self.assertEqual(res.headers["X-LLMProxy-Extensions"], "budget.v1")
                        self.assertEqual(res.headers["X-LLMProxy-Max-Output-Tokens"], "256")
                        self.assertEqual(res.headers["X-LLMProxy-Budget-Source"], "client")
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
                                self.assertEqual(meta["budget"]["reasoning"]["forwarded_tokens"],
                                    FIXTURE.effort_budgets["low"])
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
                        self.assertTrue(all(row["product"].startswith(
                            "%s/%s/" % (FIXTURE.name, FIXTURE.device)) for row in current))
                        print("live-budget:", protocol, "stream=" + str(is_stream),
                            "completion_tokens=" + str(completion["quantity"]))

    @unittest.skipUnless(isinstance(FIXTURE, QwenFixture),
        "qwen fixture semantics: unbounded default thinking burns the whole "
        "default budget on the math prompt; the GLM-5.3 analog is the "
        "glm53 incident test (its natural math thinking does not burn "
        "the whole budget)")
    async def test_native_budget_changes_reasoning_only_failure_into_content(self):
        cfg = self.app["config"]["backends"][FIXTURE.name]
        payload = {"model": FIXTURE.name, "messages": [{"role": "user", "content": FIXTURE.math_prompt}],
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
        self.assertEqual(limited["llmproxy"]["budget"]["reasoning"]["forwarded_tokens"],
            FIXTURE.effort_budgets["low"])
        self.assertLess(limited["usage"]["completion_tokens"], baseline["usage"]["completion_tokens"])
        self.assertEqual(len(await self.events(4)), 4)
        print("live-budget: baseline reasoning-only length=", baseline["usage"]["completion_tokens"],
            "; native budget produced final content in", limited["usage"]["completion_tokens"], "tokens")

    @unittest.skipUnless(isinstance(FIXTURE, GLM53Fixture) and MODEL == "glm-53-nvfp4",
        "glm53 fixture: set BUDGET_VLLM_FIXTURE=glm53 and "
        "BUDGET_VLLM_MODEL=glm-53-nvfp4 (the vLLM served name)")
    async def test_glm53_incident_reproduction_and_native_fix(self):
        """Failure shape and native fix on the glm53 profile: primes@128
        with native budgeting off burns the whole cap on reasoning
        (length, empty content, cap/cap reasoning); effort low injects
        thinking_token_budget 64 (min(64, 128-64)) and reasoning lands
        exactly on 64 -- a binding limit below the prompt's natural
        thinking -- with real content."""
        cfg = self.app["config"]["backends"][FIXTURE.name]
        payload = {"model": FIXTURE.name,
            "messages": [{"role": "user", "content": FIXTURE.primes_prompt}],
            "max_tokens": FIXTURE.small_cap, "temperature": 0, "seed": 42}
        cfg["reasoning"]["native_budget"] = False
        async with self.client.post("/v1/chat/completions", headers=AUTH, json=payload) as res:
            self.assertEqual(res.status, 200)
            baseline = await res.json()
        before = baseline["choices"][0]
        self.assertEqual(before["finish_reason"], "length")
        self.assertFalse(before["message"].get("content"))
        self.assertEqual(baseline["usage"]["completion_tokens"], FIXTURE.small_cap)
        self.assertEqual(
            baseline["usage"]["completion_tokens_details"]["reasoning_tokens"],
            FIXTURE.small_cap)
        cfg["reasoning"]["native_budget"] = True
        payload["reasoning_effort"] = "low"
        async with self.client.post("/v1/chat/completions", headers=AUTH, json=payload) as res:
            self.assertEqual(res.status, 200)
            limited = await res.json()
        reasoning = limited["llmproxy"]["budget"]["reasoning"]
        self.assertEqual(reasoning["forwarded_tokens"],
            FIXTURE.effort_budgets["low"])  # min(64, 128-64) == 64
        self.assertEqual(reasoning["source"], "proxy")
        self.assertTrue(limited["choices"][0]["message"].get("content"))
        self.assertLessEqual(limited["usage"]["completion_tokens"],
            FIXTURE.small_cap)
        self.assertEqual(
            limited["usage"]["completion_tokens_details"]["reasoning_tokens"],
            FIXTURE.effort_budgets["low"])  # binding: natural primes thinking >= 128
        self.assertEqual(len(await self.events(4)), 4)
        print("live-budget: glm53 incident baseline 128/128 reasoning; "
            "native budget produced content in",
            limited["usage"]["completion_tokens"], "tokens with reasoning",
            limited["usage"]["completion_tokens_details"]["reasoning_tokens"])

    @unittest.skipUnless(isinstance(FIXTURE, GLM53Fixture) and MODEL == "glm-53-nvfp4",
        "glm53 fixture: set BUDGET_VLLM_FIXTURE=glm53 and "
        "BUDGET_VLLM_MODEL=glm-53-nvfp4 (the vLLM served name)")
    async def test_glm53_client_ttb_precedence(self):
        """An explicit client thinking_token_budget wins over the effort
        map (budget.py:127-129); 16 binds below natural primes thinking
        (>= 128), so reasoning lands exactly on 16 with non-empty
        content."""
        payload = {"model": FIXTURE.name,
            "messages": [{"role": "user", "content": FIXTURE.primes_prompt}],
            "max_tokens": FIXTURE.small_cap, "temperature": 0, "seed": 42,
            "reasoning_effort": "low", "thinking_token_budget": 16}
        async with self.client.post("/v1/chat/completions", headers=AUTH, json=payload) as res:
            self.assertEqual(res.status, 200)
            data = await res.json()
        reasoning = data["llmproxy"]["budget"]["reasoning"]
        self.assertEqual(reasoning["forwarded_tokens"], 16)
        self.assertEqual(reasoning["requested_tokens"], 16)
        self.assertEqual(reasoning["source"], "client")
        self.assertTrue(data["choices"][0]["message"].get("content"))
        self.assertEqual(
            data["usage"]["completion_tokens_details"]["reasoning_tokens"], 16)
        self.assertEqual(len(await self.events(2)), 2)

    @unittest.skipUnless(isinstance(FIXTURE, GLM53Fixture) and MODEL == "glm-53-nvfp4",
        "glm53 fixture: set BUDGET_VLLM_FIXTURE=glm53 and "
        "BUDGET_VLLM_MODEL=glm-53-nvfp4 (the vLLM served name)")
    async def test_glm53_reserve_400_edges(self):
        """max_tokens == reserve leaves zero content tokens (budget.py:
        131-136) and conflicting chat limits are rejected before
        forwarding (budget.py:80-81); neither request is billed."""
        payload = {"model": FIXTURE.name,
            "messages": [{"role": "user", "content": FIXTURE.math_prompt}],
            "max_tokens": FIXTURE.reserve, "temperature": 0, "seed": 42}
        async with self.client.post("/v1/chat/completions", headers=AUTH, json=payload) as res:
            self.assertEqual(res.status, 400)
        payload["max_completion_tokens"] = 64
        async with self.client.post("/v1/chat/completions", headers=AUTH, json=payload) as res:
            self.assertEqual(res.status, 400)
        # Fresh per-test database: waiting out the whole billing window must
        # stay empty -- 400s never reach the backend and never bill.
        self.assertEqual(await self.events(1), [])
