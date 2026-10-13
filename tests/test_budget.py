"""Budget contract: real HTTP proxy, deterministic upstream, real SQLite billing."""

import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import time
import unittest
import warnings
from unittest import mock

import aiohttp.test_utils
import aiohttp.web

from llmproxy import auth, budget, config, metrics, provenance
from llmproxy.app import create_app, reload_config
from llmproxy.db import get_db


AUTH = {"Authorization": "Bearer budget-test-token"}
POLICY = {"mode": "enforce", "default_max_tokens": 100,
    "ceiling": 200, "apply_default": True, "clamp": True}


class BillingTestServer(aiohttp.test_utils.TestServer):
    async def _make_runner(self, **kwargs):
        kwargs["handler_cancellation"] = False
        return await super()._make_runner(**kwargs)


def contract(body=None, protocol="chat", policy=None, reasoning=None):
    return budget.Contract({"budget": {"enabled": True}},
        {"output_budget": copy.deepcopy(POLICY if policy is None else policy),
            "reasoning": reasoning or {}}, protocol, body if body is not None else {})


class TestBudgetPolicy(unittest.TestCase):
    def test_observation_leaves_all_budget_parameters_untouched(self):
        for protocol in budget.PATHS.values():
            for body in ({}, {"max_tokens": 999999},
                    {"max_completion_tokens": 100, "max_tokens": 200},
                    {"max_tokens": False}, {"max_tokens": None}):
                original = copy.deepcopy(body)
                c = contract(body, protocol, dict(POLICY, mode="observe"),
                    {"native_budget": True, "reserve_content_tokens": 10})
                self.assertEqual(body, original)
                self.assertIsNone(c.metadata["effective_max_output_tokens"])

    def test_protocol_names_default_and_clamp(self):
        for protocol, key in (("chat", "max_completion_tokens"),
                ("completions", "max_tokens"), ("responses", "max_output_tokens"),
                ("messages", "max_tokens")):
            with self.subTest(protocol=protocol):
                body = {}
                c = contract(body, protocol)
                self.assertEqual(body[key], 100)
                self.assertEqual(c.metadata["source"], "proxy")
                body = {key: 300}
                c = contract(body, protocol)
                self.assertEqual(body[key], 200)
                self.assertTrue(c.metadata["clamped"])
                self.assertEqual(c.metadata["requested_max_output_tokens"], 300)

    def test_independent_enforcement_flags(self):
        body = {"max_tokens": 300}
        contract(body, policy=dict(POLICY, clamp=False))
        self.assertEqual(body["max_tokens"], 300)
        body = {}
        c = contract(body, policy=dict(POLICY, apply_default=False))
        self.assertEqual(body, {})
        self.assertEqual(c.metadata["source"], "unknown")
        self.assertNotIn("X-LLMProxy-Max-Output-Tokens", c.headers())

    def test_invalid_limits_and_conflicts_rejected_before_forward(self):
        for value in (0, -1, True, "10", 1.5, None, {}):
            with self.subTest(value=value), self.assertRaises(budget.BudgetError):
                contract({"max_tokens": value})
        with self.assertRaises(budget.BudgetError):
            contract({"max_tokens": 10, "max_completion_tokens": 20})

    def test_native_reasoning_cap_preserves_lower_client_limit(self):
        cfg = {"native_budget": True, "reserve_content_tokens": 20,
            "effort_budgets": {"low": 30, "high": 150}}
        for body, expected in (({}, 80), ({"reasoning_effort": "low"}, 30),
                ({"reasoning_effort": "high"}, 80),
                ({"thinking_token_budget": 5, "reasoning_effort": "high"}, 5)):
            c = contract(body, reasoning=cfg)
            self.assertEqual(body["thinking_token_budget"], expected)
            self.assertEqual(c.metadata["reasoning"]["forwarded_tokens"], expected)
        with self.assertRaises(budget.BudgetError):
            contract({"max_tokens": 10}, reasoning=cfg)
        for value in ({}, [], "unsupported"):
            with self.assertRaises(budget.BudgetError):
                contract({"reasoning_effort": value}, reasoning=cfg)

    def test_disabled_reasoning_and_unverified_endpoints_are_preserved(self):
        cfg = {"native_budget": True, "reserve_content_tokens": 20}
        for body in ({"reasoning_effort": "none"},
                {"chat_template_kwargs": {"enable_thinking": False}},
                {"thinking": {"type": "disabled"}}):
            contract(body, reasoning=cfg)
            self.assertNotIn("thinking_token_budget", body)
        body = {"reasoning": {"effort": "high"}}
        contract(body, "responses", reasoning=cfg)
        self.assertNotIn("thinking_token_budget", body)

    def test_unsupported_native_reasoning_endpoints_rejected(self):
        for endpoint in ("responses", "messages"):
            with self.assertRaises(config.ConfigError):
                config.validate({"backends": {"m": {"type": "chat",
                    "reasoning": {"native_budget": True, "endpoints": [endpoint]}}}})

    def test_multi_choice_truncation_is_not_inferred_from_aggregate_usage(self):
        c = contract({"max_tokens": 100, "n": 2})
        c.observe({"choices": [{"index": 0, "finish_reason": "stop"},
            {"index": 1, "finish_reason": "length"}], "usage": {
                "completion_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 120}}})
        d = c.diagnostic()
        self.assertTrue(d["occurred"])
        self.assertEqual(d["reason"], "unknown")
        self.assertEqual(d["non_reasoning_output_tokens"], 30)
        self.assertEqual(len(d["backend_stop_reasons"]), 2)

    def test_missing_invalid_and_zero_reasoning_details(self):
        for details, expected in (({}, False), ({"reasoning_tokens": 11}, False),
                ({"reasoning_tokens": -1}, False), ({"reasoning_tokens": False}, False),
                ({"reasoning_tokens": 0}, True)):
            c = contract()
            c.observe({"choices": [{"finish_reason": "length"}], "usage": {
                "completion_tokens": 10, "completion_tokens_details": details}})
            self.assertEqual(c.diagnostic()["details_available"], expected)

    def test_serialization_fallback_preserves_bytes_and_fields(self):
        raw = b'{"value":1e400,"llmproxy":{"upstream":true},"usage":{}}'
        import decimal
        data = json.loads(raw, parse_float=decimal.Decimal)
        out, _ = provenance.mark({}, raw, data, "m", "r",
            extra_fields={"llmproxy": {"version": 1}})
        self.assertEqual(out, raw)
        self.assertEqual(data["llmproxy"], {"upstream": True})
        self.assertNotIn("provenance", data)

    def test_zero_reasoning_count_with_visible_reasoning_is_inconsistent(self):
        c = contract(protocol="responses")
        c.observe({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "thinking"}]}],
            "usage": {"output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 0}}})
        d = c.diagnostic()
        self.assertFalse(d["details_available"])
        self.assertEqual(d["details_status"], "inconsistent")
        self.assertIsNone(d["non_reasoning_output_tokens"])

    def test_observer_retains_only_counts_and_handles_null_details(self):
        c = contract()
        c.observe({"usage": {"completion_tokens": 10,
            "completion_tokens_details": None, "large_backend_field": ["unused"] * 100}})
        self.assertEqual(c.raw_usage, {"completion_tokens": 10, "completion_tokens_details": {}})
        self.assertFalse(c.diagnostic()["details_available"])
        c.observe({"usage": {"completion_tokens_details": {"reasoning_tokens": 8,
            "large_backend_field": ["unused"] * 100}}})
        self.assertEqual(c.raw_usage["completion_tokens_details"], {"reasoning_tokens": 8})
        self.assertEqual(c.diagnostic()["reasoning_tokens"], 8)


class TestBudgetConfig(unittest.TestCase):
    def test_invalid_configuration(self):
        cases = [{"budget": True}, {"budget": {"enabled": "true"}},
            {"budget": {"echo_body": 1}}, {"budget": {"typo": True}}]
        for policy in ({"mode": "maybe"}, {"ceiling": True},
                {"default_max_tokens": 201, "ceiling": 200},
                {"clamp": True}, {"apply_default": True},
                {"mode": "observe", "typo": True}):
            cases.append({"backends": {"m": {"type": "chat", "output_budget": policy}}})
        for reasoning in ({"native_budget": "true"}, {"endpoints": ["completions"]},
                {"reserve_content_tokens": 0}, {"effort_budgets": {"low": -1}},
                {"default_effort": "high"}, {"default_effort": []}, {"mapping": {}}):
            cases.append({"backends": {"m": {"type": "chat", "reasoning": reasoning}}})
        for cfg in cases:
            with self.subTest(cfg=cfg), self.assertRaises(config.ConfigError):
                config.validate(cfg)


class TestBudgetHTTP(aiohttp.test_utils.AioHTTPTestCase):
    async def asyncSetUp(self):
        warnings.simplefilter("ignore", aiohttp.web.NotAppKeyWarning)
        auth.flush_cache()
        self.received = []
        self.upstream_bytes = []
        self.block_stream = False
        self.stream_started = asyncio.Event()
        self.release_stream = asyncio.Event()
        upstream = aiohttp.web.Application()
        async def health(req):
            return aiohttp.web.Response()
        upstream.router.add_get("/health", health)
        for path in budget.PATHS:
            upstream.router.add_post(path, self.backend_handler)
        self.backend = aiohttp.test_utils.TestServer(upstream)
        await self.backend.start_server()
        await super().asyncSetUp()

    async def asyncTearDown(self):
        self.release_stream.set()
        await super().asyncTearDown()
        await self.backend.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    async def get_application(self):
        self.db_fd, self.db_path = tempfile.mkstemp()
        cfg = {"timeout_connect": 2, "timeout_read": 5,
            "db": {"uri": "sqlite://%s" % self.db_path},
            "budget": {"enabled": True}, "provenance": {"enabled": False},
            "backends": {"public": {"url": str(self.backend.make_url("")),
                "token": "backend-only", "model": "actual-model", "device": "test",
                "type": "chat", "max_model_len": 1000, "output_budget": dict(POLICY)}}}
        app = await create_app(cfg)
        db = await get_db(cfg["db"]["uri"])
        await db.db.execute("INSERT INTO api_key (id,secret,type) VALUES (?,?,?)",
            ("budget-user", hashlib.sha256(b"budget-test-token").hexdigest(), "LLM"))
        await db.db.commit()
        await db.close()
        return app

    async def get_server(self, app):
        # Match production: a vanished client must not cancel the billing tail.
        return BillingTestServer(app)

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

    async def backend_handler(self, req):
        body = await req.json()
        self.received.append(body)
        if body.get("_error"):
            return aiohttp.web.json_response({"error": {"message": "Input too long"}}, status=400)
        chat = req.path in ("/v1/chat/completions", "/v1/completions")
        details = {} if body.get("_missing_details") else {"reasoning_tokens": 8}
        usage = {"prompt_tokens": 3, "completion_tokens": 10,
            "completion_tokens_details": details} if chat else {
                "input_tokens": 3, "output_tokens": 10, "output_tokens_details": details}
        if chat:
            result = {"choices": [{"index": 0, "message": {"content": "42"},
                "finish_reason": "length"}], "usage": usage}
            chunks = [b'data: {"choices":[{"index":0,"delta":{"content":"42"},"finish_reason":"length"}]}\n\n',
                b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n",
                b"data: [DONE]\n\n"]
        elif req.path == "/v1/responses":
            result = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                "output": [], "usage": usage}
            chunks = [b"event: response.incomplete\ndata: " + json.dumps({
                "type": "response.incomplete", "response": result}).encode() + b"\n\n"]
        else:
            result = {"stop_reason": "max_tokens", "content": [], "usage": usage}
            chunks = [b"event: message_start\ndata: " + json.dumps({"type": "message_start",
                "message": {"usage": {"input_tokens": 3}}}).encode() + b"\n\n",
                b"event: message_delta\ndata: " + json.dumps({"type": "message_delta",
                    "delta": {"stop_reason": "max_tokens"},
                    "usage": {k: v for k, v in usage.items() if k != "input_tokens"}}).encode() + b"\n\n"]
        if body.get("stream"):
            self.upstream_bytes.append(b"".join(chunks))
            res = aiohttp.web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await res.prepare(req)
            await res.write(chunks[0])
            self.stream_started.set()
            if self.block_stream:
                await self.release_stream.wait()
            for chunk in chunks[1:]:
                await res.write(chunk)
            await res.write_eof()
            return res
        self.upstream_bytes.append(json.dumps(result).encode())
        return aiohttp.web.Response(body=self.upstream_bytes[-1], content_type="application/json")

    async def test_four_protocols_stream_and_json_billing_and_bytes(self):
        for path, protocol in budget.PATHS.items():
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    async with self.client.post(path, headers=AUTH, json={
                            "model": "public", "messages": [], "input": "hi", "stream": stream}) as res:
                        self.assertEqual(res.status, 200)
                        self.assertEqual(res.headers["X-LLMProxy-Max-Output-Tokens"], "100")
                        self.assertEqual(res.headers["X-LLMProxy-Budget-Source"], "proxy")
                        self.assertIn("X-LLMProxy-Truncation", res.headers["Access-Control-Expose-Headers"])
                        raw = await res.read()
                        if stream:
                            self.assertEqual(res.headers["X-LLMProxy-Truncation"], "pending")
                            self.assertEqual(raw, self.upstream_bytes[-1])
                        else:
                            d = json.loads(raw)["llmproxy"]
                            self.assertTrue(d["truncation"]["occurred"])
                            self.assertEqual(d["truncation"]["reason"], "unknown")
                            self.assertEqual(d["truncation"]["reasoning_tokens"], 8)
                            self.assertEqual(d["truncation"]["non_reasoning_output_tokens"], 2)
                        self.assertEqual(self.received[-1]["model"], "actual-model")
        rows = await self.events(16)
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(row["quantity"] for row in rows if row["product"].endswith("/completion")), 80)
        self.assertTrue(all(row["product"].startswith("public/test/") for row in rows))
        async with self.client.get("/metrics") as res:
            text = await res.text()
            self.assertIn('llmproxy_truncation_total{model="public",reason="unknown",stream="true"}', text)
            self.assertIn('llmproxy_reasoning_tokens_total{model="public"}', text)

    async def test_observe_mode_unknown_default_and_no_clamping(self):
        self.app["config"]["backends"]["public"]["output_budget"]["mode"] = "observe"
        for extra in ({}, {"max_tokens": 999999}):
            async with self.client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "public", **extra}) as res:
                self.assertEqual(res.status, 200)
                meta = (await res.json())["llmproxy"]["budget"]
                self.assertFalse(meta["clamped"])
                if extra:
                    self.assertEqual(self.received[-1]["max_tokens"], 999999)
                else:
                    self.assertNotIn("max_completion_tokens", self.received[-1])
                    self.assertNotIn("X-LLMProxy-Max-Output-Tokens", res.headers)
                    self.assertIsNone(meta["forwarded_max_output_tokens"])

    async def test_independent_body_headers_and_kill_switch(self):
        for headers_on, body_on in ((True, False), (False, True), (False, False)):
            self.app["config"]["budget"].update(echo_headers=headers_on, echo_body=body_on)
            async with self.client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "public"}) as res:
                self.assertEqual("X-LLMProxy-Extensions" in res.headers, headers_on)
                self.assertEqual("llmproxy" in await res.json(), body_on)
        self.app["config"]["budget"]["enabled"] = False
        async with self.client.post("/v1/chat/completions", headers=AUTH,
                json={"model": "public"}) as res:
            self.assertEqual(await res.read(), self.upstream_bytes[-1])
            self.assertNotIn("X-LLMProxy-Extensions", res.headers)
            self.assertNotIn("max_completion_tokens", self.received[-1])

    async def test_catalog_single_model_auth_and_bounded_metrics(self):
        async with self.client.get("/v1/models", headers=AUTH) as res:
            item = (await res.json())["data"][0]
            self.assertEqual(item["output_budget"]["default_max_tokens"], 100)
            self.assertNotIn("token", item)
        async with self.client.get("/v1/models/public", headers=AUTH) as res:
            self.assertEqual(await res.json(), item)
        async with self.client.get("/v1/models/public") as res:
            self.assertEqual(res.status, 401)
        for i in range(10):
            async with self.client.get("/v1/models/unknown-%d" % i, headers=AUTH) as res:
                self.assertEqual(res.status, 404)
                self.assertEqual((await res.json())["error"]["code"], "model_not_found")
        async with self.client.get("/metrics") as res:
            text = await res.text()
            self.assertIn('path="/v1/models/{id}"', text)
            self.assertNotIn('path="/v1/models/unknown-', text)
        self.app["config"]["budget"]["enabled"] = False
        async with self.client.get("/v1/models/public", headers=AUTH) as res:
            self.assertNotIn("output_budget", await res.json())

    async def test_errors_not_billed_and_backend_context_error_preserved(self):
        for body in ({"max_tokens": False}, {"max_tokens": 10, "max_completion_tokens": 20}):
            async with self.client.post("/v1/chat/completions", headers=AUTH,
                    json={"model": "public", **body}) as res:
                self.assertEqual(res.status, 400)
        self.assertEqual(self.received, [])
        async with self.client.post("/v1/chat/completions", headers=AUTH,
                json={"model": "public", "max_tokens": 999999, "_error": True}) as res:
            self.assertEqual(res.status, 400)
            self.assertNotIn("X-LLMProxy-Extensions", res.headers)
        self.assertEqual(self.received[-1]["max_tokens"], 200)
        self.assertEqual(await self.events(0), [])

    async def test_missing_details_billing_and_provenance_shared_serialization(self):
        self.app["config"]["provenance"]["enabled"] = True
        async with self.client.post("/v1/chat/completions", headers=AUTH,
                json={"model": "public", "_missing_details": True}) as res:
            data = await res.json()
            self.assertFalse(data["llmproxy"]["truncation"]["details_available"])
            self.assertIn("provenance", data)
            self.assertIsNone(data["llmproxy"]["truncation"]["reasoning_tokens"])
        self.assertEqual(len(await self.events(2)), 2)

    async def test_reload_kill_switch_and_invalid_reload_retains_config(self):
        cfg = copy.deepcopy(self.app["config"])
        cfg["budget"]["enabled"] = False
        self.app["config"]["_path"] = "unused-test-path"
        with mock.patch("llmproxy.config.load", return_value=cfg):
            reload_config(self.app)
        async with self.client.post("/v1/chat/completions", headers=AUTH,
                json={"model": "public"}) as res:
            self.assertEqual(await res.read(), self.upstream_bytes[-1])
        with mock.patch("llmproxy.config.load", side_effect=config.ConfigError("invalid")):
            reload_config(self.app)
        self.assertFalse(self.app["config"]["budget"]["enabled"])

    async def test_disconnect_drains_terminal_usage_and_budget_diagnostics(self):
        self.block_stream = True
        before = metrics.REASONING_TOKENS_TOTAL.labels("public")._value.get()
        res = await self.client.post("/v1/chat/completions", headers=AUTH,
            json={"model": "public", "stream": True})
        await self.stream_started.wait()
        await res.content.read(1)
        rid = res.headers["X-Request-ID"]
        res.close()
        self.release_stream.set()
        rows = await self.events(2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["rid"] == rid for row in rows))
        # Metrics run just after DB commit; wait for the handler tail too.
        for _ in range(100):
            if metrics.REASONING_TOKENS_TOTAL.labels("public")._value.get() > before:
                break
            await asyncio.sleep(.01)
        self.assertEqual(metrics.REASONING_TOKENS_TOTAL.labels("public")._value.get(), before + 8)

    async def test_inflight_policy_snapshot_survives_reload(self):
        self.block_stream = True
        res = await self.client.post("/v1/chat/completions", headers=AUTH,
            json={"model": "public", "stream": True})
        self.app["config"]["budget"]["enabled"] = False
        self.app["config"]["backends"]["public"]["output_budget"]["default_max_tokens"] = 150
        self.release_stream.set()
        self.assertEqual(res.headers["X-LLMProxy-Max-Output-Tokens"], "100")
        self.assertEqual(await res.read(), self.upstream_bytes[-1])
        self.assertEqual(len(await self.events(2)), 2)

    @unittest.skipUnless(importlib.util.find_spec("openai"), "optional: install openai to test SDK compatibility")
    async def test_official_openai_sdk_parses_metadata_and_untouched_stream(self):
        import httpx
        import openai
        async with openai.AsyncOpenAI(api_key="budget-test-token",
                base_url=str(self.server.make_url("/v1")), max_retries=0,
                http_client=httpx.AsyncClient(trust_env=False)) as client:
            raw = await client.chat.completions.with_raw_response.create(
                model="public", messages=[{"role": "user", "content": "hi"}])
            completion = raw.parse()
            self.assertEqual(raw.headers["X-LLMProxy-Extensions"], "budget.v1")
            self.assertEqual(completion.model_extra["llmproxy"]["budget"]["forwarded_max_output_tokens"], 100)
            chunks = [chunk async for chunk in await client.chat.completions.create(
                model="public", messages=[{"role": "user", "content": "hi"}], stream=True)]
            self.assertEqual(chunks[-1].usage.completion_tokens, 10)
            self.assertEqual(chunks[0].choices[0].delta.content, "42")
            response = await client.responses.create(model="public", input="hi")
            self.assertTrue(response.model_extra["llmproxy"]["truncation"]["occurred"])
            events = [event async for event in await client.responses.create(
                model="public", input="hi", stream=True)]
            self.assertEqual(events[-1].type, "response.incomplete")
        self.assertEqual(len(await self.events(8)), 8)
