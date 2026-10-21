import asyncio
import copy
import datetime
import importlib.resources
import json
import tomllib
import unittest
import uuid
from unittest import mock

import aiohttp.web
import aiohttp.test_utils

from llmproxy import basal, billing, config, decision_adapters, proxy, ratelimit
from llmproxy.decision_contract import DecisionCapabilities, DecisionResult, DecisionUsage
from llmproxy.app import reload_config
from llmproxy.db import DatabaseError, _billing_document, get_db
from tests.test_proxy import LLMProxyAppTestCase


def payload():
    return {"model": "basal-20-max", "state": {"text": "Synthetic test"},
        "questions": {"category": {"criteria": {"yes": "Yes", "no": "No"}}}}


class TestDecisionConfig(unittest.TestCase):
    def test_packaged_basal_example_loads(self):
        source = importlib.resources.files("llmproxy").joinpath("config.toml").read_text()
        example = source[source.index('#[backends.basal-20-max]'):]
        cfg = tomllib.loads("\n".join(line.removeprefix("#") for line in example.splitlines()))
        config.validate(cfg)
        self.assertEqual(list(cfg["backends"]), ["basal-20-max"])
        self.assertEqual(cfg["backends"]["basal-20-max"]["decision_limits"]["max_branches"], 64)

    def test_configuration(self):
        backend = {"type": "decision", "url": "https://example.org",
            "model": "basal", "device": "a5000"}
        config.validate({"backends": {"basal": backend}})
        for delta in ({"token": None}, {"model": ""}, {"device": None},
                {"url": ""}, {"timeout": float("inf")}, {"timeout": float("nan")},
                {"decision_limits": []}, {"decision_limits": {"other": 1}},
                {"decision_limits": {"max_questions": True}},
                {"decision_limits": {"max_branches": 0}}, {"output_budget": {}}):
            with self.subTest(delta=delta), self.assertRaises(config.ConfigError):
                config.validate({"backends": {"basal": backend | delta}})

    def test_branches(self):
        body = payload()
        body["questions"] = {"q": {"type": "choice", "criteria": {str(i): str(i) for i in range(255)}}}
        self.assertEqual(basal.prepare(body, config.DECISION_LIMITS).branches, 1)
        for question, count in (({"type": "multi", "criteria": ["a", "b"]}, 2),
                ({"type": "tool", "multi": True, "tools": [{"name": "a"}]}, 1),
                ({"type": "tool", "multi": True, "tools": [{"name": "a"}], "extra": ["none"]}, 2)):
            body["questions"] = {"q": question}
            with self.subTest(question=question):
                self.assertEqual(basal.prepare(body, config.DECISION_LIMITS).branches, count)
                basal.prepare(body, config.DECISION_LIMITS | {"max_branches": count})
                if count > 1:
                    with self.assertRaises(ValueError):
                        basal.prepare(body, config.DECISION_LIMITS | {"max_branches": count - 1})

    def test_adapter_configuration(self):
        backend = {"type": "decision", "url": "https://example.org",
            "model": "basal", "device": "a5000"}
        config.validate({"backends": {"basal": backend | {"decision_adapter": "basal"}}})
        self.assertIs(decision_adapters.get_adapter(backend), basal.ADAPTER)
        for value in ("unknown", "", None, True, []):
            with self.subTest(adapter=value), self.assertRaises(config.ConfigError):
                config.validate({"backends": {"basal": backend | {"decision_adapter": value}}})
        with self.assertRaises(config.ConfigError):
            config.validate({"backends": {"chat": {"type": "chat", "decision_adapter": "basal"}}})

    def test_preparation_preserves_client_input_and_request_specific_validation(self):
        body = payload()
        original = copy.deepcopy(body)
        first = basal.ADAPTER.prepare(body, config.DECISION_LIMITS)
        second = basal.ADAPTER.prepare(body | {"questions": {"other": {"type": "noul"}}},
            config.DECISION_LIMITS)
        self.assertEqual(body, original)
        self.assertNotIn("model", first.body)
        self.assertEqual(first.question_count, 1)
        data = {"model": "basal", "answers": {"category": {"type": "choice", "choice": "yes",
            "probabilities": {"yes": 1}, "confidence": 1}},
            "usage": {"input_tokens": 15, "output_tokens": 0}}
        result = first.decode(data)
        self.assertEqual(result.usage, DecisionUsage(15, 0))
        self.assertEqual(result.answers, data["answers"])
        self.assertEqual(result.native, data)
        with self.assertRaises(ValueError):
            second.decode(data)

    def test_contract_rejects_unbillable_usage(self):
        for value in (-1, True, "1", 1.5, 2 ** 63, float("nan")):
            for counts in ((value, 0), (1, value)):
                with self.subTest(counts=counts), self.assertRaises(ValueError):
                    DecisionUsage(*counts)

    def test_reload(self):
        app = {"config": {"_path": "test", "backends": {} }}
        app = type("App", (dict,), {"logger": mock.Mock()})(app)
        backend = {"type": "decision", "url": "https://example.org", "model": "basal", "device": "gpu"}
        with mock.patch("llmproxy.config.load", return_value={"backends": {"basal": backend}}):
            reload_config(app)
        self.assertEqual(app["config"]["backends"]["basal"], backend)
        with mock.patch("llmproxy.config.load", side_effect=config.ConfigError("bad")):
            reload_config(app)
        self.assertIn("basal", app["config"]["backends"])
        with mock.patch("llmproxy.config.load", return_value={}):
            reload_config(app)
        self.assertEqual(app["config"]["backends"], {})


class TestDecisions(LLMProxyAppTestCase):
    async def get_server(self, app):
        class Server(aiohttp.test_utils.TestServer):
            async def _make_runner(self, **kwargs):
                kwargs["handler_cancellation"] = False
                return await super()._make_runner(**kwargs)
        return Server(app)

    async def get_application(self):
        app = await super().get_application()
        app["config"]["backends"]["basal-20-max"] = {
            "url": str(self.backend.make_url("" )).rstrip("/"),
            "type": "decision", "model": "Remek/basal-2.0-max-GGUF",
            "device": "a5000", "max_model_len": 32768, "timeout": 5}
        ratelimit.flush()
        return app

    async def post(self, body=None, path="/v1/systemone", token="mytoken", **kwargs):
        return await self.client.post(path, json=payload() if body is None else body,
            headers={"Authorization": "Bearer " + token}, **kwargs)

    @property
    def control(self):
        return self.backend.app["decision_control"]

    @property
    def calls(self):
        return self.backend.app["decision_calls"]

    @property
    def backend_config(self):
        return self.app["config"]["backends"]["basal-20-max"]

    async def test_native_route_forwards_and_bills_once(self):
        self.app["config"]["budget"] = {"enabled": True}
        for path in ("/v1/systemone",):
            body = payload()
            body["questions"].update({"q%d" % i: {"type": "choice"} for i in range(5)})
            res = await self.post(body, path)
            self.assertEqual(res.status, 200, await res.text())
            self.assertIn("X-Request-ID", res.headers)
            self.assertEqual((await res.json())["usage"], {"input_tokens": 880, "output_tokens": 0})
            forwarded, headers = self.calls[-1]
            self.assertEqual(forwarded, {k: v for k, v in body.items() if k != "model"})
            self.assertNotIn("Authorization", headers)
        self.assertEqual([x["quantity"] for x in await self.get_events()], [880, 0])
        self.assertEqual(self.backend.app["paths"].count("/v1/systemone"), 1)

    async def test_backend_token_and_catalog(self):
        self.backend_config["token"] = "upstream-secret"
        self.assertEqual((await self.post()).status, 200)
        self.assertEqual(self.calls[-1][1]["Authorization"], "Bearer upstream-secret")
        res = await self.client.get("/v1/models/basal-20-max", headers={"Authorization": "Bearer mytoken"})
        item = await res.json()
        self.assertEqual(item["type"], "decision")
        self.assertEqual(item["endpoints"], ["/v1/systemone", "/v1/decisions"])
        self.assertEqual(item["decision_capabilities"]["openai_question_types"], ["predicate", "choice", "score"])
        self.assertEqual(item["decision_capabilities"]["openai_input"], "text_or_user_messages")
        self.assertFalse(item["decision_capabilities"]["openai_streaming"])
        self.assertNotIn("chat_input", item["decision_capabilities"])
        self.assertNotIn("token", item)
        self.assertNotIn("url", item)

    async def test_native_response_names_the_alias_and_keeps_the_rest(self):
        # The backend echoes the model id the proxy sent it (the GGUF repo);
        # the client asked for the alias. Everything else is the backend's:
        # values, key order, unknown fields and non-ASCII text.
        self.backend_config["decision_adapter"] = "basal"
        raw = json.dumps({"model": "Remek/basal-2.0-max-GGUF", "answers": {"category": {
            "type": "choice", "choice": "tak", "probabilities": {"tak": 0.749313235282898,
                "nie": 1e-05}, "confidence": 0.749313235282898, "uwaga": "zażółć"}},
            "usage": {"input_tokens": 23, "output_tokens": 0},
            "backend_metadata": {"calibration": "native"}}, indent=2, ensure_ascii=False).encode()
        self.control["raw"] = raw
        response = await self.post(path="/v1/systemone")
        self.assertEqual(response.status, 200)
        text = await response.text()
        self.assertEqual(json.loads(text), json.loads(raw) | {"model": "basal-20-max"})
        self.assertEqual(list(json.loads(text)), ["model", "answers", "usage", "backend_metadata"])
        self.assertIn('"confidence":0.749313235282898', text)
        self.assertIn("zażółć", text)
        self.assertNotIn("Remek", text)
        self.assertEqual([event["quantity"] for event in await self.get_events()], [23, 0])

    async def test_other_adapter_routes_and_bills_its_normalized_usage(self):
        # A deliberately non-Basal wire format proves the HTTP service does
        # not require Basal paths, question fields or native token field names.
        received = []

        async def handle(request):
            received.append((await request.json(), request.headers.get("Authorization")))
            return aiohttp.web.json_response({"value": "yes", "tokens": 12})

        class Prepared:
            question_count = 1
            branches = 1

            def __init__(self, body):
                self.body = {"text": body["document"]}

            def decode(self, data):
                if data.get("value") != "yes":
                    raise ValueError("invalid_answers")
                return DecisionResult("other", {"result": data["value"]},
                    DecisionUsage(data["tokens"], 0), data)

        class Adapter:
            path = "decide"
            capabilities = DecisionCapabilities(frozenset({"binary"}))

            def prepare(self, body, limits):
                return Prepared(body)

        app = aiohttp.web.Application()
        app.router.add_post("/decide", handle)
        async with aiohttp.test_utils.TestServer(app) as server:
            self.app["config"]["backends"]["other-decision"] = {
                "type": "decision", "decision_adapter": "other", "model": "other",
                "device": "cpu", "url": str(server.make_url("")).rstrip("/"), "token": "other-key"}
            with mock.patch.dict(decision_adapters._ADAPTERS, {"other": Adapter()}):
                for path in ("/v1/systemone",):
                    res = await self.post({"model": "other-decision", "document": "Synthetic"}, path=path)
                    self.assertEqual(res.status, 200, await res.text())
                    self.assertEqual(await res.json(), {"value": "yes", "tokens": 12})
                res = await self.post({"model": "other-decision", "input": "Synthetic",
                    "questions": [{"type": "predicate", "instructions": "Yes?"}]}, path="/v1/decisions")
                self.assertEqual(res.status, 400)
                catalog = await self.client.get("/v1/models/other-decision",
                    headers={"Authorization": "Bearer mytoken"})
                self.assertEqual((await catalog.json())["endpoints"], ["/v1/systemone"])
        self.assertEqual(received, [({"text": "Synthetic"}, "Bearer other-key")])
        self.assertEqual(self.calls, [])
        self.assertEqual([event["quantity"] for event in await self.get_events()], [12, 0])

    async def test_rejections_do_not_contact_backend(self):
        cases = [(payload(), "badtoken", 401),
            (payload() | {"model": "missing"}, "mytoken", 404),
            (payload() | {"model": "mymodel"}, "mytoken", 404),
            (payload() | {"questions": []}, "mytoken", 400),
            (payload() | {"questions": {"q": {"type": []}}}, "mytoken", 400),
            (payload() | {"stream": True}, "mytoken", 400)]
        for field in basal.CHAT_FIELDS:
            cases.append((payload() | {field: []}, "mytoken", 400))
        body = payload()
        del body["state"]
        cases.append((body, "mytoken", 400))
        for path in ("/v1/systemone",):
            for body, token, status in cases:
                with self.subTest(path=path, body=body):
                    self.assertEqual((await self.post(body, path=path, token=token)).status, status)
        res = await self.post(path="/v1/chat/completions")
        self.assertEqual(res.status, 404)
        self.assertEqual((await res.json())["error"]["code"], "model_not_supported")
        self.assertEqual(self.calls, [])
        self.assertEqual(await self.get_events(), [])

    async def test_body_size_and_content_type(self):
        for headers, data, status in (({"Content-Type": "multipart/form-data; boundary=x"}, b"", 415),
                ({"Content-Type": "text/plain"}, b"{}", 415),
                ({"Content-Type": "application/json"}, b"{", 400)):
            res = await self.client.post("/v1/systemone", data=data,
                headers=headers | {"Authorization": "Bearer mytoken"})
            self.assertEqual(res.status, status)
        self.app["config"]["max_json_body"] = 100
        self.assertEqual((await self.post()).status, 413)
        async def chunks():
            yield json.dumps(payload()).encode()
        res = await self.client.post("/v1/systemone", data=chunks(),
            headers={"Authorization": "Bearer mytoken", "Content-Type": "application/json"})
        self.assertEqual(res.status, 413)
        self.assertEqual(self.calls, [])

    async def test_limits_before_forward(self):
        self.backend_config["decision_limits"] = {"max_questions": 1, "max_branches": 2}
        body = payload()
        body["questions"]["second"] = {"type": "noul"}
        self.assertEqual((await self.post(body)).status, 400)
        body["questions"] = {"q": {"type": "multi", "criteria": ["a", "b", "c"]}}
        self.assertEqual((await self.post(body)).status, 400)
        self.assertEqual(self.calls, [])

    async def test_invalid_usage_and_answers_are_not_billed(self):
        valid = {"model": "basal", "answers": {"category": {"type": "choice", "choice": "yes",
            "probabilities": {"yes": 0.9, "no": 0.1}, "confidence": 0.9}},
            "usage": {"input_tokens": 880, "output_tokens": 0}}
        cases = [valid | {"usage": None}, valid | {"answers": {}}, [], valid | {"model": None}]
        for field in ("input_tokens", "output_tokens"):
            for value in (None, True, "1", -1, 1.5, 2 ** 63):
                cases.append(valid | {"usage": valid["usage"] | {field: value}})
            usage = dict(valid["usage"])
            del usage[field]
            cases.append(valid | {"usage": usage})
        for value in (float("nan"), float("inf"), True, -0.1, 1.1):
            case = copy.deepcopy(valid)
            case["answers"]["category"]["probabilities"]["yes"] = value
            cases.append(case)
        for data in cases:
            with self.subTest(data=data):
                self.control["response"] = data
                self.assertEqual((await self.post()).status, 502)
        self.assertEqual(await self.get_events(), [])

    async def test_response_limits_and_invalid_json(self):
        self.backend_config["decision_limits"] = {"max_response_bytes": 100}
        for chunked in (False, True):
            self.control.update(raw=b"x" * 101, chunked=chunked)
            self.assertEqual((await self.post()).status, 502)
        for raw, content_type in ((b"{", "application/json"), (b"<html>", "text/html"),
                (b"data: {}\n\n", "text/event-stream")):
            self.control.update(raw=raw, content_type=content_type)
            self.assertEqual((await self.post()).status, 502)
        self.assertEqual(await self.get_events(), [])

    async def test_errors_timeout_no_retry_or_billing(self):
        for status, expected in ((422, 422), (503, 502), (201, 502)):
            self.control.update(status=status, raw=b'{"error":"synthetic"}')
            res = await self.post()
            self.assertEqual(res.status, expected)
            if status == 422:
                self.assertEqual(await res.json(), {"error": "synthetic"})
        self.control.clear()
        self.backend_config["timeout"] = 0.02
        self.control["delay"] = 0.1
        self.assertEqual((await self.post()).status, 504)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(await self.get_events(), [])

    async def test_rpm_per_user_and_alias(self):
        self.backend_config["rate_limit"] = {"rpm": 1}
        await self.insert_key("second-token", "second")
        self.assertEqual((await self.post()).status, 200)
        self.assertEqual((await self.post(path="/v1/systemone")).status, 429)
        self.assertEqual((await self.post(path="/v1/systemone")).status, 429)
        self.assertEqual((await self.post(token="second-token")).status, 200)
        self.assertEqual(len(self.calls), 2)

    async def test_concurrency_released_on_error(self):
        self.backend_config["rate_limit"] = {"concurrency": 1}
        started, release = asyncio.Event(), asyncio.Event()
        self.control.update(started=started, release=release, raw=b"invalid")
        task = asyncio.create_task(self.post())
        await asyncio.wait_for(started.wait(), 2)
        self.assertEqual((await self.post()).status, 429)
        release.set()
        self.assertEqual((await task).status, 502)
        self.control.clear()
        self.assertEqual((await self.post()).status, 200)

    async def test_disconnect_still_bills(self):
        # TestServer mirrors app.py's explicit handler_cancellation=False.
        self.assertFalse(self.server.runner.server.handler_cancellation)
        started, release = asyncio.Event(), asyncio.Event()
        self.control.update(started=started, release=release)
        reader, writer = await asyncio.open_connection(self.server.host, self.server.port)
        raw = json.dumps(payload()).encode()
        writer.write(b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer mytoken\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(raw)).encode() + b"\r\n\r\n" + raw)
        await writer.drain()
        await asyncio.wait_for(started.wait(), 2)
        writer.close()
        await writer.wait_closed()
        release.set()
        self.assertEqual(len(await self.get_events(expect=2)), 2)

    async def test_large_aggregate_usage(self):
        self.control["response"] = {"model": "basal", "answers": {"category": {
            "type": "choice", "choice": "yes", "probabilities": {"yes": 1}, "confidence": 1}},
            "usage": {"input_tokens": 100000, "output_tokens": 0}}
        self.assertEqual((await self.post()).status, 200)
        self.assertEqual((await self.get_events())[0]["quantity"], 100000)

    async def test_native_types_and_tool_definitions_are_preserved(self):
        body = payload() | {"questions": {
            "category": {"type": "choice", "criteria": {"refund": "Refund", "other": "Other"}},
            "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
            "urgency": {"type": "score", "levels": ["Low", "Medium", "High"]},
            "labels": {"type": "multi", "criteria": {"damage": "Damaged product", "refund": "Refund requested"}},
            "action": {"type": "act", "criteria": {"true": "Refund warranted", "false": "Refund not warranted"},
                "costs": {"wrong": 10, "defer": 0.1}},
            "tool": {"type": "tool", "multi": True, "tools": [{"type": "function",
                "function": {"name": "refund_order", "description": "Refund a damaged order",
                    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"]}}}]},
        }}
        answers = {
            "category": {"type": "choice", "choice": "refund", "confidence": 1,
                "probabilities": {"refund": 1, "other": 0}},
            "refund": {"type": "noul", "noul": 0.9, "confidence": 0.9,
                "probabilities": {"true": 0.9, "false": 0.1}},
            "urgency": {"type": "score", "score": 1, "confidence": 1,
                "probabilities": {"0": 0, "1": 1, "2": 0}},
            "labels": {"type": "multi", "selected": [], "set_confidence": 1,
                "probabilities": {"damage": 0, "refund": 0}, "threshold": 0.5},
            "action": {"type": "act", "action": "defer", "confidence": 0.5,
                "expected_costs": {"defer": 0.1, "true": 5},
                "probabilities": {"true": 0.5, "false": 0.5}},
            "tool": {"type": "tool", "multi": True, "tools": [],
                "set_confidence": 0.9, "probabilities": {"refund_order": 0.1}},
        }
        data = {"model": "basal", "answers": answers,
            "usage": {"input_tokens": 880, "output_tokens": 0, "branches": 7,
                "longest_prompt_tokens": 115, "latency_ms": 12.3}}
        self.control["response"] = data
        self.assertEqual(await (await self.post(body)).json(), data | {"model": "basal-20-max"})
        self.assertEqual(self.calls[0][0], {k: v for k, v in body.items() if k != "model"})
        self.assertEqual(len(await self.get_events()), 2)

    async def test_missing_model_is_400_on_both_frontends(self):
        openai_body = {"input": "Synthetic test", "questions": [
            {"type": "predicate", "instructions": "Is this a test?"}]}
        for path, body in (("/v1/systemone", {k: v for k, v in payload().items() if k != "model"}),
                ("/v1/decisions", openai_body), ("/v1/decisions", openai_body | {"model": None})):
            with self.subTest(path=path, body=body):
                res = await self.post(body, path)
                self.assertEqual(res.status, 400)
                self.assertEqual((await res.json())["error"]["param"], "model")
        self.assertEqual(self.calls, [])

    async def test_native_backend_errors(self):
        # A client error keeps the backend's status and message (only the
        # backend validates native semantics). Anything else is the proxy's
        # problem: the client's key was accepted, so a backend 401/403 must
        # not read as "your key is wrong", and no upstream body comes back.
        for status in (400, 413, 422, 429):
            with self.subTest(status=status):
                self.control.update(status=status, raw=b'{"error":"state is required"}')
                res = await self.post()
                self.assertEqual(res.status, status)
                self.assertEqual(await res.json(), {"error": "state is required"})
        for status in (401, 403, 404, 405, 409):
            with self.subTest(status=status):
                self.control.update(status=status, raw=b'{"error":"upstream secret detail"}')
                res = await self.post()
                self.assertEqual(res.status, 502)
                error = (await res.json())["error"]
                self.assertEqual(error["type"], "server_error")
                self.assertNotIn("upstream secret detail", json.dumps(error))
        self.assertEqual(await self.get_events(), [])

    async def test_error_body_is_not_logged(self):
        self.control.update(status=422, raw=b'{"error":"private synthetic text"}')
        with self.assertLogs(self.app.logger, level="ERROR") as logs:
            self.assertEqual((await self.post()).status, 422)
        self.assertNotIn("private synthetic text", " ".join(logs.output))

    async def test_revoked_key_stops_before_backend(self):
        self.assertEqual((await self.post()).status, 200)
        db = await get_db(self.app["config"]["db"]["uri"])
        await db.db.execute("DELETE FROM api_key WHERE id = 'myuser'")
        await db.db.commit()
        await db.close()
        from llmproxy import auth
        auth.flush_cache()
        self.assertEqual((await self.post()).status, 401)
        self.assertEqual(len(self.calls), 1)

    async def test_metrics_failure_does_not_drop_billing_or_success(self):
        with mock.patch("llmproxy.metrics.observe_text_tokens", side_effect=RuntimeError("synthetic")):
            self.assertEqual((await self.post()).status, 200)
        self.assertEqual(len(await self.get_events()), 2)

    async def test_handler_does_not_return_success_when_billing_fails(self):
        with mock.patch("llmproxy.billing.record", mock.AsyncMock(side_effect=RuntimeError("synthetic"))):
            self.assertEqual((await self.post()).status, 500)
        self.assertEqual(await self.get_events(), [])


class TestDecisionBilling(unittest.IsolatedAsyncioTestCase):
    async def test_database_failure_preserves_fail_loud_policy(self):
        db = mock.Mock(billing_record_add=mock.AsyncMock(side_effect=DatabaseError("synthetic")))
        app = type("App", (dict,), {"logger": mock.Mock()})(config={"db": {"uri": "unused"}})
        req = type("Request", (dict,), {"app": app})(request_id=uuid.uuid4())
        with mock.patch("llmproxy.billing.get_db", mock.AsyncMock(return_value=db)):
            with self.assertRaises(aiohttp.web.GracefulExit):
                await billing.record(req, {"id": "test"}, {"basal/a5000/prompt": 880})

    def test_mongo_document_resources(self):
        resources = {"basal-20-max/a5000/prompt": 880, "basal-20-max/a5000/completion": 0}
        user = {key: "test" for key in ("_namespace", "_user_id", "id", "_org_id", "_tier")}
        result = _billing_document(user, datetime.datetime.now(datetime.UTC), resources, uuid.uuid4())
        self.assertEqual(result["resources"], resources)
