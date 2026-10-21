import asyncio
import importlib.util
import json
import unittest
from unittest import mock

from llmproxy import ratelimit
from llmproxy.decision_openai import PATH
from tests.test_proxy import LLMProxyAppTestCase


def has_decisions_sdk():
    try:
        return importlib.util.find_spec("openai.resources.decisions") is not None
    except ModuleNotFoundError:
        return False


def payload():
    return {"model": "basal-20-max", "input": "The customer received a damaged product.",
        "questions": [
            {"type": "predicate", "name": "damaged", "instructions": "Is the product damaged?"},
            {"type": "choice", "instructions": "Choose a value.",
                "choices": [{"value": True, "description": "Boolean"}, {"value": "true"}]},
            {"type": "score", "name": "severity", "instructions": "Rate the damage.",
                "levels": [{"label": "Low"}, {"label": "Medium"}, {"label": "High", "description": "Unusable"}]}]}


def backend_response():
    return {"model": "Remek/basal-2.0-max-GGUF", "answers": {
        "q2": {"type": "score", "score": 1.7, "confidence": 0.8,
            "probabilities": {"2": 0.8, "0": 0.1, "1": 0.1}},
        "q1": {"type": "choice", "choice": "o0", "confidence": 0.9,
            "probabilities": {"o1": 0.1, "o0": 0.9}},
        "q0": {"type": "noul", "noul": 0.95, "confidence": 0.95,
            "probabilities": {"true": 0.95, "false": 0.05}}},
        "usage": {"input_tokens": 123, "output_tokens": 0}}


class TestOpenAIDecisions(LLMProxyAppTestCase):
    async def get_application(self):
        app = await super().get_application()
        app["config"]["backends"]["basal-20-max"] = {
            "type": "decision", "url": str(self.backend.make_url("")).rstrip("/"),
            "model": "Remek/basal-2.0-max-GGUF", "device": "a5000", "timeout": 5}
        self.control["response"] = backend_response()
        ratelimit.flush()
        return app

    @property
    def control(self):
        return self.backend.app["decision_control"]

    @property
    def backend_config(self):
        return self.app["config"]["backends"]["basal-20-max"]

    async def post(self, body=None, token="mytoken", path=PATH):
        return await self.client.post(path, json=payload() if body is None else body,
            headers={"Authorization": "Bearer " + token})

    async def test_all_types_order_typed_values_and_single_billing(self):
        self.backend_config["token"] = "private-key"
        response = await self.post()
        self.assertEqual(response.status, 200, await response.text())
        data = await response.json()
        self.assertEqual(data["model"], "basal-20-max")
        self.assertEqual(data["answers"], [
            {"type": "predicate", "name": "damaged", "probability": 0.95},
            {"type": "choice", "name": None, "choice": True, "confidence": 0.9,
                "probabilities": [{"value": True, "probability": 0.9}, {"value": "true", "probability": 0.1}]},
            {"type": "score", "name": "severity", "score": 1.7, "confidence": 0.8,
                "probabilities": [{"label": "Low", "value": 0, "probability": 0.1},
                    {"label": "Medium", "value": 1, "probability": 0.1},
                    {"label": "High", "value": 2, "probability": 0.8}]}])
        self.assertIs(data["answers"][1]["choice"], True)
        self.assertEqual(data["usage"], {"input_tokens": 123, "output_tokens": 0, "total_tokens": 123,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0}})
        forwarded, headers = self.backend.app["decision_calls"][0]
        self.assertNotIn("model", forwarded)
        self.assertEqual(headers["Authorization"], "Bearer private-key")
        self.assertEqual(forwarded["state"], payload()["input"])
        self.assertEqual(forwarded["questions"]["q1"], {"type": "choice", "instructions": "Choose a value.",
            "option_keys": "hide", "criteria": {"o0": "true: Boolean", "o1": '"true"'}})
        self.assertEqual(forwarded["questions"]["q2"]["levels"], ["Low", "Medium", "High: Unusable"])
        self.assertEqual([p for p in self.backend.app["paths"] if p != "/health"], ["/v1/systemone"])
        self.assertEqual(await self.get_events(), [
            {"product": "basal-20-max/a5000/prompt", "quantity": 123},
            {"product": "basal-20-max/a5000/completion", "quantity": 0}])
        self.assertNotIn("X-AI-Generated", response.headers)

    async def test_string_choice_is_not_coerced_to_boolean(self):
        answer = self.control["response"]["answers"]["q1"]
        answer.update(choice="o1", probabilities={"o0": 0.1, "o1": 0.9})
        data = await (await self.post()).json()
        self.assertIs(type(data["answers"][1]["choice"]), str)
        self.assertEqual(data["answers"][1]["choice"], "true")

    async def test_user_messages_preserve_table_text_and_boundaries(self):
        table = "| Item | Amount |\n| Refund | 100 |"
        body = payload() | {"input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Invoice:"},
                {"type": "input_text", "text": table}]},
            {"type": "message", "role": "user", "content": "Check the amount."}]}
        response = await self.post(body)
        self.assertEqual(response.status, 200, await response.text())
        forwarded = self.backend.app["decision_calls"][0][0]
        self.assertEqual(forwarded["state"], "Invoice:\n" + table + "\n\nCheck the amount.")
        self.assertNotIn("media", forwarded)
        self.assertEqual(len(await self.get_events()), 2)

    async def test_images_files_audio_and_instruction_roles_are_rejected_before_inference(self):
        inputs = [
            [{"role": "system", "content": "Instruction"}],
            [{"role": "assistant", "content": "History"}],
            [{"type": "function_call", "role": "user", "content": "Call"}],
            [{"role": "user", "content": []}],
            [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}]}],
            [{"role": "user", "content": [{"type": "input_file", "file_id": "file-1"}]}],
            [{"role": "user", "content": [{"type": "input_audio", "data": "aGVsbG8="}]}],
            [{"role": "user", "content": [{"type": "input_text", "text": "x", "unknown": True}]}],
            [{"role": "user", "content": [{"type": "input_text", "text": None}]}]]
        for value in inputs:
            with self.subTest(value=value):
                response = await self.post(payload() | {"input": value})
                self.assertEqual(response.status, 400, await response.text())
                self.assertEqual((await response.json())["error"]["type"], "invalid_request_error")
        self.assertEqual(self.backend.app["decision_calls"], [])
        self.assertEqual(await self.get_events(), [])

    async def test_invalid_and_unsupported_requests_never_reach_backend(self):
        cases = [payload() | delta for delta in ({"input": []}, {"input": None}, {"stream": False},
            {"safety_identifier": "user"}, {"prompt_cache_key": "cache"}, {"questions": []}, {"questions": {}},
            {"temperature": 0})]
        for delta in ({"type": "multi"}, {"type": []}, {"instructions": None}, {"name": None},
                {"criteria": {}}, {"choices": [{"value": True}]},
                {"choices": [{"value": True}, {"value": True}]},
                {"choices": [{"value": 1}, {"value": "1"}]},
                {"choices": [{"value": "a", "description": None}, {"value": "b"}]}):
            body = payload()
            body["questions"][1].update(delta)
            cases.append(body)
        for delta in ({"levels": [{"label": "same"}] * 2}, {"levels": [{"label": False}, {"label": "b"}]}):
            body = payload()
            body["questions"][2].update(delta)
            cases.append(body)
        for body in cases:
            with self.subTest(body=body):
                response = await self.post(body)
                self.assertEqual(response.status, 400, await response.text())
                self.assertEqual((await response.json())["error"]["type"], "invalid_request_error")
        self.assertEqual(self.backend.app["decision_calls"], [])
        self.assertEqual(await self.get_events(), [])

    async def test_auth_routing_limits_and_rate_limit_are_shared(self):
        response = await self.post(token="wrong")
        self.assertEqual(response.status, 401)
        self.assertEqual((await response.json())["error"]["type"], "authentication_error")
        for model in ("missing", "mymodel", "myembedding"):
            self.assertEqual((await self.post(payload() | {"model": model})).status, 404)
        self.backend_config["decision_limits"] = {"max_questions": 2}
        self.assertEqual((await self.post()).status, 400)
        self.backend_config["decision_limits"] = {"max_branches": 2}
        self.assertEqual((await self.post()).status, 400)
        self.backend_config.pop("decision_limits")
        self.backend_config["rate_limit"] = {"rpm": 1}
        self.assertEqual((await self.post()).status, 200)
        self.assertEqual((await self.post({"model": "basal-20-max", "state": "test",
            "questions": {"a": {"type": "noul"}}}, path="/v1/systemone")).status, 429)
        self.assertEqual(len(self.backend.app["decision_calls"]), 1)
        self.assertEqual(len(await self.get_events()), 2)

    async def test_incomplete_or_inconsistent_backend_results_are_not_billed(self):
        cases = []
        for key, delta in (("q0", {"noul": 0.8}), ("q0", {"probabilities": {"true": 0.95}}),
                ("q1", {"probabilities": {"o0": 0.9, "other": 0.1}}),
                ("q1", {"probabilities": {"o0": 0.9, "o1": 0.9}}),
                ("q2", {"score": 1.2}), ("q2", {"probabilities": {"0": 0.1, "1": 0.9}})):
            data = backend_response()
            data["answers"][key].update(delta)
            cases.append(data)
        data = backend_response()
        data["usage"]["input_tokens"] = True
        cases.append(data)
        for data in cases:
            with self.subTest(data=data):
                self.control["response"] = data
                response = await self.post()
                self.assertEqual(response.status, 502, await response.text())
                self.assertEqual((await response.json())["error"]["code"], "invalid_backend_response")
        self.assertEqual(await self.get_events(), [])

    async def test_upstream_errors_and_timeouts_hide_private_data(self):
        for status, expected in ((401, 502), (403, 502), (422, 422), (429, 429), (503, 502)):
            self.control.update(status=status, raw=b'{"error":"private document"}')
            with self.subTest(status=status), self.assertLogs(self.app.logger, level="ERROR") as logs:
                response = await self.post()
            self.assertEqual(response.status, expected)
            self.assertIsInstance((await response.json())["error"], dict)
            self.assertNotIn("private document", await response.text())
            self.assertNotIn("private document", " ".join(logs.output))
        self.control.clear()
        self.control["delay"] = 0.1
        self.backend_config["timeout"] = 0.01
        response = await self.post()
        self.assertEqual(response.status, 504)
        self.assertEqual((await response.json())["error"]["type"], "server_error")
        self.assertEqual(await self.get_events(), [])

    async def test_native_routes_keep_native_format_and_chat_rejects_decision_models(self):
        self.control.clear()
        native = {"model": "basal-20-max", "state": "test", "questions": {"category": {"type": "choice"}}}
        for path in ("/v1/systemone",):
            response = await self.post(native, path=path)
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())["answers"]["category"]["choice"], "yes")
        before = len(await self.get_events())
        calls = len(self.backend.app["decision_calls"])
        model = native.pop("model")
        response = await self.post({"model": model,
            "messages": [{"role": "user", "content": json.dumps(native)}]}, path="/v1/chat/completions")
        self.assertEqual(response.status, 404)
        self.assertEqual((await response.json())["error"]["code"], "model_not_supported")
        self.assertEqual(len(self.backend.app["decision_calls"]), calls)
        self.assertEqual(len(await self.get_events()), before)

    async def test_old_native_payload_and_removed_routes_are_rejected(self):
        native = {"model": "basal-20-max", "state": "test", "questions": {"q": {"type": "noul"}}}
        response = await self.post(native)
        self.assertEqual(response.status, 400)
        self.assertEqual((await response.json())["error"]["type"], "invalid_request_error")
        for path in ("/experimental/v1/decisions", "/v1/basal"):
            response = await self.post(path=path)
            self.assertEqual(response.status, 404)
        self.assertEqual((await self.post(payload(), path="/v1/systemone")).status, 400)
        self.assertEqual(self.backend.app["decision_calls"], [])
        self.assertEqual(await self.get_events(), [])

    async def test_concurrent_requests_keep_their_own_names(self):
        started, release = asyncio.Event(), asyncio.Event()
        self.control.update(started=started, release=release)
        first = asyncio.create_task(self.post())
        await asyncio.wait_for(started.wait(), 2)
        started.clear()
        body = payload()
        body["questions"][0]["name"] = "different-name"
        second = asyncio.create_task(self.post(body))
        await asyncio.wait_for(started.wait(), 2)
        release.set()
        responses = await asyncio.gather(first, second)
        for response, name in zip(responses, ("damaged", "different-name")):
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())["answers"][0]["name"], name)
        self.assertEqual(len(await self.get_events()), 4)

    async def test_response_size_limit_and_billing_failure_do_not_return_success(self):
        self.backend_config["decision_limits"] = {"max_response_bytes": 16}
        self.assertEqual((await self.post()).status, 502)
        self.backend_config.pop("decision_limits")
        with mock.patch("llmproxy.billing.record", mock.AsyncMock(side_effect=RuntimeError("synthetic"))):
            self.assertEqual((await self.post()).status, 500)
        self.assertEqual(await self.get_events(), [])

    @unittest.skipUnless(has_decisions_sdk(), "optional: requires OpenAI Decisions SDK")
    async def test_official_sdk_strictly_validates_all_answers(self):
        import httpx2
        import openai
        async with openai.AsyncOpenAI(api_key="mytoken",
                base_url=str(self.server.make_url("/v1")), max_retries=0,
                _strict_response_validation=True, http_client=httpx2.AsyncClient(trust_env=False)) as client:
            data = await client.decisions.create(**payload())
            self.assertEqual(data.model, "basal-20-max")
            self.assertEqual(data.answers[0].probability, 0.95)
            self.assertIs(data.answers[1].choice, True)
            self.assertEqual(data.answers[1].probabilities[1].value, "true")
            self.assertEqual(data.answers[2].score, 1.7)
            self.assertEqual(data.usage.total_tokens, 123)
            with self.assertRaises(openai.BadRequestError):
                await client.decisions.create(**(payload() | {"input": []}))
            messages = payload() | {"input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Amount,Total\nRefund,100"}]}]}
            result = await client.decisions.create(**messages)
            self.assertEqual(result.answers[0].probability, 0.95)
            self.assertEqual(self.backend.app["decision_calls"][-1][0]["state"], "Amount,Total\nRefund,100")
        self.assertEqual(len(await self.get_events()), 4)
