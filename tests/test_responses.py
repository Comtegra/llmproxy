import unittest

from llmproxy.responses import (
    StatefulNotSupported,
    _ResponsesStreamUsage,
    force_stateless,
    to_billing_tokens,
    usage_from_event,
)
from llmproxy.streaming import parse_sse_event

from tests.test_proxy import LLMProxyAppTestCase


COMPLETED_EVENT = (
    b"event: response.completed\n"
    b'data: {"type":"response.completed","sequence_number":7,'
    b'"response":{"id":"resp_abc","object":"response","status":"completed",'
    b'"model":"qwen3","output":[{"type":"message","role":"assistant",'
    b'"content":[{"type":"output_text","text":"Pekin."}]}],'
    b'"usage":{"input_tokens":34,"input_tokens_details":{"cached_tokens":8},'
    b'"output_tokens":21,"output_tokens_details":{"reasoning_tokens":13},'
    b'"total_tokens":55}}}\n\n'
)

DELTA_EVENT = (
    b"event: response.output_text.delta\n"
    b'data: {"type":"response.output_text.delta","sequence_number":5,'
    b'"delta":"Pe"}\n\n'
)

CREATED_EVENT = (
    b"event: response.created\n"
    b'data: {"type":"response.created","sequence_number":0,'
    b'"response":{"id":"resp_abc","status":"in_progress"}}\n\n'
)


class TestParseSseEvent(unittest.TestCase):
    def test_extracts_json_from_data_line(self):
        self.assertEqual(parse_sse_event(COMPLETED_EVENT)["type"],
            "response.completed")

    def test_strips_event_line(self):
        self.assertEqual(parse_sse_event(DELTA_EVENT)["delta"], "Pe")

    def test_done_sentinel_is_none(self):
        self.assertIsNone(parse_sse_event(b"data: [DONE]\n\n"))

    def test_comment_keepalive_is_none(self):
        self.assertIsNone(parse_sse_event(b": ping\n\n"))
        self.assertIsNone(parse_sse_event(b"event: response.created\n\n"))

    def test_malformed_json_is_none(self):
        self.assertIsNone(parse_sse_event(b"data: {not json\n\n"))

    def test_multiline_data_joined(self):
        self.assertEqual(parse_sse_event(b'data: {"a":\ndata: 1}\n\n')["a"], 1)


class TestUsageFromEvent(unittest.TestCase):
    def test_completed_yields_usage(self):
        u = usage_from_event(parse_sse_event(COMPLETED_EVENT))
        self.assertEqual(u["input_tokens"], 34)
        self.assertEqual(u["output_tokens"], 21)

    def test_incomplete_also_billed(self):
        ev = COMPLETED_EVENT.replace(
            b"response.completed", b"response.incomplete")
        self.assertEqual(
            usage_from_event(parse_sse_event(ev))["output_tokens"], 21)

    def test_non_terminal_is_none(self):
        self.assertIsNone(usage_from_event(parse_sse_event(DELTA_EVENT)))
        self.assertIsNone(usage_from_event(parse_sse_event(CREATED_EVENT)))

    def test_none_input(self):
        self.assertIsNone(usage_from_event(None))

    def test_failed_event_billed(self):
        # response.failed carries usage for a run that generated output.
        ev = COMPLETED_EVENT.replace(b"response.completed", b"response.failed")
        self.assertEqual(
            usage_from_event(parse_sse_event(ev))["output_tokens"], 21)


class TestToBillingTokens(unittest.TestCase):
    def test_maps_to_prompt_completion(self):
        b = to_billing_tokens(usage_from_event(parse_sse_event(COMPLETED_EVENT)))
        self.assertEqual(b["prompt_tokens"], 34)
        self.assertEqual(b["completion_tokens"], 21)

    def test_reasoning_not_double_counted(self):
        b = to_billing_tokens(usage_from_event(parse_sse_event(COMPLETED_EVENT)))
        self.assertEqual(b["completion_tokens"], 21)   # NOT 21+13 reasoning
        self.assertEqual(b["reasoning_tokens"], 13)
        self.assertEqual(b["cached_tokens"], 8)

    def test_missing_output_tokens_fails_loud(self):
        # A present usage lacking output_tokens must raise, not bill 0.
        with self.assertRaises(KeyError):
            to_billing_tokens({"input_tokens": 5})

    def test_present_zero_is_legitimate(self):
        b = to_billing_tokens({"input_tokens": 0, "output_tokens": 0})
        self.assertEqual((b["prompt_tokens"], b["completion_tokens"]), (0, 0))


class TestForceStateless(unittest.TestCase):
    def test_plain_downgraded_to_store_false(self):
        body = {"model": "qwen3", "store": True}
        force_stateless(body)
        self.assertFalse(body["store"])

    def test_previous_response_id_fails_loud(self):
        with self.assertRaises(StatefulNotSupported):
            force_stateless({"model": "qwen3", "previous_response_id": "resp_x"})

    def test_conversation_fails_loud(self):
        with self.assertRaises(StatefulNotSupported):
            force_stateless({"model": "qwen3", "conversation": "conv_x"})

    def test_background_fails_loud(self):
        with self.assertRaises(StatefulNotSupported):
            force_stateless({"model": "qwen3", "background": True})

    def test_is_value_error(self):
        self.assertTrue(issubclass(StatefulNotSupported, ValueError))


class TestResponsesStreamUsage(unittest.TestCase):
    def test_usage_from_terminal_event(self):
        acc = _ResponsesStreamUsage()
        for c in [CREATED_EVENT, DELTA_EVENT, COMPLETED_EVENT]:
            acc(c)
        b = to_billing_tokens(acc.usage())
        self.assertEqual((b["prompt_tokens"], b["completion_tokens"]), (34, 21))

    def test_missing_usage_raises(self):
        acc = _ResponsesStreamUsage()
        acc(CREATED_EVENT)
        with self.assertRaises(ValueError):
            acc.usage()


class TestResponsesRoute(LLMProxyAppTestCase):
    async def test_streaming_billing(self):
        body = {"model": "mymodel", "stream": True,
            "input": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/responses",
            headers={"Authorization": "Bearer mytoken"}, json=body)
        async with req as res:
            self.assertEqual(res.status, 200)
            text = await res.text()
            self.assertIn("response.completed", text)  # backend stream forwarded
        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 7},
            {"product": "mymodel/none/completion", "quantity": 9},
        ])

    async def test_nonstream_billing(self):
        body = {"model": "mymodel",
            "input": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/responses",
            headers={"Authorization": "Bearer mytoken"}, json=body)
        async with req as res:
            self.assertEqual(res.status, 200)
        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 7},
            {"product": "mymodel/none/completion", "quantity": 9},
        ])

    async def test_stateful_rejected_400_unbilled(self):
        # previous_response_id -> 400 fail-loud BEFORE any backend call/billing.
        body = {"model": "mymodel", "previous_response_id": "resp_x",
            "input": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/responses",
            headers={"Authorization": "Bearer mytoken"}, json=body)
        async with req as res:
            self.assertEqual(res.status, 400)
            data = await res.json()
            self.assertEqual(data["error"]["code"], "stateful_not_supported")
        self.assertListEqual(await self.get_events(), [])

    async def test_failed_terminal_billed(self):
        # A run that fails AFTER generating output still bills its usage.
        body = {"model": "mymodel", "stream": True,
            "_terminal": "response.failed",
            "input": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/responses",
            headers={"Authorization": "Bearer mytoken"}, json=body)
        async with req as res:
            self.assertEqual(res.status, 200)
        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 7},
            {"product": "mymodel/none/completion", "quantity": 9},
        ])
