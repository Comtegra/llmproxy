import unittest

from llmproxy.messages import _MessagesStreamUsage, _input_tokens

from tests.test_proxy import LLMProxyAppTestCase


class TestMessagesUsage(unittest.TestCase):
    def test_input_from_start_output_from_last_delta(self):
        acc = _MessagesStreamUsage()
        for c in [
            b'event: message_start\ndata: {"type":"message_start",'
            b'"message":{"usage":{"input_tokens":10,'
            b'"cache_read_input_tokens":2,"output_tokens":1}}}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta",'
            b'"usage":{"output_tokens":4}}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta",'
            b'"usage":{"output_tokens":7}}\n\n',
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]:
            acc(c)
        # input(+cache) = 12 prompt; output = LAST delta 7 (not 4+7 summed)
        self.assertEqual(acc.usage(),
            {"prompt_tokens": 12, "completion_tokens": 7})

    def test_input_tokens_folds_cache(self):
        self.assertEqual(_input_tokens(
            {"input_tokens": 5, "cache_creation_input_tokens": 3,
             "cache_read_input_tokens": 2}), 10)
        self.assertEqual(_input_tokens({"input_tokens": 5}), 5)

    def test_input_tokens_requires_input_tokens(self):
        # Absent input_tokens -> fail loud (not silent 0); a present 0 is legit
        # (fully cache-read turn).
        with self.assertRaises(KeyError):
            _input_tokens({"output_tokens": 5})
        self.assertEqual(
            _input_tokens({"input_tokens": 0, "cache_read_input_tokens": 50}),
            50)

    def test_stream_start_without_input_tokens_fails_loud(self):
        acc = _MessagesStreamUsage()
        acc(b'event: message_start\ndata: {"type":"message_start",'
            b'"message":{"usage":{"output_tokens":1}}}\n\n')  # no input_tokens
        acc(b'event: message_delta\ndata: {"type":"message_delta",'
            b'"usage":{"output_tokens":5}}\n\n')
        with self.assertRaises(KeyError):
            acc.usage()

    def test_missing_events_fail_loud(self):
        # A truncated stream (message_start but no final message_delta) must
        # raise, never silently bill completion=0.
        acc = _MessagesStreamUsage()
        acc(b'event: message_start\ndata: {"type":"message_start",'
            b'"message":{"usage":{"input_tokens":10,"output_tokens":1}}}\n\n')
        with self.assertRaises(ValueError):
            acc.usage()


class TestMessagesRoute(LLMProxyAppTestCase):
    async def test_streaming_billing(self):
        body = {"model": "mymodel", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/messages",
            headers={"Authorization": "Bearer mytoken"}, json=body)
        async with req as res:
            self.assertEqual(res.status, 200)
            text = await res.text()
            self.assertIn("message_start", text)  # backend stream forwarded
        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 3},
            {"product": "mymodel/none/completion", "quantity": 5},
        ])

    async def test_nonstream_billing(self):
        body = {"model": "mymodel",
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/messages",
            headers={"Authorization": "Bearer mytoken"}, json=body)
        async with req as res:
            self.assertEqual(res.status, 200)
        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 3},
            {"product": "mymodel/none/completion", "quantity": 5},
        ])

    async def test_missing_usage_fails_loud(self):
        # A 200 with no usage object must 500 (fail loud), never bill 0.
        body = {"model": "mymodel", "_no_usage": True,
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/messages",
            headers={"Authorization": "Bearer mytoken"}, json=body)
        async with req as res:
            self.assertEqual(res.status, 500)
        self.assertListEqual(await self.get_events(), [])

    async def test_missing_input_tokens_fails_loud(self):
        # usage present but no input_tokens must 500, never bill prompt=0.
        body = {"model": "mymodel", "_no_input": True,
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/messages",
            headers={"Authorization": "Bearer mytoken"}, json=body)
        async with req as res:
            self.assertEqual(res.status, 500)
        self.assertListEqual(await self.get_events(), [])
