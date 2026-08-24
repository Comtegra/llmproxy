"""EU AI Act Art. 50(2) provenance marking: unit tests for the module and
route tests asserting which endpoints are marked (body+header / header-only /
not at all) and that billing is unaffected either way."""

import decimal
import json
import unittest
import uuid

import aiohttp

from llmproxy import provenance

from tests.test_proxy import LLMProxyAppTestCase


AUTH = {"Authorization": "Bearer mytoken"}


class TestProvenanceMark(unittest.TestCase):
    def test_build_fields(self):
        rid = uuid.uuid4()
        p = provenance.build({}, "mymodel", rid)
        self.assertIs(p["ai_generated"], True)
        self.assertEqual(p["digital_source_type"],
            "http://cv.iptc.org/newscodes/digitalsourcetype/"
            "trainedAlgorithmicMedia")
        self.assertEqual(p["legal_basis"],
            "Regulation (EU) 2024/1689, Article 50(2)")
        self.assertEqual(p["generator"], "comtegra-llmproxy")
        # Never pin the exact version: CI runs from a bare checkout and gets
        # the 0.0.0-dev fallback; installed environments get pyproject's.
        self.assertIsInstance(p["generator_version"], str)
        self.assertTrue(p["generator_version"])
        self.assertEqual(p["model_id"], "mymodel")
        self.assertEqual(p["request_id"], str(rid))
        self.assertRegex(p["timestamp"],
            r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")

    def test_build_generator_override(self):
        p = provenance.build({"provenance": {"generator": "acme"}}, "m", "r")
        self.assertEqual(p["generator"], "acme")

    def test_mark_disabled_returns_original_bytes(self):
        body = b'{"usage": {"prompt_tokens": 1}}'
        data = json.loads(body, parse_float=decimal.Decimal)
        out, hdrs = provenance.mark({"provenance": {"enabled": False}},
            body, data, "m", "r")
        self.assertIs(out, body)
        self.assertEqual(hdrs, {})
        self.assertNotIn(provenance.FIELD, data)

    def test_mark_injects_field_and_header(self):
        body = b'{"id":"x","usage":{"prompt_tokens":1}}'
        data = json.loads(body, parse_float=decimal.Decimal)
        out, hdrs = provenance.mark({}, body, data, "mymodel", "rid-1")
        self.assertEqual(hdrs, {provenance.HEADER: "true"})
        parsed = json.loads(out)
        self.assertEqual(parsed["id"], "x")
        self.assertEqual(parsed["usage"], {"prompt_tokens": 1})
        self.assertIs(parsed[provenance.FIELD]["ai_generated"], True)
        self.assertEqual(parsed[provenance.FIELD]["model_id"], "mymodel")
        self.assertEqual(parsed[provenance.FIELD]["request_id"], "rid-1")

    def test_mark_preserves_decimal_representation(self):
        # parse_float=Decimal in the handlers must not inflate float literals
        # (0.1 must not become 0.1000000000000000055511151231257827).
        body = b'{"logprob":0.1,"usage":{}}'
        data = json.loads(body, parse_float=decimal.Decimal)
        out, _ = provenance.mark({}, body, data, "m", "r")
        self.assertIn(b'"logprob":0.1,', out)

    def test_mark_preserves_non_ascii_raw(self):
        body = '{"c":"zażółć 😀","usage":{}}'.encode()
        data = json.loads(body, parse_float=decimal.Decimal)
        out, _ = provenance.mark({}, body, data, "m", "r")
        self.assertIn("zażółć 😀".encode(), out)

    def test_mark_survives_lone_surrogate(self):
        # A generation truncated mid-emoji by max_tokens: valid JSON, but not
        # UTF-8-encodable. mark() must fall back to ASCII escapes, never raise
        # (an exception here turns a billable 200 into an unbilled 500).
        body = b'{"a": "\\ud83d", "usage": {}}'
        data = json.loads(body, parse_float=decimal.Decimal)
        out, hdrs = provenance.mark({}, body, data, "m", "r")
        self.assertEqual(hdrs, {provenance.HEADER: "true"})
        parsed = json.loads(out)
        self.assertEqual(parsed["a"], "\ud83d")
        self.assertIn(provenance.FIELD, parsed)

    def test_mark_overflow_falls_back_to_unmarked_body(self):
        # A finite literal beyond double range (float(Decimal) -> inf) would
        # re-serialize as bare Infinity: invalid JSON. mark() must forward
        # the upstream bytes and keep only the header layer.
        body = b'{"x": 1e400, "usage": {}}'
        data = json.loads(body, parse_float=decimal.Decimal)
        out, hdrs = provenance.mark({}, body, data, "m", "r")
        self.assertIs(out, body)
        self.assertEqual(hdrs, {provenance.HEADER: "true"})
        self.assertNotIn(provenance.FIELD, data)

    def test_mark_replaces_upstream_provenance_key(self):
        # Documented behavior: the proxy's marking is authoritative; an
        # upstream top-level "provenance" key is replaced wholesale.
        body = b'{"provenance": {"foo": 1}, "usage": {}}'
        data = json.loads(body, parse_float=decimal.Decimal)
        out, _ = provenance.mark({}, body, data, "m", "r")
        parsed = json.loads(out)
        self.assertIs(parsed[provenance.FIELD]["ai_generated"], True)
        self.assertNotIn("foo", parsed[provenance.FIELD])

    def test_default_decimal_only(self):
        self.assertEqual(provenance._default(decimal.Decimal("0.5")), 0.5)
        with self.assertRaises(TypeError):
            provenance._default(uuid.uuid4())

    def test_headers_toggle(self):
        self.assertEqual(provenance.headers({}), {provenance.HEADER: "true"})
        self.assertEqual(
            provenance.headers({"provenance": {"enabled": False}}), {})


class TestProvenanceRoutes(LLMProxyAppTestCase):
    async def _marked_nonstream(self, path, body):
        req = self.client.request("POST", path, headers=AUTH, json=body)
        async with req as res:
            self.assertEqual(res.status, 200)
            self.assertEqual(res.headers["X-AI-Generated"], "true")
            self.assertIn("X-AI-Generated",
                res.headers.get("Access-Control-Expose-Headers", ""))
            request_id = res.headers["X-Request-ID"]
            data = await res.json()
        p = data["provenance"]
        self.assertIs(p["ai_generated"], True)
        self.assertEqual(p["model_id"], "mymodel")
        self.assertEqual(p["request_id"], request_id)
        return data

    async def test_chat_nonstream_marked(self):
        data = await self._marked_nonstream("/v1/chat/completions",
            {"model": "mymodel",
                "messages": [{"role": "user", "content": "hi"}]})
        # Upstream payload intact and billed as before.
        self.assertEqual(data["choices"][0]["message"]["content"],
            "you said: hi")
        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 1},
            {"product": "mymodel/none/completion", "quantity": 2},
        ])

    async def test_completions_legacy_marked(self):
        # Shared handler with /v1/chat/completions; the mock backend expects a
        # chat-shaped body on the legacy path too, which is fine -- the proxy
        # forwards path-preserving without interpreting the payload shape.
        await self._marked_nonstream("/v1/completions",
            {"model": "mymodel",
                "messages": [{"role": "user", "content": "hi"}]})

    async def test_messages_nonstream_marked(self):
        data = await self._marked_nonstream("/v1/messages",
            {"model": "mymodel",
                "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(data["content"][0]["text"], "hi")

    async def test_responses_nonstream_marked(self):
        data = await self._marked_nonstream("/v1/responses",
            {"model": "mymodel", "input": "hi"})
        self.assertEqual(data["object"], "response")

    async def test_streams_header_only(self):
        # Stream BODIES must stay byte-for-byte (no injected field, terminal
        # events intact); the marking rides on the headers.
        cases = [
            ("/v1/chat/completions",
                {"model": "mymodel", "stream": True,
                    "messages": [{"role": "user", "content": "hi"}]},
                "data: [DONE]"),
            ("/v1/messages",
                {"model": "mymodel", "stream": True,
                    "messages": [{"role": "user", "content": "hi"}]},
                "message_stop"),
            ("/v1/responses",
                {"model": "mymodel", "stream": True, "input": "hi"},
                "response.completed"),
        ]
        for path, body, terminal in cases:
            with self.subTest(path=path):
                req = self.client.request("POST", path, headers=AUTH,
                    json=body)
                async with req as res:
                    self.assertEqual(res.status, 200)
                    self.assertEqual(res.headers["X-AI-Generated"], "true")
                    self.assertIn("X-AI-Generated",
                        res.headers.get("Access-Control-Expose-Headers", ""))
                    text = await res.text()
                self.assertIn(terminal, text)
                self.assertNotIn("provenance", text)
        # 2 billed rows per stream; billing must be unaffected by the marking.
        self.assertEqual(len(await self.get_events(expect=6)), 6)

    async def test_embeddings_not_marked(self):
        req = self.client.request("POST", "/v1/embeddings", headers=AUTH,
            json={"model": "mymodel", "input": "hi"})
        async with req as res:
            self.assertEqual(res.status, 200)
            self.assertNotIn("X-AI-Generated", res.headers)
            data = await res.json()
        # Vectors are not "synthetic content" under Art. 50(2): body forwarded
        # byte-for-byte, no marking at all.
        self.assertNotIn("provenance", data)
        self.assertEqual(data["data"][0]["embedding"], [0.1, 0.2, 0.3])

    async def test_audio_header_only(self):
        form = aiohttp.FormData()
        form.add_field("model", "mymodel")
        form.add_field("file", b"RIFFfake-audio", filename="a.wav",
            content_type="audio/wav")
        req = self.client.request("POST", "/v1/audio/transcriptions",
            headers=AUTH, data=form)
        async with req as res:
            self.assertEqual(res.status, 200)
            self.assertEqual(res.headers["X-AI-Generated"], "true")
            data = await res.json()
        self.assertNotIn("provenance", data)
        self.assertEqual(data["duration"], 12.5)

    async def test_kill_switch_disables_marking(self):
        self.app["config"]["provenance"] = {"enabled": False}

        body = {"model": "mymodel",
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers=AUTH, json=body)
        async with req as res:
            self.assertEqual(res.status, 200)
            self.assertNotIn("X-AI-Generated", res.headers)
            data = await res.json()
        self.assertNotIn("provenance", data)
        self.assertEqual(data["choices"][0]["message"]["content"],
            "you said: hi")

        req = self.client.request("POST", "/v1/chat/completions",
            headers=AUTH, json={**body, "stream": True})
        async with req as res:
            self.assertEqual(res.status, 200)
            self.assertNotIn("X-AI-Generated", res.headers)
            await res.text()

        # Billing must keep working with the switch off.
        self.assertEqual(len(await self.get_events(expect=4)), 4)

    async def test_generator_override(self):
        self.app["config"]["provenance"] = {"generator": "acme"}
        data = await self._marked_nonstream("/v1/chat/completions",
            {"model": "mymodel",
                "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(data["provenance"]["generator"], "acme")

    async def test_errors_not_marked(self):
        # 4xx forwarded verbatim, 5xx masked to 502 -- neither is generated
        # content, neither may carry the marking.
        for err, expected in ((400, 400), (500, 502)):
            with self.subTest(err=err):
                body = {"model": "mymodel", "_trigger_error": err,
                    "messages": [{"role": "user", "content": "hi"}]}
                req = self.client.request("POST", "/v1/chat/completions",
                    headers=AUTH, json=body)
                async with req as res:
                    self.assertEqual(res.status, expected)
                    self.assertNotIn("X-AI-Generated", res.headers)
                    text = await res.text()
                self.assertNotIn("provenance", text)
