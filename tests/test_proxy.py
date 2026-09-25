import asyncio
import hashlib
import importlib
import importlib.resources
import json
import os
import sqlite3
import tempfile
import time
import unittest
import warnings

import aiohttp
import aiohttp.test_utils
import prometheus_client

from llmproxy import auth, config, ratelimit
from llmproxy.app import create_app, reload_config
from llmproxy.db import get_db

from . import mockbackend


class LLMProxyAppTestCase(aiohttp.test_utils.AioHTTPTestCase):
    async def asyncSetUp(self):
        # Don't care about type checkers
        warnings.simplefilter("ignore", category=aiohttp.web.NotAppKeyWarning)

        # The auth cache is process-global and every test builds a fresh
        # database, so it has to be dropped between cases or one test's key
        # lookup satisfies the next test's request.
        auth.flush_cache()

        self.backend = aiohttp.test_utils.TestServer(mockbackend.create_app())
        await self.backend.start_server()

        await super().asyncSetUp()

    async def asyncTearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)
        await super().asyncTearDown()

    async def get_application(self):
        self.db_fd, self.db_path = tempfile.mkstemp()

        url = "http://%s:%d" % (self.backend.host, self.backend.port)
        app = await create_app({
            "timeout_connect": 1,
            "timeout_read": 1,
            "max_json_body": 1024 * 1024,
            "db": {"uri": "sqlite://%s" % self.db_path},
            # All backends point at the one mock server; each non-chat type
            # gets its own alias so the endpoint/type check lets it through.
            "backends": {
                "mymodel": {
                    "url": url,
                    "token": "mybackendtoken",
                    "device": "none",
                    "type": "chat",
                    "max_model_len": 12345,
                },
                "nolimit": {
                    "url": url,
                    "token": "secret-backend-token",
                    "device": "none",
                    "type": "chat",
                    "model": "mymodel",
                    "verify_ssl": False,
                },
                "slowok": {
                    "url": url,
                    "token": "mybackendtoken",
                    "device": "none",
                    "type": "chat",
                    "model": "mymodel",
                    "timeout": 5,
                },
                "myembedding": {
                    "url": url,
                    "token": "mybackendtoken",
                    "device": "none",
                    "type": "embedding",
                    "model": "mymodel",
                },
                "mywhisper": {
                    "url": url,
                    "token": "mybackendtoken",
                    "device": "none",
                    "type": "transcription",
                },
                "mymarker": {
                    "url": url,
                    "token": "mybackendtoken",
                    "device": "none",
                    "type": "conversion",
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

    async def get_events(self, expect=None, timeout=5.0):
        """Read the billed events.

        A STREAMING handler bills after ``write_eof()``, so the client can have
        the complete response in hand while the server is still running its
        billing tail -- reading immediately races the write. Pass ``expect=N``
        to poll until N rows land (or the timeout elapses, so a genuine billing
        failure still fails the assertion rather than hanging).

        Non-streaming handlers bill before responding, and the "must not be
        billed" assertions want the current state, so the default does not
        poll."""
        deadline = time.monotonic() + timeout

        while True:
            db = await get_db(self.app["config"]["db"]["uri"])
            cur = await db.db.execute(
                "SELECT product, quantity FROM event_oneoff")
            rows = await cur.fetchall()
            await db.close()

            if expect is None or len(rows) >= expect \
                    or time.monotonic() >= deadline:
                return rows

            await asyncio.sleep(0.01)


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
        # model_repo carries the backend's optional real model id (repo name);
        # backends without it expose null.
        self.assertEqual(models["nolimit"]["model_repo"], "mymodel")
        self.assertEqual(models["slowok"]["model_repo"], "mymodel")
        self.assertIsNone(models["mymodel"]["model_repo"])
        # type tells clients which endpoints each model is served on.
        self.assertEqual({name: m["type"] for name, m in models.items()}, {
            "mymodel": "chat", "nolimit": "chat", "slowok": "chat",
            "myembedding": "embedding", "mywhisper": "transcription",
            "mymarker": "conversion"})
        for model in models.values():
            self.assertLessEqual(
                set(model),
                {"id", "object", "created", "owned_by", "type", "device",
                    "model_repo", "max_model_len"},
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

    async def test_post_with_query_string_succeeds(self):
        # Claude Code sends every request as POST /v1/messages?beta=true.
        # Joining the raw rel_url drags the query into the backend path
        # ("?" -> %3F), which 404s; the forward must use the path alone.
        cases = [
            ("/v1/messages?beta=true",
                {"model": "mymodel", "max_tokens": 4,
                    "messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/chat/completions?foo=1",
                {"model": "mymodel",
                    "messages": [{"role": "user", "content": "hi"}]}),
        ]
        for path, body in cases:
            with self.subTest(path=path):
                req = self.client.request("POST", path,
                    headers={"Authorization": "Bearer mytoken"}, json=body)

                async with req as res:
                    self.assertEqual(res.status, 200)
                    await res.read()

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

        self.assertListEqual(await self.get_events(expect=2), [
            {"product": "mymodel/none/prompt", "quantity": 1},
            {"product": "mymodel/none/completion", "quantity": 2},
        ])

    async def test_embeddings_billing(self):
        body = {"model": "myembedding", "input": "hello"}
        req = self.client.request("POST", "/v1/embeddings",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)
            data = await res.json()

        self.assertEqual(data["usage"]["prompt_tokens"], 7)
        self.assertListEqual(await self.get_events(), [
            {"product": "myembedding/none/embedding", "quantity": 7},
        ])

    async def test_audio_transcription_billing(self):
        form = aiohttp.FormData()
        form.add_field("model", "mywhisper")
        form.add_field("file", b"RIFFfake-audio", filename="a.wav",
            content_type="audio/wav")
        req = self.client.request("POST", "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer mytoken"}, data=form)

        async with req as res:
            self.assertEqual(res.status, 200)
            data = await res.json()

        # Billed per second of audio; fractional durations must survive.
        self.assertEqual(data["duration"], 12.5)
        self.assertListEqual(await self.get_events(), [
            {"product": "mywhisper/none/transcription", "quantity": 12.5},
        ])

    async def test_file_convert_billing(self):
        form = aiohttp.FormData()
        form.add_field("model", "mymarker")
        form.add_field("_pages", "5")
        form.add_field("file", b"%PDF-1.4 fake", filename="doc.pdf",
            content_type="application/pdf")
        req = self.client.request("POST", "/v1/files/convert",
            headers={"Authorization": "Bearer mytoken"}, data=form)

        async with req as res:
            self.assertEqual(res.status, 200)
            self.assertIn("X-Request-ID", res.headers)
            data = await res.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["output"], "# Converted\n\nhello")
        self.assertEqual(data["page_count"], 5)
        # Billed per page; quantity = page_count).
        self.assertListEqual(await self.get_events(), [
            {"product": "mymarker/none/conversion", "quantity": 5},
        ])

    async def test_file_convert_missing_page_count_fails_loud(self):
        form = aiohttp.FormData()
        form.add_field("model", "mymarker")
        form.add_field("_omit_page_count", "1")
        form.add_field("file", b"%PDF-1.4 fake", filename="doc.pdf",
            content_type="application/pdf")
        req = self.client.request("POST", "/v1/files/convert",
            headers={"Authorization": "Bearer mytoken"}, data=form)

        async with req as res:
            self.assertEqual(res.status, 502)

        self.assertListEqual(await self.get_events(), [])

    async def test_file_convert_failure_not_billed(self):
        form = aiohttp.FormData()
        form.add_field("model", "mymarker")
        form.add_field("_fail", "1")
        form.add_field("file", b"%PDF-1.4 fake", filename="doc.pdf",
            content_type="application/pdf")
        req = self.client.request("POST", "/v1/files/convert",
            headers={"Authorization": "Bearer mytoken"}, data=form)

        async with req as res:
            self.assertEqual(res.status, 422)
            data = await res.json()

        self.assertFalse(data.get("success", True))
        self.assertListEqual(await self.get_events(), [])

    async def test_file_convert_requires_file(self):
        # Plain string-only FormData is urlencoded (415); include a file field
        # under the wrong name so we still hit multipart + the handler guard.
        form = aiohttp.FormData()
        form.add_field("model", "mymarker")
        form.add_field("not_file", b"%PDF-1.4", filename="doc.pdf",
            content_type="application/pdf")
        req = self.client.request("POST", "/v1/files/convert",
            headers={"Authorization": "Bearer mytoken"}, data=form)

        async with req as res:
            self.assertEqual(res.status, 422)

        self.assertListEqual(await self.get_events(), [])

    async def test_streaming_usage_in_trailing_chunk(self):
        # Realistic vLLM: usage arrives in a separate trailing chunk, not the
        # content chunk. The proxy must keep the last non-[DONE] chunk.
        body = {"model": "mymodel", "stream": True,
            "_stream_mode": "split_usage",
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)
            text = await res.text()
            self.assertIn("data: [DONE]", text)

        self.assertListEqual(await self.get_events(expect=2), [
            {"product": "mymodel/none/prompt", "quantity": 1},
            {"product": "mymodel/none/completion", "quantity": 2},
        ])

    async def test_per_backend_timeout_allows_slow_backend(self):
        # Global sock_read=1 would 504 a 2s backend; a backend with a long
        # per-backend timeout waits and still bills.
        body = {"model": "slowok", "_trigger_error": "slow",
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)

        self.assertListEqual(await self.get_events(), [
            {"product": "slowok/none/prompt", "quantity": 1},
            {"product": "slowok/none/completion", "quantity": 2},
        ])

    async def test_large_audio_upload_accepted(self):
        # A >1 MiB upload must not be rejected: aiohttp's default client_max_size
        # is 1 MiB, and the app raises it so audio works.
        big = b"\x00" * (2 * 1024 * 1024)
        form = aiohttp.FormData()
        form.add_field("model", "mywhisper")
        form.add_field("file", big, filename="a.wav",
            content_type="audio/wav")
        req = self.client.request("POST", "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer mytoken"}, data=form)

        async with req as res:
            self.assertEqual(res.status, 200)

        self.assertListEqual(await self.get_events(), [
            {"product": "mywhisper/none/transcription", "quantity": 12.5},
        ])

    async def test_unbillable_duration_fails_loud(self):
        # Missing, zero, negative, NaN and Infinity durations are all
        # unbillable -> 502 + no billing (never a corrupt or zero billing row).
        cases = [("_omit_duration", "1"), ("_duration", "zero"),
            ("_duration", "neg"), ("_duration", "nan"), ("_duration", "inf"),
            ("_duration", "true")]
        for field, value in cases:
            with self.subTest(case=value):
                form = aiohttp.FormData()
                form.add_field("model", "mywhisper")
                form.add_field(field, value)
                form.add_field("file", b"RIFFfake", filename="a.wav",
                    content_type="audio/wav")
                req = self.client.request("POST", "/v1/audio/transcriptions",
                    headers={"Authorization": "Bearer mytoken"}, data=form)

                async with req as res:
                    self.assertEqual(res.status, 502)

                self.assertListEqual(await self.get_events(), [])

    async def test_oversized_json_body_rejected(self):
        # A JSON body over max_json_body is rejected with 413 before it is
        # buffered, protecting the text endpoints from a memory-exhaustion DoS.
        big = {"model": "mymodel",
            "messages": [{"role": "user", "content": "x" * (2 * 1024 * 1024)}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=big)

        async with req as res:
            self.assertEqual(res.status, 413)

        self.assertListEqual(await self.get_events(), [])

    async def test_oversized_chunked_body_rejected(self):
        # The cap must also hold for a chunked body (no Content-Length), which
        # would slip past a Content-Length-only check.
        async def gen():
            for _ in range(2 * 1024):  # ~2 MiB in 1 KiB chunks
                yield b"x" * 1024

        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken",
                "Content-Type": "application/json"}, data=gen())

        async with req as res:
            self.assertEqual(res.status, 413)

        self.assertListEqual(await self.get_events(), [])

    @staticmethod
    def _multipart_form():
        # A file field (filename set) forces real multipart/form-data encoding;
        # a form of plain string fields would be sent as urlencoded, which
        # proxy.request already rejects on a different path.
        form = aiohttp.FormData()
        form.add_field("model", "mymodel")
        form.add_field("payload", b"x" * 4096, filename="p.bin",
            content_type="application/octet-stream")
        return form

    async def test_multipart_rejected_on_text_endpoints(self):
        # multipart on a JSON endpoint is rejected with 415 in the middleware,
        # BEFORE proxy.request would call post() and buffer the whole body into
        # memory (aiohttp's post() reads each non-file field WHOLE before the
        # size cap is checked -> an authenticated client could OOM the single
        # worker). Without the guard these reach post()+the backend and 502;
        # the 415 (not 502) proves the body was never read and bills nothing.
        for path in ("/v1/chat/completions", "/v1/completions",
                "/v1/embeddings", "/v1/messages", "/v1/responses"):
            with self.subTest(path=path):
                req = self.client.request("POST", path,
                    headers={"Authorization": "Bearer mytoken"},
                    data=self._multipart_form())

                async with req as res:
                    self.assertEqual(res.status, 415)
                    self.assertIn("X-Request-ID", res.headers)

                self.assertListEqual(await self.get_events(), [])

    async def test_multipart_rejected_before_auth(self):
        # The guard lives in limit_request_body, ahead of the handler's auth,
        # so the DoS vector is dropped without even a DB auth lookup: an INVALID
        # token still yields 415, not 401 (which is what it would be if the
        # multipart body reached the handler's auth+post()).
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer badtoken"},
            data=self._multipart_form())

        async with req as res:
            self.assertEqual(res.status, 415)

        self.assertListEqual(await self.get_events(), [])

    async def test_chunked_nonstream_billed_via_nonstream_path(self):
        # Streaming is detected by Content-Type, not Transfer-Encoding: a
        # non-stream JSON body arriving chunked (buffering proxy / HTTP-2)
        # must still be billed via the non-stream path, not silently dropped.
        body = {"model": "mymodel", "_chunked_nonstream": True,
            "messages": [{"role": "user", "content": "hi"}]}
        req = self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

        async with req as res:
            self.assertEqual(res.status, 200)

        self.assertListEqual(await self.get_events(), [
            {"product": "mymodel/none/prompt", "quantity": 1},
            {"product": "mymodel/none/completion", "quantity": 2},
        ])


class TestModelRouting(LLMProxyAppTestCase):
    """A backend's ``type`` decides which endpoints may reach it."""

    AUTH = {"Authorization": "Bearer mytoken"}

    # One backend alias per type (see get_application).
    MODEL_BY_TYPE = {"chat": "mymodel", "embedding": "myembedding",
        "transcription": "mywhisper", "conversion": "mymarker"}

    # Every model-routed endpoint and the backend type it serves.
    ENDPOINT_TYPES = {
        "/v1/chat/completions": "chat",
        "/v1/completions": "chat",
        "/v1/messages": "chat",
        "/v1/responses": "chat",
        "/v1/embeddings": "embedding",
        "/v1/audio/transcriptions": "transcription",
        "/v1/files/convert": "conversion",
    }

    def _post(self, path, model):
        if path in ("/v1/audio/transcriptions", "/v1/files/convert"):
            form = aiohttp.FormData()
            form.add_field("model", model)
            form.add_field("file", b"payload", filename="f.bin",
                content_type="application/octet-stream")
            return self.client.request("POST", path, headers=self.AUTH,
                data=form)
        body = {"model": model, "max_tokens": 4, "input": "hi",
            "messages": [{"role": "user", "content": "hi"}]}
        return self.client.request("POST", path, headers=self.AUTH, json=body)

    async def test_model_of_another_type_rejected_before_backend(self):
        # Every endpoint x every other type. E.g. chat completions on an
        # embedding model used to be forwarded, answered with generated text
        # and billed as prompt/completion tokens the model has no price for.
        backend_paths = self.backend.app["paths"]
        for path, served in self.ENDPOINT_TYPES.items():
            for model_type, model in self.MODEL_BY_TYPE.items():
                if model_type == served:
                    continue
                with self.subTest(path=path, model=model):
                    seen = len(backend_paths)
                    async with self._post(path, model) as res:
                        self.assertEqual(res.status, 404)
                        self.assertIn("X-Request-ID", res.headers)
                        data = await res.json()

                    if path == "/v1/messages":
                        self.assertEqual(data["type"], "error")
                        self.assertEqual(data["error"]["type"],
                            "not_found_error")
                    else:
                        self.assertEqual(data["error"]["type"],
                            "invalid_request_error")
                        self.assertEqual(data["error"]["code"],
                            "model_not_supported")
                        self.assertEqual(data["error"]["param"], "model")
                    self.assertIn(model, data["error"]["message"])
                    self.assertIn(model_type, data["error"]["message"])
                    self.assertIn(path, data["error"]["message"])

                    self.assertEqual(backend_paths[seen:], [])
                    self.assertListEqual(await self.get_events(), [])

    async def test_type_checked_before_endpoint_body_validation(self):
        # The client is told the model is wrong for this endpoint, not what
        # the endpoint's own body validation would say about the payload.
        form = aiohttp.FormData()
        form.add_field("model", "mywhisper")
        form.add_field("not_file", b"%PDF-1.4", filename="doc.pdf",
            content_type="application/pdf")
        cases = [
            # else 400 stateful_not_supported (responses.force_stateless)
            ("/v1/responses", {"json": {"model": "myembedding", "input": "hi",
                "previous_response_id": "resp_1"}}),
            # else 422 "file is required" (files.prepare_marker_body)
            ("/v1/files/convert", {"data": form}),
        ]
        for path, kwargs in cases:
            with self.subTest(path=path):
                req = self.client.request("POST", path, headers=self.AUTH,
                    **kwargs)
                async with req as res:
                    self.assertEqual(res.status, 404)
                    data = await res.json()
                self.assertEqual(data["error"]["code"], "model_not_supported")

    async def test_unknown_model_is_404_json(self):
        # Was 401 "Incorrect model" (plain text), which clients read as a bad
        # API key (some keep retrying it).
        for path in ("/v1/chat/completions", "/v1/messages",
                "/v1/files/convert"):
            with self.subTest(path=path):
                async with self._post(path, "nosuchmodel") as res:
                    self.assertEqual(res.status, 404)
                    data = await res.json()

                if path == "/v1/messages":
                    self.assertEqual(data["error"]["type"], "not_found_error")
                else:
                    self.assertEqual(data["error"]["code"], "model_not_found")
                    self.assertEqual(data["error"]["param"], "model")
                self.assertIn("'nosuchmodel'", data["error"]["message"])

        self.assertListEqual(await self.get_events(), [])

    async def test_missing_model_is_404_json(self):
        req = self.client.request("POST", "/v1/chat/completions",
            headers=self.AUTH,
            json={"messages": [{"role": "user", "content": "hi"}]})
        async with req as res:
            self.assertEqual(res.status, 404)
            data = await res.json()

        self.assertEqual(data["error"]["code"], "model_not_found")
        self.assertEqual(data["error"]["message"],
            "The model None does not exist.")

    async def test_rejected_model_does_not_consume_rate_limit(self):
        # The type check runs before ratelimit.slot, so a rejected request
        # must not use up the caller's rpm budget.
        ratelimit.flush()
        self.addCleanup(ratelimit.flush)
        self.app["config"]["rate_limit"] = {"rpm": 1}

        async with self._post("/v1/chat/completions", "myembedding") as res:
            self.assertEqual(res.status, 404)
        async with self._post("/v1/chat/completions", "mymodel") as res:
            self.assertEqual(res.status, 200)


class TestMalformedBody(LLMProxyAppTestCase):
    """A request body the proxy cannot parse or route is the client's fault:
    400 in the client's error format, never a 500 and never forwarded."""

    AUTH = {"Authorization": "Bearer mytoken"}

    # Upload endpoints parse JSON bodies too (only multipart is route-limited).
    PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/messages",
        "/v1/responses", "/v1/embeddings", "/v1/audio/transcriptions",
        "/v1/files/convert")
    UPLOAD_PATHS = ("/v1/audio/transcriptions", "/v1/files/convert")

    def _raw(self, path, body, content_type="application/json"):
        return self.client.request("POST", path, data=body,
            headers={**self.AUTH, "Content-Type": content_type})

    async def _assert_400(self, req, path, param=None):
        seen = len(self.backend.app["paths"])
        async with req as res:
            self.assertEqual(res.status, 400)
            self.assertIn("X-Request-ID", res.headers)
            data = await res.json()

        if path == "/v1/messages":
            self.assertEqual(data["type"], "error")
            self.assertEqual(data["error"]["type"], "invalid_request_error")
        else:
            self.assertEqual(data["error"]["type"], "invalid_request_error")
            self.assertEqual(data["error"]["param"], param)
        self.assertEqual(self.backend.app["paths"][seen:], [])
        self.assertListEqual(await self.get_events(), [])
        return data["error"]["message"]

    async def test_non_object_json_body_is_400(self):
        # Valid JSON, but there is no "model" to route on. Used to be a 500
        # (AttributeError: 'list' object has no attribute 'get').
        for body in (b"[]", b'[{"model": "mymodel"}]', b'"mymodel"', b"null",
                b"5", b"1.5", b"true"):
            for path in self.PATHS:
                with self.subTest(body=body, path=path):
                    msg = await self._assert_400(self._raw(path, body), path)
                    self.assertEqual(msg,
                        "The request body must be a JSON object.")

    async def test_non_string_model_is_400(self):
        # Lists and objects used to be a 500 (unhashable dict key); numbers
        # and booleans a misleading 404 "The model True does not exist.".
        for model in (["mymodel"], {"id": "mymodel"}, [], {}, 123, 1.5,
                True):
            for path in self.PATHS:
                with self.subTest(model=model, path=path):
                    body = json.dumps({"model": model, "input": "hi",
                        "messages": [{"role": "user", "content": "hi"}]})
                    msg = await self._assert_400(self._raw(path, body), path,
                        param="model")
                    self.assertEqual(msg,
                        "Invalid type for 'model': expected a string.")

    async def test_non_string_multipart_model_is_400(self):
        # A "model" part sent as a file becomes a FileField, one with a
        # non-text Content-Type a bytearray. Both are unhashable: used to be
        # a 500.
        file_part = (b'--xyz\r\nContent-Disposition: form-data; name="file"; '
            b'filename="f.bin"\r\n\r\npayload\r\n--xyz--\r\n')
        parts = {
            "file": b'--xyz\r\nContent-Disposition: form-data; name="model"; '
                b'filename="model.txt"\r\n\r\nmywhisper\r\n',
            "bytes": b'--xyz\r\nContent-Disposition: form-data; name="model"'
                b"\r\nContent-Type: application/octet-stream\r\n\r\n"
                b"mywhisper\r\n",
        }
        for path in self.UPLOAD_PATHS:
            for kind, part in parts.items():
                with self.subTest(path=path, model_part=kind):
                    req = self._raw(path, part + file_part,
                        "multipart/form-data; boundary=xyz")
                    await self._assert_400(req, path, param="model")

    async def test_null_model_is_still_404(self):
        # null is "no model": model_not_found, like a missing key.
        async with self._raw("/v1/chat/completions", b'{"model": null}') as res:
            self.assertEqual(res.status, 404)
            data = await res.json()
        self.assertEqual(data["error"]["code"], "model_not_found")

    async def test_undecodable_json_body_is_400(self):
        # Only JSONDecodeError used to be caught; everything else was a 500.
        cases = [
            ("syntax error", b'{"model": ', "application/json"),
            ("empty body", b"", "application/json"),
            ("invalid utf-8", b'{"model": "\xff"}', "application/json"),
            ("unknown charset", b'{"model": "mymodel"}',
                "application/json; charset=no-such-charset"),
            ("deep nesting", b"[" * 100000 + b"]" * 100000,
                "application/json"),
            ("int over digit limit", b'{"model": ' + b"9" * 5000 + b"}",
                "application/json"),
        ]
        for name, body, content_type in cases:
            for path in ("/v1/chat/completions", "/v1/messages"):
                with self.subTest(case=name, path=path):
                    msg = await self._assert_400(
                        self._raw(path, body, content_type), path)
                    self.assertTrue(msg.startswith("JSON decode error: "), msg)

    async def test_json_body_in_declared_charset_still_parsed(self):
        # The fix must not break a body that is valid in its declared charset.
        body = json.dumps({"model": "mymodel",
            "messages": [{"role": "user", "content": "zażółć"}]})
        req = self._raw("/v1/chat/completions", body.encode("utf-16"),
            "application/json; charset=utf-16")
        async with req as res:
            self.assertEqual(res.status, 200)

    async def test_malformed_multipart_is_400(self):
        part = (b'--xyz\r\nContent-Disposition: form-data; name="model"\r\n'
            b'\r\nmywhisper\r\n')
        file_part = (b'--xyz\r\nContent-Disposition: form-data; name="file"; '
            b'filename="a.wav"\r\n')
        cases = [
            ("no boundary", b"garbage", "multipart/form-data"),
            ("garbage body", b"garbage", "multipart/form-data; boundary=xyz"),
            # aiohttp's reader asserts here (AssertionError: Reading after EOF)
            ("truncated", part[:-2], "multipart/form-data; boundary=xyz"),
            # ...and here (assert field.name is not None)
            ("part without name",
                b'--xyz\r\nContent-Disposition: form-data\r\n\r\nx\r\n'
                b"--xyz--\r\n", "multipart/form-data; boundary=xyz"),
            ("unknown transfer encoding", part + file_part
                + b"Content-Transfer-Encoding: rot13\r\n\r\nabc\r\n--xyz--\r\n",
                "multipart/form-data; boundary=xyz"),
            ("bad base64", part + file_part
                + b"Content-Transfer-Encoding: base64\r\n\r\na\r\n--xyz--\r\n",
                "multipart/form-data; boundary=xyz"),
            ("unknown part charset",
                b'--xyz\r\nContent-Disposition: form-data; name="model"\r\n'
                b"Content-Type: text/plain; charset=no-such-charset\r\n\r\n"
                b"mywhisper\r\n--xyz--\r\n",
                "multipart/form-data; boundary=xyz"),
            ("invalid utf-8 field",
                b'--xyz\r\nContent-Disposition: form-data; name="model"\r\n'
                b"\r\n\xff\xfe\r\n--xyz--\r\n",
                "multipart/form-data; boundary=xyz"),
            # http_exceptions.InvalidHeader
            ("part header without colon",
                b'--xyz\r\nContent-Disposition: form-data; name="model"\r\n'
                b"NoColonHere\r\n\r\nmywhisper\r\n--xyz--\r\n",
                "multipart/form-data; boundary=xyz"),
            # http_exceptions.LineTooLong
            ("part header too long",
                b'--xyz\r\nContent-Disposition: form-data; name="model"\r\n'
                b"X-Pad: " + b"a" * 20000 + b"\r\n\r\nmywhisper\r\n--xyz--\r\n",
                "multipart/form-data; boundary=xyz"),
            ("nested multipart",
                b'--xyz\r\nContent-Disposition: form-data; name="model"\r\n'
                b"Content-Type: multipart/mixed; boundary=abc\r\n\r\n"
                b"--abc\r\n\r\nx\r\n--abc--\r\n--xyz--\r\n",
                "multipart/form-data; boundary=xyz"),
            ("boundary over 70 chars", b"--" + b"b" * 80 + b"\r\n",
                "multipart/form-data; boundary=" + "b" * 80),
        ]
        for name, body, content_type in cases:
            for path in self.UPLOAD_PATHS:
                with self.subTest(case=name, path=path):
                    msg = await self._assert_400(
                        self._raw(path, body, content_type), path)
                    self.assertEqual(msg,
                        "Malformed multipart/form-data body.")


class TestConfigValidation(unittest.IsolatedAsyncioTestCase):
    def test_validate_requires_known_backend_type(self):
        # Required: a backend without a type would let any endpoint reach it
        # (e.g. chat completions on an embedding model).
        for meta in ({}, {"type": "completion"}, {"type": "Chat"},
                {"type": ""}, {"type": ["chat"]}, {"type": True}):
            with self.subTest(meta=meta):
                with self.assertRaisesRegex(config.ConfigError,
                        'Backend "m" type must be one of'):
                    config.validate({"backends": {"m": meta}})

    def test_validate_accepts_every_backend_type(self):
        for backend_type in config.BACKEND_TYPES:
            with self.subTest(type=backend_type):
                config.validate({"backends": {"m": {"type": backend_type}}})

    def test_validate_accepts_positive_integer_max_model_len(self):
        config.validate({"backends": {"mymodel": {"type": "chat",
            "max_model_len": 131072}}})

    def test_validate_rejects_invalid_max_model_len(self):
        for value in (0, -1, "131072", True):
            with self.subTest(value=value):
                with self.assertRaises(config.ConfigError):
                    config.validate({
                        "backends": {"mymodel": {"type": "chat",
                            "max_model_len": value}},
                    })

    def test_validate_rejects_invalid_body_limits(self):
        for key in ("client_max_size", "max_json_body"):
            for value in (0, -1, "100", True):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(config.ConfigError):
                        config.validate({key: value})

    def test_validate_rejects_invalid_backend_timeout(self):
        for value in (0, -1, "5", True):
            with self.subTest(value=value):
                with self.assertRaises(config.ConfigError):
                    config.validate({"backends": {"m": {"type": "chat",
                        "timeout": value}}})

    def test_validate_accepts_valid_timeout_and_client_max_size(self):
        config.validate({
            "client_max_size": 2147483648,
            "backends": {"m": {"type": "chat", "timeout": 1800},
                "n": {"type": "chat", "timeout": 0.5}},
        })

    def test_validate_rejects_invalid_provenance(self):
        for cfg in (
                {"provenance": 5},
                {"provenance": True},
                {"provenance": {"enabled": "true"}},
                {"provenance": {"enabled": 1}},
                {"provenance": {"enabled": None}},
                {"provenance": {"generator": ""}},
                {"provenance": {"generator": "   "}},
                {"provenance": {"generator": 5}},
                {"provenance": {"generator": True}}):
            with self.subTest(cfg=cfg):
                with self.assertRaises(config.ConfigError):
                    config.validate(cfg)

    def test_validate_accepts_valid_provenance(self):
        config.validate({"provenance": {}})
        config.validate({"provenance": {"enabled": False, "generator": "x"}})

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
                    "backends": {"mymodel": {"type": "chat",
                        "max_model_len": 0}},
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

    def test_reload_config_updates_provenance(self):
        # SIGHUP is the operational kill switch: [provenance] from the
        # re-read file must replace the live table without a restart.
        app = aiohttp.web.Application()
        app["config"] = {"_path": "dummy.toml", "backends": {},
            "provenance": {}}

        old_load = config.load

        def load_disabled(path):
            return {"backends": {}, "provenance": {"enabled": False}}

        config.load = load_disabled
        try:
            reload_config(app)
        finally:
            config.load = old_load

        self.assertEqual(app["config"]["provenance"], {"enabled": False})


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

    async def test_new_endpoints_get_their_own_path_label(self):
        """/v1/messages and /v1/responses are registered routes, so they must be
        labelled by their real path, not collapsed to "unknown"."""
        for path, body in (
            ("/v1/messages", {"model": "mymodel", "max_tokens": 4,
                "messages": [{"role": "user", "content": "hi"}]}),
            ("/v1/responses", {"model": "mymodel", "input": "hi"}),
        ):
            async with self.client.request("POST", path,
                    headers={"Authorization": "Bearer mytoken"},
                    json=body) as res:
                self.assertEqual(res.status, 200)

        async with self.client.request("GET", "/metrics") as res:
            parsed = _parse_metrics(await res.text())

        lines = parsed.get("llmproxy_requests_total", [])
        for path in ("/v1/messages", "/v1/responses"):
            with self.subTest(path=path):
                self.assertTrue(
                    any('path="%s"' % path in line for line in lines),
                    "requests_total missing path=%s (collapsed to unknown?)"
                        % path)

    async def test_text_endpoints_increment_token_counter(self):
        """chat, messages and responses must each add their billed prompt/
        completion token counts to llmproxy_tokens_total, for both streaming and
        non-streaming responses (previously only chat/embeddings did). The
        registry is process-global, so assert on the delta measured around each
        individual call."""
        def token_value(parsed, type_):
            for line in parsed.get("llmproxy_tokens_total", []):
                if 'model="mymodel"' in line and 'type="%s"' % type_ in line:
                    return float(line.rsplit(" ", 1)[1])
            return 0.0

        async def scrape():
            async with self.client.request("GET", "/metrics") as res:
                return _parse_metrics(await res.text())

        # Mock backend usage is identical stream and non-stream:
        # chat 1/2, messages 3/5, responses 7/9.
        cases = [
            ("/v1/chat/completions",
                {"messages": [{"role": "user", "content": "hi"}]}, 1, 2),
            ("/v1/messages",
                {"max_tokens": 4,
                    "messages": [{"role": "user", "content": "hi"}]}, 3, 5),
            ("/v1/responses", {"input": "hi"}, 7, 9),
        ]
        for path, extra, exp_prompt, exp_completion in cases:
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    before = await scrape()
                    p0 = token_value(before, "prompt")
                    c0 = token_value(before, "completion")

                    body = {"model": "mymodel", "stream": stream, **extra}
                    async with self.client.request("POST", path,
                            headers={"Authorization": "Bearer mytoken"},
                            json=body) as res:
                        self.assertEqual(res.status, 200)
                        await res.read()  # drain streaming body so billing runs

                    after = await scrape()
                    self.assertEqual(
                        token_value(after, "prompt") - p0, exp_prompt)
                    self.assertEqual(
                        token_value(after, "completion") - c0, exp_completion)

    async def test_unknown_path_collapses_to_unknown_label(self):
        """An unregistered path must collapse to path="unknown" so a client
        hammering random paths cannot inflate the label cardinality."""
        async with self.client.request("GET", "/no/such/route",
                headers={"Authorization": "Bearer mytoken"}) as res:
            self.assertEqual(res.status, 404)

        async with self.client.request("GET", "/metrics") as res:
            parsed = _parse_metrics(await res.text())

        lines = parsed.get("llmproxy_requests_total", [])
        self.assertTrue(any('path="unknown"' in line for line in lines),
            "unregistered path should be counted under path=unknown")
        self.assertFalse(
            any('path="/no/such/route"' in line for line in lines),
            "raw unknown path must not get its own series")


class TestRateLimit(LLMProxyAppTestCase):
    """Phase 1 rpm + concurrency: global + per-model, in-memory."""

    async def asyncSetUp(self):
        # Counters are module-global; clear between cases.
        ratelimit.flush()
        await super().asyncSetUp()

    async def get_application(self):
        app = await super().get_application()
        app["config"]["rate_limit"] = {"rpm": 2, "concurrency": 1}
        return app

    async def _chat(self, model="mymodel", **extra):
        body = {"model": model,
            "messages": [{"role": "user", "content": "hi"}], **extra}
        return self.client.request("POST", "/v1/chat/completions",
            headers={"Authorization": "Bearer mytoken"}, json=body)

    async def test_rpm_rejects_excess_with_retry_after(self):
        statuses = []
        retry_after = None
        data = None
        for _ in range(3):
            async with await self._chat() as res:
                statuses.append(res.status)
                if res.status == 429:
                    retry_after = res.headers.get("Retry-After")
                    data = await res.json()
                else:
                    await res.read()
        self.assertEqual(statuses, [200, 200, 429])
        self.assertIsNotNone(retry_after)
        self.assertEqual(data["error"]["type"], "rate_limit_exceeded")
        self.assertIn("mymodel", data["error"]["message"])

    async def test_concurrency_rejects_second_no_retry_after(self):
        body = {"model": "slowok", "_trigger_error": "slow",
            "messages": [{"role": "user", "content": "hi"}]}

        async def one():
            async with self.client.request("POST", "/v1/chat/completions",
                    headers={"Authorization": "Bearer mytoken"},
                    json=body) as res:
                return res.status, res.headers.get("Retry-After")

        task_a = asyncio.ensure_future(one())
        await asyncio.sleep(0.3)  # let A acquire the slot + open the backend
        task_b = asyncio.ensure_future(one())
        sa, _ = await task_a
        sb, ra_b = await task_b
        self.assertEqual(sa, 200)
        self.assertEqual(sb, 429)
        self.assertIsNone(ra_b)

    async def test_per_model_limit_scoped_to_that_model(self):
        # Per-model limit on mymodel only; nolimit (same backend, no sub-table)
        # is unaffected.
        self.app["config"]["rate_limit"] = {}
        self.app["config"]["backends"]["mymodel"]["rate_limit"] = {"rpm": 1}
        ratelimit.flush()

        statuses = []
        for _ in range(2):
            async with await self._chat() as res:
                statuses.append(res.status)
                if res.status == 429:
                    await res.json()
                else:
                    await res.read()
        self.assertEqual(statuses, [200, 429])

        # nolimit has no limit -> succeeds despite mymodel being capped.
        async with await self._chat(model="nolimit") as res:
            self.assertEqual(res.status, 200)
            await res.read()

    async def test_zero_means_unlimited(self):
        # Global rpm=1 but per-model rpm=0 exempts mymodel.
        self.app["config"]["rate_limit"] = {"rpm": 1}
        self.app["config"]["backends"]["mymodel"]["rate_limit"] = {"rpm": 0}
        ratelimit.flush()
        for _ in range(3):
            async with await self._chat() as res:
                self.assertEqual(res.status, 200)
                await res.read()

    async def test_messages_429_is_anthropic_flavour(self):
        self.app["config"]["rate_limit"] = {"rpm": 1}
        self.app["config"]["backends"]["mymodel"]["rate_limit"] = {}
        ratelimit.flush()
        body = {"model": "mymodel", "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}]}

        async with self.client.request("POST", "/v1/messages",
                headers={"Authorization": "Bearer mytoken"}, json=body) as res:
            self.assertEqual(res.status, 200)
            await res.read()
        async with self.client.request("POST", "/v1/messages",
                headers={"Authorization": "Bearer mytoken"}, json=body) as res:
            self.assertEqual(res.status, 429)
            data = await res.json()
        self.assertEqual(data["type"], "error")
        self.assertEqual(data["error"]["type"], "rate_limit_error")
