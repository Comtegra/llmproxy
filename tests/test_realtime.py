"""Tests for the /v1/realtime WebSocket endpoint."""

import asyncio
import json
import time
import unittest
from unittest import mock

import aiohttp
import aiohttp.test_utils
import aiohttp.web

from llmproxy import config

from . import mockbackend
from .test_proxy import LLMProxyAppTestCase, _parse_metrics


class TestRealtimeConfig(unittest.TestCase):
    def _backends(self, **meta):
        base = {"url": "https://example.test", "token": "t",
            "device": "none", "type": "realtime"}
        base.update(meta)
        return {"myrealtime": base}

    def _validate(self, **meta):
        config.validate({"backends": self._backends(**meta)})

    def test_accepts_realtime_backend(self):
        self._validate()

    def test_accepts_heartbeat(self):
        self._validate(heartbeat=30)

    def test_token_required(self):
        for token in (None, "", "  "):
            with self.subTest(token=token):
                with self.assertRaises(config.ConfigError):
                    self._validate(token=token)

    def test_device_required(self):
        with self.assertRaises(config.ConfigError):
            self._validate(device=None)

    def test_heartbeat_positive_number(self):
        for value in (0, -5, "x", True):
            with self.subTest(value=value):
                with self.assertRaises(config.ConfigError):
                    self._validate(heartbeat=value)


class MockBackendRealtimeTest(unittest.IsolatedAsyncioTestCase):
    """The mock WS backend itself: the proxy-level tests sit on top of it.

    The mock speaks the OpenAI Realtime transcription subset the real
    backend speaks: token check first (error event + close 1008), scripted
    completed events with usage seconds on commit, everything else echoed."""

    async def asyncSetUp(self):
        self.app = mockbackend.create_app()
        self.server = aiohttp.test_utils.TestServer(self.app)
        await self.server.start_server()
        self.session = aiohttp.ClientSession()
        self.url = "http://%s:%d" % (self.server.host, self.server.port)

    async def asyncTearDown(self):
        await self.session.close()
        await self.server.close()

    async def connect(self, token="mybackendtoken", **kwargs):
        headers = {"Authorization": "Bearer %s" % token} if token else {}
        return await self.session.ws_connect(self.url + "/v1/realtime",
            headers=headers, **kwargs)

    async def test_commit_answers_completed_with_usage(self):
        ws = await self.connect()
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        evt = json.loads((await ws.receive()).data)
        self.assertEqual(evt["type"],
            "conversation.item.input_audio_transcription.completed")
        self.assertEqual(evt["usage"], {"type": "duration", "seconds": 3.7})
        await ws.close()

    async def test_other_frames_are_echoed(self):
        ws = await self.connect()
        await ws.send_str(json.dumps(
            {"type": "session.update", "session": {}}))
        evt = json.loads((await ws.receive()).data)
        self.assertEqual(evt["type"], "session.update")
        await ws.close()

    async def test_wrong_token_is_error_then_1008(self):
        ws = await self.connect(token="nope")
        evt = json.loads((await ws.receive()).data)
        self.assertEqual(evt["type"], "error")
        msg = await ws.receive()
        self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSE)
        self.assertEqual(msg.data, 1008)

    async def test_max_sessions_closes_1013(self):
        self.app["ws_control"]["max_sessions"] = True
        ws = await self.connect()
        msg = await ws.receive()
        self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSE)
        self.assertEqual(msg.data, 1013)

    async def test_close_code_ends_the_session(self):
        self.app["ws_control"]["close_code"] = 1000
        self.app["ws_control"]["echo"] = False
        ws = await self.connect()
        await ws.send_str(json.dumps(
            {"type": "input_audio_buffer.append", "audio": "QQ=="}))
        msg = await ws.receive()
        self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSE)
        self.assertEqual(msg.data, 1000)

    async def test_records_headers_and_model(self):
        self.app["ws_control"]["usage_seconds"] = 2.5
        ws = await self.connect(params={"model": "mymodel"})
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.receive()
        await ws.close()
        rec = self.app["ws_records"][0]
        self.assertEqual(rec["headers"]["Authorization"],
            "Bearer mybackendtoken")
        self.assertEqual(rec["model"], "mymodel")


class _NoCancellationTestServer(aiohttp.test_utils.TestServer):
    """The proxy's real server runs with handler_cancellation=False (a
    billing-critical pin in app.py), so a client disconnect never kills the
    handler mid-drain. aiohttp's test harness forces the opposite; this
    server restores production behavior."""

    async def _make_runner(self, **kwargs):
        kwargs["handler_cancellation"] = False
        return aiohttp.web.AppRunner(self.app, **kwargs)


class RealtimeTestCase(LLMProxyAppTestCase):
    """Proxy-level tests: one realtime backend pointing at the mock.

    Pre-upgrade failures (auth, model routing) are asserted with a plain GET:
    those paths raise before the WebSocket is prepared, so the GET response
    carries exactly the JSON body a failed handshake would."""

    AUTH = {"Authorization": "Bearer mytoken"}

    async def get_server(self, app):
        return _NoCancellationTestServer(app)

    async def get_application(self):
        app = await super().get_application()
        app["config"]["backends"]["myrealtime"] = {
            "url": "http://%s:%d" % (self.backend.host, self.backend.port),
            "token": "mybackendtoken",
            "device": "none",
            "type": "realtime",
        }
        return app

    async def error_body(self, **kwargs):
        resp = await self.client.get("/v1/realtime", **kwargs)
        return resp.status, await resp.json()


class TestRealtimeHandshake(RealtimeTestCase):
    async def test_no_credentials_is_401(self):
        status, body = await self.error_body()
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["type"], "authentication_error")

    async def test_bad_key_is_401(self):
        status, body = await self.error_body(
            headers={"Authorization": "Bearer nope"})
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["type"], "authentication_error")

    async def test_bad_key_fails_the_handshake_itself(self):
        with self.assertRaises(aiohttp.WSServerHandshakeError) as caught:
            await self.client.ws_connect("/v1/realtime")
        self.assertEqual(caught.exception.status, 401)

    async def test_unknown_model_is_404(self):
        status, body = await self.error_body(headers=self.AUTH,
            params={"model": "doesnotexist"})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "model_not_found")

    async def test_model_of_another_type_is_404(self):
        # A chat model must never reach a backend over the realtime endpoint.
        status, body = await self.error_body(headers=self.AUTH,
            params={"model": "mymodel"})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "model_not_supported")

    async def test_missing_model_with_two_backends_is_400(self):
        self.app["config"]["backends"]["myrealtime2"] = {
            "url": "http://%s:%d" % (self.backend.host, self.backend.port),
            "token": "mybackendtoken",
            "device": "none",
            "type": "realtime",
        }
        status, body = await self.error_body(headers=self.AUTH)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["param"], "model")

    async def test_no_realtime_backends_is_404(self):
        del self.app["config"]["backends"]["myrealtime"]
        status, body = await self.error_body(headers=self.AUTH)
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "model_not_found")
        self.assertIn("No realtime model", body["error"]["message"])

    async def test_backend_unreachable_is_gateway_error(self):
        self.app["config"]["backends"]["myrealtime"]["url"] = \
            "http://127.0.0.1:1"
        with self.assertRaises(aiohttp.WSServerHandshakeError) as caught:
            await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        # 502 = refused (Linux), 504 = the same refusal surfacing as a
        # connect timeout (Windows): either way the session never opens.
        self.assertIn(caught.exception.status, (502, 504))

    async def test_bearer_header_auth_connects(self):
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        try:
            await ws.send_str(json.dumps(
                {"type": "session.update", "session": {}}))
            msg = await ws.receive()
            self.assertEqual(json.loads(msg.data)["type"], "session.update")
        finally:
            await ws.close()


class TestRealtimeSession(RealtimeTestCase):
    async def audio_seconds(self):
        async with self.client.request("GET", "/metrics") as res:
            parsed = _parse_metrics(await res.text())
        for line in parsed.get("llmproxy_audio_seconds_total", []):
            if 'model="myrealtime"' in line:
                return float(line.rsplit(" ", 1)[1])
        return 0.0

    async def test_commit_is_answered_and_metered(self):
        before = await self.audio_seconds()
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        evt = json.loads((await ws.receive()).data)
        self.assertEqual(evt["type"],
            "conversation.item.input_audio_transcription.completed")
        self.assertEqual(evt["usage"]["seconds"], 3.7)
        await ws.close()
        self.assertAlmostEqual(await self.audio_seconds() - before, 3.7)

    async def test_non_completed_frames_are_not_metered(self):
        before = await self.audio_seconds()
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str(json.dumps(
            {"type": "session.update", "session": {}}))
        self.assertEqual(json.loads((await ws.receive()).data)["type"],
            "session.update")
        await ws.close()
        self.assertAlmostEqual(await self.audio_seconds(), before)

    async def test_idle_session_survives_the_http_read_timeout(self):
        # The global timeout_read bounds HTTP response reads, not WebSocket
        # sessions: a realtime session idles whenever the client pauses
        # audio (the backend enforces its own idle limits), so it must
        # outlive the HTTP timeout of this harness (1 s) without a ping.
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        try:
            await asyncio.sleep(1.5)
            await ws.send_str(json.dumps(
                {"type": "session.update", "session": {}}))
            msg = await ws.receive()
            self.assertEqual(json.loads(msg.data)["type"], "session.update")
        finally:
            await ws.close()

    async def test_garbage_text_frame_is_still_forwarded(self):
        before = await self.audio_seconds()
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str("not json at all")
        msg = await ws.receive()
        self.assertEqual(msg.data, "not json at all")
        await ws.close()
        self.assertAlmostEqual(await self.audio_seconds(), before)

    async def test_backend_sees_only_its_token_and_rewritten_model(self):
        # The public alias maps to the backend's real model id; the client's
        # key must never reach the backend.
        self.app["config"]["backends"]["myrealtime"]["model"] = \
            "nvidia/parakeet-tdt-0.6b-v3"
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str(json.dumps(
            {"type": "input_audio_buffer.append", "audio": "QQ=="}))
        await ws.receive()  # the echo
        await ws.close()
        rec = self.backend.app["ws_records"][0]
        self.assertEqual(rec["headers"]["Authorization"],
            "Bearer mybackendtoken")
        self.assertNotIn("Sec-WebSocket-Protocol", rec["headers"])
        self.assertEqual(rec["model"], "nvidia/parakeet-tdt-0.6b-v3")
        self.assertEqual(json.loads(rec["frames"][0])["type"],
            "input_audio_buffer.append")

    async def test_backend_auth_rejection_is_masked_as_1011(self):
        # A backend that rejects the proxy's token is a CONFIGURATION fault:
        # the client sees a generic 1011 and a CRITICAL names the backend,
        # never the backend's own reason.
        self.backend.app["ws_control"]["reject_auth"] = True
        with mock.patch.object(self.app.logger, "critical") as critical:
            ws = await self.client.ws_connect("/v1/realtime",
                headers=self.AUTH)
            self.assertEqual(json.loads((await ws.receive()).data)["type"],
                "error")
            msg = await ws.receive()
            self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSE)
            self.assertEqual(msg.data, 1011)
            critical.assert_called_once()
            self.assertIn("myrealtime", critical.call_args[0][0] %
                critical.call_args[0][1:]
                if critical.call_args[0][1:] else critical.call_args[0][0])

    async def test_session_limit_is_1013_passthrough(self):
        self.backend.app["ws_control"]["max_sessions"] = True
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        msg = await ws.receive()
        self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSE)
        self.assertEqual(msg.data, 1013)

    async def test_backend_close_code_propagates(self):
        self.backend.app["ws_control"]["close_code"] = 1000
        self.backend.app["ws_control"]["echo"] = False
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str(json.dumps(
            {"type": "input_audio_buffer.append", "audio": "QQ=="}))
        msg = await ws.receive()
        self.assertEqual(msg.type, aiohttp.WSMsgType.CLOSE)
        self.assertEqual(msg.data, 1000)

    async def test_client_disconnect_still_meters_inflight_usage(self):
        # The client hangs up right after committing: the completed event is
        # in flight and the drain must still meter its seconds (billing
        # later plugs into the same hook).
        before = await self.audio_seconds()
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.close()
        async def metered():
            # The counter accumulates float seconds across the suite
            # (process-global by Prometheus nature), so the delta is
            # compared with a rounding tolerance, never exactly.
            while round(await self.audio_seconds() - before, 6) < 3.7:
                await asyncio.sleep(0.05)
        await asyncio.wait_for(metered(), 5)

    async def test_two_inflight_events_are_both_metered_after_disconnect(self):
        # Every in-flight completed event must be metered, not just the
        # first: a send to the gone client must never kill the drain pump.
        before = await self.audio_seconds()
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.close()
        async def metered():
            # Float accumulation across the suite: compare with tolerance.
            while round(await self.audio_seconds() - before, 6) < 7.4:
                await asyncio.sleep(0.05)
        await asyncio.wait_for(metered(), 5)


class TestRealtimeBilling(RealtimeTestCase):
    """Per-segment ONEOFF billing on the usage hook: each completed segment
    is billed the moment it lands (never a lump at session end), with the
    same product shape as offline transcription."""

    async def test_completed_segments_are_billed_incrementally(self):
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        evt = json.loads((await ws.receive()).data)
        self.assertEqual(evt["type"],
            "conversation.item.input_audio_transcription.completed")
        # The first segment is billed while the session is still live:
        # a per-session lump at disconnect would never show up here.
        self.assertListEqual(await self.get_events(expect=1), [
            {"product": "myrealtime/none/transcription", "quantity": 3.7},
        ])
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        evt = json.loads((await ws.receive()).data)
        self.assertEqual(evt["usage"]["seconds"], 3.7)
        await ws.close()
        # One event per segment (2 x 3.7), never one event per session (7.4).
        self.assertListEqual(await self.get_events(expect=2), [
            {"product": "myrealtime/none/transcription", "quantity": 3.7},
            {"product": "myrealtime/none/transcription", "quantity": 3.7},
        ])

    async def test_inflight_segment_is_billed_after_client_disconnect(self):
        # The client hangs up right after committing: the completed event
        # is in flight and the drain grace must bill it, like it meters it.
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.close()
        self.assertListEqual(await self.get_events(expect=1), [
            {"product": "myrealtime/none/transcription", "quantity": 3.7},
        ])

    async def test_non_completed_frames_are_not_billed(self):
        # Echoed frames carry no usage: billing, like metering, only ever
        # touches completed events.
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.send_str(json.dumps(
            {"type": "session.update", "session": {}}))
        self.assertEqual(json.loads((await ws.receive()).data)["type"],
            "session.update")
        await ws.close()
        self.assertListEqual(await self.get_events(), [])


class TestRealtimeRateLimits(RealtimeTestCase):
    async def test_concurrency_bounds_parallel_sessions(self):
        self.app["config"]["backends"]["myrealtime"]["rate_limit"] = \
            {"concurrency": 1}
        first = await self.client.ws_connect("/v1/realtime",
            headers=self.AUTH)
        try:
            with self.assertRaises(aiohttp.WSServerHandshakeError) as caught:
                await self.client.ws_connect("/v1/realtime",
                    headers=self.AUTH)
            self.assertEqual(caught.exception.status, 429)
            # The rejected handshake never opened a backend session.
            self.assertEqual(len(self.backend.app["ws_records"]), 1)
        finally:
            await first.close()
        # The slot outlives the client close by the drain grace (the backend
        # session is still alive), so poll until the handler released it.
        deadline = time.monotonic() + 10
        while True:
            try:
                second = await self.client.ws_connect("/v1/realtime",
                    headers=self.AUTH)
                break
            except aiohttp.WSServerHandshakeError as exc:
                if time.monotonic() > deadline or exc.status != 429:
                    raise
                await asyncio.sleep(0.25)
        await second.close()

    async def test_rpm_bounds_handshakes(self):
        self.app["config"]["backends"]["myrealtime"]["rate_limit"] = \
            {"rpm": 1}
        ws = await self.client.ws_connect("/v1/realtime", headers=self.AUTH)
        await ws.close()
        # The window is per (user, model): a second handshake within 60 s
        # of the admitted one is rejected before any backend connection.
        async with self.client.get("/v1/realtime", headers=self.AUTH) as resp:
            self.assertEqual(resp.status, 429)
            self.assertEqual(resp.headers["Retry-After"], "60")
            body = await resp.json()
            self.assertEqual(body["error"]["code"], "rate_limit_exceeded")
        self.assertEqual(len(self.backend.app["ws_records"]), 1)
