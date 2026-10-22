"""Tests for the /v1/realtime WebSocket endpoint.

Auth (Task 1): token extraction for WS handshakes and the require_token
entry point, ahead of the endpoint itself. Browser WebSocket clients cannot
set headers, so their key rides the handshake subprotocols
(`Sec-WebSocket-Protocol: bearer, <key>`).
"""

import json
import unittest
from unittest import mock

import aiohttp
import aiohttp.test_utils
import aiohttp.web

from llmproxy import auth, config

from . import mockbackend


class _App:
    logger = mock.MagicMock()

    def __init__(self, config):
        self.config = config


class _Req(dict):
    """Request stub carrying just what require_token touches: .app and
    .headers, plus dict access for the per-request db slot."""

    def __init__(self, headers, app=None):
        super().__init__()
        self.app = app or _App({"db": {"uri": "sqlite://:memory:"}})
        self.headers = headers


class TestExtractToken(unittest.TestCase):
    def test_bearer_header(self):
        req = _Req({"Authorization": "Bearer mytoken"})
        self.assertEqual(auth.extract_token(req), "mytoken")

    def test_browser_subprotocol_pair(self):
        req = _Req({"Sec-WebSocket-Protocol": "bearer, mytoken"})
        self.assertEqual(auth.extract_token(req), "mytoken")

    def test_missing_credentials(self):
        self.assertIsNone(auth.extract_token(_Req({})))

    def test_header_wins_over_subprotocol(self):
        req = _Req({"Authorization": "Bearer header-token",
            "Sec-WebSocket-Protocol": "bearer, sub-token"})
        self.assertEqual(auth.extract_token(req), "header-token")

    def test_subprotocol_must_be_bearer_key(self):
        # One subprotocol, or a pair not led by "bearer": not an auth
        # attempt -- a key must never be extracted by accident.
        for value in ("bearer", "mytoken, bearer", "foo, mytoken",
                "bearer, key, extra"):
            with self.subTest(value=value):
                self.assertIsNone(auth.extract_token(
                    _Req({"Sec-WebSocket-Protocol": value})))

    def test_authorization_scheme_must_be_bearer(self):
        self.assertIsNone(auth.extract_token(
            _Req({"Authorization": "Basic mytoken"})))


class TestRequireToken(unittest.IsolatedAsyncioTestCase):
    async def test_missing_token_is_401_without_a_lookup(self):
        # A handshake with no credentials anywhere must not cost a
        # database round trip: fail before the digest is even computed.
        req = _Req({})
        with mock.patch("llmproxy.auth.get_db") as get_db:
            with self.assertRaises(aiohttp.web.HTTPUnauthorized) as caught:
                await auth.require_token(req, None)
        get_db.assert_not_called()
        self.assertIn("Incorrect API key", caught.exception.text)


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
