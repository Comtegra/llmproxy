import datetime
import decimal
import unittest

from llmproxy import streaming
from llmproxy.chat import _ChatStreamUsage
from llmproxy.db import _billing_document

try:
    import bson  # noqa: F401
    HAS_BSON = True
except ImportError:
    HAS_BSON = False


class TestDrain(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_and_captures_all_blocks(self):
        chunks = [b"data: a\n\n", b"data: b\n\n", b"data: [DONE]\n\n", b""]
        it = iter(chunks)
        written = []

        async def read_block():
            return next(it)

        async def write_block(c):
            written.append(c)

        seen = []
        await streaming.drain(read_block, write_block, seen.append)

        self.assertEqual(seen, chunks[:3])
        self.assertEqual(written, chunks[:3])

    async def test_keeps_reading_after_client_disconnect(self):
        # The moat: writes fail after the first block (client gone), but every
        # remaining block — including the trailing usage chunk — must still be
        # read and fed to on_chunk, so billing survives a mid-stream disconnect.
        chunks = [b"data: content\n\n", b"data: usage\n\n",
            b"data: [DONE]\n\n", b""]
        it = iter(chunks)
        written = []

        async def read_block():
            return next(it)

        async def write_block(c):
            written.append(c)
            raise ConnectionResetError("client gone")

        seen = []
        disconnects = []
        await streaming.drain(read_block, write_block, seen.append,
            on_disconnect=disconnects.append)

        self.assertEqual(seen, chunks[:3])        # every block was seen
        self.assertEqual(written, [chunks[0]])    # client only got the first
        self.assertEqual(len(disconnects), 1)

    async def test_starts_disconnected_still_drains(self):
        # Client already gone before the first write (e.g. a header-flush /
        # prepare-time disconnect): every block must still be read and fed to
        # on_chunk, and nothing is written.
        chunks = [b"data: content\n\n", b"data: usage\n\n", b""]
        it = iter(chunks)

        async def read_block():
            return next(it)

        written = []

        async def write_block(c):
            written.append(c)

        seen = []
        await streaming.drain(read_block, write_block, seen.append,
            disconnected=True)

        self.assertEqual(seen, chunks[:2])
        self.assertEqual(written, [])


class TestStreamThroughPrepareDisconnect(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_time_disconnect_still_captures_usage(self):
        # Regression guard: if the client vanishes as headers are flushed,
        # prepare() raises OSError. stream_through must still drain the backend
        # so usage is captured and billing can run (the pre-refactor behavior).
        import logging as _logging
        from unittest import mock

        import aiohttp.web

        from llmproxy.chat import _ChatStreamUsage

        blocks = (
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            b'data: {"choices":[],"usage":'
            b'{"prompt_tokens":1,"completion_tokens":2}}\n\n'
            b"data: [DONE]\n\n"
        )

        class FakeContent:
            def __init__(self, data):
                self._d = data
                self._i = 0

            async def read(self, n):
                chunk = self._d[self._i:self._i + n]
                self._i += n
                return chunk

        class FakeBackendResp:
            headers = {}
            content = FakeContent(blocks)

        class FakeApp:
            logger = _logging.getLogger("test")

            def __getitem__(self, key):
                return {}

        class FakeReq:
            app = FakeApp()

            def __getitem__(self, key):
                return "rid"

        acc = _ChatStreamUsage()
        with mock.patch.object(aiohttp.web.StreamResponse, "prepare",
                side_effect=ConnectionResetError("client gone")), \
             mock.patch.object(aiohttp.web.StreamResponse, "write",
                new=mock.AsyncMock()), \
             mock.patch.object(aiohttp.web.StreamResponse, "write_eof",
                new=mock.AsyncMock()):
            await streaming.stream_through(FakeReq(), FakeBackendResp(), acc)

        # Usage captured despite the prepare-time disconnect -> billing can run.
        self.assertEqual(acc.usage(),
            {"prompt_tokens": 1, "completion_tokens": 2})


class TestChatStreamUsage(unittest.TestCase):
    def test_usage_read_from_trailing_chunk(self):
        acc = _ChatStreamUsage()
        for c in [
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            b'data: {"choices":[],"usage":'
            b'{"prompt_tokens":3,"completion_tokens":5}}\n\n',
            b"data: [DONE]\n\n",
        ]:
            acc(c)
        self.assertEqual(acc.usage(),
            {"prompt_tokens": 3, "completion_tokens": 5})


class TestBillingDocument(unittest.TestCase):
    @staticmethod
    def _user():
        return {"_namespace": "ns", "_user_id": "uid", "id": "keyid",
            "_org_id": "org", "_tier": "pro"}

    def test_shape_and_request_id(self):
        t = datetime.datetime(2026, 7, 1, tzinfo=datetime.UTC)
        doc = _billing_document(
            self._user(), t,
            {"whisper-1/a5000/transcription": decimal.Decimal("125.4")},
            "req-123")

        self.assertEqual(doc["namespace"], "ns")
        self.assertEqual(doc["user_id"], "uid")
        self.assertEqual(doc["api_key_id"], "keyid")
        self.assertEqual(doc["org_id"], "org")
        self.assertEqual(doc["tier"], "pro")
        self.assertEqual(doc["type"], "oneoff")
        self.assertEqual(doc["created_at"], t)
        self.assertEqual(doc["request_id"], "req-123")
        self.assertEqual(doc["resources"],
            {"whisper-1/a5000/transcription": decimal.Decimal("125.4")})

    def test_request_id_stringified(self):
        import uuid
        rid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        doc = _billing_document(self._user(),
            datetime.datetime(2026, 7, 1, tzinfo=datetime.UTC),
            {"m/d/prompt": 1}, rid)
        self.assertEqual(doc["request_id"],
            "12345678-1234-5678-1234-567812345678")


@unittest.skipUnless(HAS_BSON, "bson (pymongo) not installed")
class TestDecimalToBson(unittest.TestCase):
    def test_whole_number_stored_as_int(self):
        from llmproxy.db import _decimal_to_bson
        r = _decimal_to_bson(decimal.Decimal("120.0"))
        self.assertIsInstance(r, int)
        self.assertEqual(r, 120)

    def test_fractional_stored_as_decimal128(self):
        from llmproxy.db import _decimal_to_bson
        r = _decimal_to_bson(decimal.Decimal("125.4"))
        self.assertEqual(type(r).__name__, "Decimal128")
        self.assertEqual(str(r), "125.4")


if __name__ == "__main__":
    unittest.main()
