"""End-to-end tests for /v1/files/convert against the production marker
microservice (Datalab-compat sync PDF -> Markdown over HTTPS).

The mock-backed tests (test_proxy, test_provenance) verify the proxy's
contract against tests/mockbackend.py. These tests instead run the REAL
proxy app against the REAL microservice, covering everything the mock
cannot: the /healthz startup probe (health_path), the HTTPS multipart
forward through aiohttp's tempfile payload path, production Marker's
response shape (success/output/page_count/provenance, x-ai-* headers)
and the fail-loud billing policy on the live failure semantics.

They hit PRODUCTION, so they are strictly opt-in and never run in CI:
set MARKER_URL and MARKER_TOKEN (the microservice APP_TOKEN) in the
environment, e.g.

    MARKER_URL=https://marker-microservice.apps.cgc-krk-01.comtegra.cloud \\
    MARKER_TOKEN=<app-token> \\
    .venv/bin/python -m pytest tests/test_e2e_marker.py -v

The token is never committed: it lives in the environment only.
Without both variables the module is skipped, like any other network-
dependent e2e suite.
"""

import hashlib
import os
import tempfile
import unittest
import warnings

import aiohttp
import aiohttp.test_utils

from llmproxy import auth
from llmproxy.app import create_app
from llmproxy.db import get_db

from tests.test_proxy import LLMProxyAppTestCase

MARKER_URL = os.environ.get("MARKER_URL")
MARKER_TOKEN = os.environ.get("MARKER_TOKEN")
MARKER_DEVICE = os.environ.get("MARKER_DEVICE", "cpu")

# Printed on the generated page. OCR can mangle ambiguous glyphs
# ("llmproxy" -> "Ilmproxy" on the live service), so assertions match the
# case-insensitive stable prefix only.
PDF_TEXT = "Hello World from llmproxy e2e"


def build_pdf(text=PDF_TEXT):
    """Minimal single-page PDF (one line of Helvetica text), hand-rolled.

    No external deps: catalog/pages/page/content/font objects plus a valid
    xref table, so Marker's PDF parser (pypdfium2) accepts it.
    """
    stream = "BT /F1 24 Tf 72 720 Td (%s) Tj ET" % text
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        "<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body.encode())
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1, xref_pos)
    return bytes(out)


@unittest.skipIf(not MARKER_URL or not MARKER_TOKEN,
    "e2e: set MARKER_URL and MARKER_TOKEN to run against the production "
    "marker microservice")
class TestMarkerE2E(LLMProxyAppTestCase):
    """Real proxy -> real marker microservice, per-test fresh app+db."""

    BACKEND = "pdf-to-markdown"

    async def asyncSetUp(self):
        # Same reset semantics as the base harness (process-global auth
        # cache would otherwise leak keys across the fresh per-test db),
        # but no mock backend: this suite talks to production over HTTPS.
        warnings.simplefilter("ignore", category=aiohttp.web.NotAppKeyWarning)
        auth.flush_cache()
        await aiohttp.test_utils.AioHTTPTestCase.asyncSetUp(self)

    async def get_application(self):
        self.db_fd, self.db_path = tempfile.mkstemp()

        # Mirrors the config.toml example backend. The /healthz probe runs
        # on startup via check_backends: if health_path were wrong the test
        # errors out before any request is sent.
        app = await create_app({
            "timeout_connect": 15,
            "timeout_read": 60,
            "max_json_body": 1024 * 1024,
            "db": {"uri": "sqlite://%s" % self.db_path},
            "backends": {
                self.BACKEND: {
                    "url": MARKER_URL,
                    "token": MARKER_TOKEN,
                    "device": MARKER_DEVICE,
                    # Real conversion is slower than the mock; per-backend
                    # sock_read override (same mechanism as audio).
                    "timeout": 600,
                    "health_path": "healthz",
                },
            },
        })

        # Same test credential the mock harness uses.
        secret = hashlib.sha256(b"mytoken").hexdigest()
        db = await get_db(app["config"]["db"]["uri"])
        await db.db.execute(
            "INSERT INTO api_key (id, secret, type) "
            "VALUES ('myuser', ?, 'LLM')", (secret,))
        await db.db.commit()
        await db.close()

        return app

    def post_convert(self, **file_kwargs):
        """Client request for /v1/files/convert selecting the marker backend.

        Pass filename/blocks/content_type to add the `file` field; omit
        them (and pass nothing) to test the missing-file guard.
        """
        form = aiohttp.FormData()
        form.add_field("model", self.BACKEND)
        if file_kwargs:
            form.add_field("file", **file_kwargs)
        return self.client.request("POST", "/v1/files/convert",
            headers={"Authorization": "Bearer mytoken"}, data=form)

    async def test_startup_healthz_probe(self):
        # check_backends already passed during asyncSetUp (it raises and
        # the whole class errors otherwise); assert the configured shape
        # explicitly so a health_path regression fails here, not just by
        # blowing up the fixture.
        cfg = self.app["config"]["backends"][self.BACKEND]
        self.assertEqual(cfg["health_path"], "healthz")
        async with self.app["client"].get(
                "%s/healthz" % MARKER_URL) as res:
            self.assertEqual(res.status, 200)
            self.assertEqual((await res.json())["status"], "ok")

    async def test_convert_pdf_bills_per_page(self):
        req = self.post_convert(
            content_type="application/pdf", filename="e2e.pdf",
            value=build_pdf())

        async with req as res:
            self.assertEqual(res.status, 200)
            data = await res.json()

        self.assertIs(data["success"], True)
        self.assertEqual(data["page_count"], 1)
        self.assertEqual(data["output_format"], "markdown")
        self.assertIn("hello world", data["output"].lower())
        self.assertListEqual(await self.get_events(), [
            {"product": "%s/%s/conversion" % (self.BACKEND, MARKER_DEVICE),
             "quantity": 1},
        ])

    async def test_convert_marks_headers_and_forwards_provenance(self):
        req = self.post_convert(
            content_type="application/pdf", filename="e2e.pdf",
            value=build_pdf())

        async with req as res:
            self.assertEqual(res.status, 200)
            # Proxy's Art. 50(2) header-only marking for conversions.
            self.assertEqual(res.headers["X-AI-Generated"], "true")
            # Marker's own AI headers are forwarded, not dropped.
            self.assertEqual(res.headers.get("X-AI-Processed"), "true")
            data = await res.json()

        # Body forwarded byte-for-byte: marker's own provenance object
        # survives (the proxy never injects/overwrites it on this route).
        self.assertIs(data["provenance"]["ai_processed"], True)
        self.assertEqual(data["provenance"]["processing"],
            "pdf-to-markdown-conversion")

    async def test_invalid_pdf_is_422_and_not_billed(self):
        # Datalab contract: HTTP 200 + success=false. The proxy must map
        # that to 422 for the client and emit no billing row.
        req = self.post_convert(
            content_type="application/pdf", filename="fake.pdf",
            value=b"this is not a PDF at all")

        async with req as res:
            self.assertEqual(res.status, 422)
            data = await res.json()

        self.assertIs(data.get("success"), False)
        self.assertIn("pdf", data.get("error", "").lower())
        self.assertListEqual(await self.get_events(), [])

    async def test_missing_file_is_422_and_not_billed(self):
        # A file field under the wrong name keeps the body multipart
        # (string-only FormData would be urlencoded and rejected 415 by
        # the middleware before reaching the handler), so this exercises
        # the handler guard. The backend is never contacted.
        form = aiohttp.FormData()
        form.add_field("model", self.BACKEND)
        form.add_field("not_file", b"%PDF-1.4", filename="doc.pdf",
            content_type="application/pdf")
        req = self.client.request("POST", "/v1/files/convert",
            headers={"Authorization": "Bearer mytoken"}, data=form)

        async with req as res:
            self.assertEqual(res.status, 422)

        self.assertListEqual(await self.get_events(), [])
