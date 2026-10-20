"""Background file conversions: owner-bound ids, polling and billing at the
first pickup (POST /v1/files/convert with background=true, then
GET /v1/files/convert/{id})."""

import asyncio
import base64
import copy
import json
import secrets
import string
import unittest
from unittest import mock

import aiohttp

from llmproxy import config, conversion_id, files, metrics
from llmproxy.app import reload_config

from .test_proxy import LLMProxyAppTestCase


TOKEN = "mybackendtoken"
RID = "0123456789abcdef0123456789abcdef"


def _body(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


class TestConversionId(unittest.TestCase):
    def test_round_trip(self):
        job_id = conversion_id.make(TOKEN, "myuser", "mymarker", RID)
        self.assertTrue(job_id.startswith("fc_"))
        parsed = conversion_id.parse(job_id)
        self.assertEqual((parsed.backend, parsed.request_id),
            ("mymarker", RID))
        self.assertTrue(conversion_id.verify(TOKEN, "myuser", parsed))
        # URL-safe without escaping, short enough to paste around.
        self.assertLessEqual(set(job_id),
            set(string.ascii_letters + string.digits + "_-."))
        self.assertLess(len(job_id), 120)

    def test_bound_to_owner_and_token(self):
        parsed = conversion_id.parse(
            conversion_id.make(TOKEN, "myuser", "mymarker", RID))
        for token, owner in ((TOKEN, "otheruser"), ("othertoken", "myuser"),
                (TOKEN, "myuser "), ("", "myuser"), (None, "myuser")):
            with self.subTest(token=token, owner=owner):
                self.assertFalse(conversion_id.verify(token, owner, parsed))

    def test_owner_compared_as_string(self):
        # Mongo key ids are ObjectIds; the id must verify with the same value
        # that made it, whatever its type.
        class Oid:
            def __str__(self):
                return "65f0c0ffee"
        job_id = conversion_id.make(TOKEN, Oid(), "mymarker", RID)
        self.assertTrue(conversion_id.verify(TOKEN, "65f0c0ffee",
            conversion_id.parse(job_id)))

    def test_every_changed_character_is_rejected(self):
        job_id = conversion_id.make(TOKEN, "myuser", "mymarker", RID)
        alphabet = string.ascii_letters + string.digits + "_-."
        for i, ch in enumerate(job_id):
            replacement = alphabet[(alphabet.index(ch) + 1) % len(alphabet)]
            forged = job_id[:i] + replacement + job_id[i + 1:]
            with self.subTest(position=i):
                parsed = conversion_id.parse(forged)
                self.assertFalse(parsed is not None
                    and conversion_id.verify(TOKEN, "myuser", parsed))

    def test_truncated_and_extended_ids_rejected(self):
        job_id = conversion_id.make(TOKEN, "myuser", "mymarker", RID)
        for forged in [job_id[:n] for n in range(len(job_id))] \
                + [job_id + "A", job_id + "=", job_id + ".", " " + job_id]:
            with self.subTest(forged=forged):
                parsed = conversion_id.parse(forged)
                self.assertFalse(parsed is not None
                    and conversion_id.verify(TOKEN, "myuser", parsed))

    def test_malformed_ids_do_not_parse(self):
        def b64(data):
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

        tag = b64(b"x" * 16)
        cases = [
            "", "fc_", "fc_.", "?", "*", "..", "fc_*.*", "fc_..%2f",
            "fc_%s" % b64(b'["mymarker","%s"]' % RID.encode()),
            "xx_%s.%s" % (b64(b'["mymarker","%s"]' % RID.encode()), tag),
            "fc_%s.%s" % (b64(b"not json"), tag),
            "fc_%s.%s" % (b64(b'{"a":1}'), tag),
            "fc_%s.%s" % (b64(b'["mymarker"]'), tag),
            "fc_%s.%s" % (b64(b'["mymarker","%s","x"]' % RID.encode()), tag),
            "fc_%s.%s" % (b64(b'["","%s"]' % RID.encode()), tag),
            "fc_%s.%s" % (b64(b'[1,"%s"]' % RID.encode()), tag),
            "fc_%s.%s" % (b64(b'["mymarker",123]'), tag),
            # Request ids that would escape the backend URL path segment.
            "fc_%s.%s" % (b64(b'["mymarker","../../healthz"]'), tag),
            "fc_%s.%s" % (b64(b'["mymarker","a/b"]'), tag),
            "fc_%s.%s" % (b64(b'["mymarker","a?b"]'), tag),
            "fc_%s.%s" % (b64(b'["mymarker","%2e%2e"]'), tag),
            "fc_%s.%s" % (b64(b'["mymarker",""]'), tag),
            "fc_%s.%s" % (b64(b'["mymarker","%s"]' % (b"a" * 129)), tag),
            # Wrong tag length.
            "fc_%s.%s" % (b64(b'["mymarker","%s"]' % RID.encode()),
                b64(b"x" * 15)),
            "fc_%s.%s" % (b64(b'["mymarker","%s"]' % RID.encode()),
                b64(b"x" * 32)),
            # Padding is never part of an id.
            "fc_%s=.%s" % (b64(b'["mymarker","%s"]' % RID.encode()), tag),
            "fc_" + "A" * 600 + "." + tag,
        ]
        for job_id in cases:
            with self.subTest(job_id=job_id[:80]):
                self.assertIsNone(conversion_id.parse(job_id))

    def test_make_refuses_unsafe_request_id(self):
        for rid in ("", "a/b", "../x", "a" * 129, "a b", "ą"):
            with self.subTest(rid=rid), self.assertRaises(ValueError):
                conversion_id.make(TOKEN, "myuser", "mymarker", rid)

    def test_overlong_id_rejected_even_if_well_formed(self):
        # MAX_LENGTH is checked before decoding, not just implied by a
        # decoding failure: a correctly signed but huge id is refused too.
        job_id = conversion_id.make(TOKEN, "myuser", "m" * 400, RID)
        self.assertGreater(len(job_id), conversion_id.MAX_LENGTH)
        self.assertIsNone(conversion_id.parse(job_id))


class TestBackgroundConversion(LLMProxyAppTestCase):
    AUTH = {"Authorization": "Bearer mytoken"}

    async def submit(self, background="true", headers=None, **fields):
        form = aiohttp.FormData()
        form.add_field("model", fields.pop("model", "mymarker"))
        if background is not None:
            form.add_field("background", background)
        for key, value in fields.items():
            form.add_field(key, value)
        form.add_field("file", b"%PDF-1.4 synthetic", filename="doc.pdf",
            content_type="application/pdf")
        req = self.client.request("POST", "/v1/files/convert",
            headers=headers or self.AUTH, data=form)
        async with req as res:
            return res.status, dict(res.headers), _body(await res.text())

    async def poll(self, job_id, headers=None):
        req = self.client.request("GET", "/v1/files/convert/%s" % job_id,
            headers=self.AUTH if headers is None else headers)
        async with req as res:
            return res.status, dict(res.headers), _body(await res.text())

    async def queued(self, **fields):
        status, headers, data = await self.submit(**fields)
        self.assertEqual(status, 202, data)
        return data["id"]

    def backend_calls(self):
        return list(self.backend.app["paths"])

    async def test_submit_answers_202_with_id_and_bills_nothing(self):
        status, headers, data = await self.submit(_pages="5")

        self.assertEqual(status, 202)
        self.assertEqual(set(data), {"id", "object", "model", "status"})
        self.assertEqual(data["object"], "file.conversion")
        self.assertEqual(data["model"], "mymarker")
        self.assertEqual(data["status"], "queued")
        self.assertTrue(data["id"].startswith("fc_"))
        self.assertEqual(headers["Location"],
            "/v1/files/convert/%s" % data["id"])
        self.assertEqual(headers["Retry-After"], "5")
        self.assertIn("X-Request-ID", headers)
        self.assertNotIn("X-AI-Generated", headers)
        self.assertListEqual(await self.get_events(), [])

        form = self.backend.app["marker_forms"][-1]
        self.assertEqual(form["background"], "true")
        self.assertNotIn("model", form)

    async def test_background_spellings(self):
        for value, expected in (("true", "true"), ("TRUE", "true"),
                ("1", "true"), ("false", "false"), ("False", "false"),
                ("0", "false"), (None, "false")):
            with self.subTest(value=value):
                status, _, data = await self.submit(background=value)
                self.assertEqual(status, 202 if expected == "true" else 200,
                    data)
                self.assertEqual(
                    self.backend.app["marker_forms"][-1]["background"],
                    expected)

    async def test_invalid_background_is_400_before_backend(self):
        for value in ("yes", "", "2", "true ", "null"):
            with self.subTest(value=value):
                seen = len(self.backend_calls())
                status, _, data = await self.submit(background=value)
                self.assertEqual(status, 400)
                self.assertEqual(data["error"]["param"], "background")
                self.assertEqual(data["error"]["type"],
                    "invalid_request_error")
                self.assertEqual(self.backend_calls()[seen:], [])
        self.assertListEqual(await self.get_events(), [])

    async def test_sync_sends_background_false_and_answers_unchanged(self):
        status, headers, data = await self.submit(background=None,
            _pages="3")

        self.assertEqual(status, 200)
        self.assertEqual(data, {"success": True,
            "output": "# Converted\n\nhello", "images": {},
            "output_format": "markdown", "page_count": 3})
        self.assertEqual(self.backend.app["marker_forms"][-1]["background"],
            "false")
        self.assertListEqual(await self.get_events(), [
            {"product": "mymarker/none/conversion", "quantity": 3},
        ])

    async def test_poll_until_completed_then_billed_once(self):
        job_id = await self.queued(_polls="2", _pages="7")

        for _ in range(2):
            status, headers, data = await self.poll(job_id)
            self.assertEqual(status, 200)
            self.assertEqual(data, {"id": job_id, "object": "file.conversion",
                "model": "mymarker", "status": "in_progress"})
            self.assertEqual(headers["Retry-After"], "5")
            self.assertListEqual(await self.get_events(), [])

        status, headers, data = await self.poll(job_id)
        self.assertEqual(status, 200)
        self.assertEqual(data, {"id": job_id, "object": "file.conversion",
            "model": "mymarker", "status": "completed", "success": True,
            "output": "# Converted\n\nhello", "output_format": "markdown",
            "page_count": 7, "images": {"a.png": "aGk="},
            "provenance": {"ai_generated": True}})
        self.assertEqual(headers["X-AI-Generated"], "true")
        self.assertEqual(headers["X-AI-Processed"], "true")
        self.assertListEqual(await self.get_events(), [
            {"product": "mymarker/none/conversion", "quantity": 7},
        ])

        # Picking the result up again returns it, unbilled.
        for _ in range(3):
            status, _, again = await self.poll(job_id)
            self.assertEqual(status, 200)
            self.assertEqual(again, data)
        self.assertListEqual(await self.get_events(), [
            {"product": "mymarker/none/conversion", "quantity": 7},
        ])

        # Every poll carries the backend token and asks for the claim.
        for b_hdrs in self.backend.app["marker_polls"]:
            self.assertEqual(b_hdrs["Authorization"], "Bearer %s" % TOKEN)
            self.assertEqual(b_hdrs["X-Delivery-Claim"], "true")

    async def test_concurrent_first_pickups_bill_once(self):
        self.app["config"]["files"] = {"max_polls_in_flight": 12}
        job_id = await self.queued(_pages="4")
        before = metrics.FILE_CONVERSIONS_TOTAL.labels("mymarker")._value.get()

        results = await asyncio.gather(*(self.poll(job_id)
            for _ in range(12)))

        self.assertEqual([r[0] for r in results], [200] * 12)
        self.assertTrue(all(r[2]["status"] == "completed" for r in results))
        self.assertListEqual(await self.get_events(), [
            {"product": "mymarker/none/conversion", "quantity": 4},
        ])
        self.assertEqual(
            metrics.FILE_CONVERSIONS_TOTAL.labels("mymarker")._value.get(),
            before + 4)

    async def test_output_format_names_the_content_key(self):
        # The backend stores the text under output_format unless that would
        # shadow one of its own fields; the client always gets "output".
        for fmt in ("html", "MarkDown", "page_count", "delivery_claimed"):
            with self.subTest(fmt=fmt):
                job_id = await self.queued(output_format=fmt)
                status, _, data = await self.poll(job_id)
                self.assertEqual(status, 200, data)
                self.assertEqual(data["output"], "# Converted\n\nhello")
                self.assertEqual(data["output_format"], fmt)
                self.assertNotIn("delivery_claimed", data)

    async def test_failed_conversion_reported_and_not_billed(self):
        job_id = await self.queued(_poll_fail="1")

        status, headers, data = await self.poll(job_id)

        self.assertEqual(status, 200)
        self.assertEqual(data, {"id": job_id, "object": "file.conversion",
            "model": "mymarker", "status": "failed", "error": {
                "message": "PDF is encrypted", "code": "conversion_failed"}})
        self.assertNotIn("X-AI-Generated", headers)
        self.assertListEqual(await self.get_events(), [])

    async def test_job_gone_from_backend_is_404(self):
        job_id = await self.queued()
        self.backend.app["marker_jobs"].clear()

        status, _, data = await self.poll(job_id)

        self.assertEqual(status, 404)
        self.assertEqual(data["error"]["code"], "conversion_not_found")
        self.assertListEqual(await self.get_events(), [])

    async def test_other_api_key_gets_404_without_backend_call(self):
        await self.insert_key("othertoken", "otheruser")
        job_id = await self.queued()
        seen = len(self.backend_calls())

        status, _, data = await self.poll(job_id,
            headers={"Authorization": "Bearer othertoken"})

        self.assertEqual(status, 404)
        self.assertEqual(data["error"]["code"], "conversion_not_found")
        self.assertEqual(self.backend_calls()[seen:], [])
        self.assertListEqual(await self.get_events(), [])

        # The owner still gets (and pays for) the result.
        status, _, data = await self.poll(job_id)
        self.assertEqual(data["status"], "completed")
        self.assertEqual(len(await self.get_events()), 1)

    async def test_forged_ids_are_404_without_backend_call(self):
        job_id = await self.queued()
        rid = conversion_id.parse(job_id).request_id
        cases = [
            job_id[:-1],
            job_id[:-1] + ("A" if job_id[-1] != "A" else "B"),
            job_id.replace("fc_", "fc_A", 1),
            "fc_x.y", "%3F", "*", "fc_..",
            # Signed with the right token, but for another key.
            conversion_id.make(TOKEN, "otheruser", "mymarker", rid),
            # A chat backend's token signs nothing here.
            conversion_id.make(TOKEN, "myuser", "mymodel", rid),
            conversion_id.make("secret-backend-token", "myuser", "nolimit",
                rid),
            # Unknown backend.
            conversion_id.make(TOKEN, "myuser", "nosuchmodel", rid),
        ]
        seen = len(self.backend_calls())
        for forged in cases:
            with self.subTest(forged=forged):
                status, _, data = await self.poll(forged)
                self.assertEqual(status, 404)
                self.assertEqual(data["error"]["code"],
                    "conversion_not_found")
        self.assertEqual(self.backend_calls()[seen:], [])
        self.assertListEqual(await self.get_events(), [])

    async def test_poll_requires_auth(self):
        job_id = await self.queued()
        seen = len(self.backend_calls())

        for headers in ({}, {"Authorization": "Bearer wrongtoken"}):
            with self.subTest(headers=headers):
                status, _, _ = await self.poll(job_id, headers=headers)
                self.assertEqual(status, 401)
        self.assertEqual(self.backend_calls()[seen:], [])

    async def test_head_cannot_claim(self):
        job_id = await self.queued()
        seen = len(self.backend_calls())

        req = self.client.request("HEAD", "/v1/files/convert/%s" % job_id,
            headers=self.AUTH)
        async with req as res:
            self.assertEqual(res.status, 405)

        self.assertEqual(self.backend_calls()[seen:], [])
        self.assertListEqual(await self.get_events(), [])

    async def test_backend_without_background_support_is_502(self):
        # An old file-processor ignores background=true and converts
        # synchronously: there is no request id to hand out.
        status, _, _ = await self.submit(_no_request_id="1")
        self.assertEqual(status, 502)
        self.assertListEqual(await self.get_events(), [])

    async def test_backend_without_claim_never_delivers(self):
        job_id = await self.queued(_no_claim="1")

        for _ in range(2):
            status, _, data = await self.poll(job_id)
            self.assertEqual(status, 502)
            self.assertNotIn("hello", str(data))
        self.assertListEqual(await self.get_events(), [])

    async def test_bad_page_count_never_delivers(self):
        # The claim is spent on the first poll; a later poll must not hand
        # out the result for free either.
        job_id = await self.queued(_omit_page_count="1")

        for _ in range(2):
            status, _, data = await self.poll(job_id)
            self.assertEqual(status, 502)
            self.assertNotIn("hello", str(data))
        self.assertListEqual(await self.get_events(), [])

    async def test_backend_errors_on_poll(self):
        job_id = await self.queued()
        control = self.backend.app["marker_control"]

        control["poll_status"] = 500
        status, _, _ = await self.poll(job_id)
        self.assertEqual(status, 502)

        control["poll_status"] = 401
        status, _, data = await self.poll(job_id)
        self.assertEqual(status, 401)
        self.assertEqual(data, {"error": "boom"})

        del control["poll_status"]
        control["poll_delay"] = 2  # timeout_read is 1 s in tests
        status, _, _ = await self.poll(job_id)
        self.assertEqual(status, 504)
        self.assertListEqual(await self.get_events(), [])

    async def test_backend_unreachable_on_poll_is_502(self):
        job_id = await self.queued()
        await self.backend.close()

        status, _, _ = await self.poll(job_id)

        self.assertEqual(status, 502)

    async def test_polls_do_not_use_rate_limit(self):
        self.app["config"]["backends"]["mymarker"]["rate_limit"] = {
            "rpm": 1, "concurrency": 1}
        job_id = await self.queued(_polls="5")

        for _ in range(6):
            status, _, _ = await self.poll(job_id)
            self.assertEqual(status, 200)

    async def test_poll_metrics_use_route_template(self):
        job_id = await self.queued(_polls="1")
        counter = metrics.REQUESTS_TOTAL.labels("GET",
            "/v1/files/convert/{id}", "200")
        before = counter._value.get()

        await self.poll(job_id)

        self.assertEqual(counter._value.get(), before + 1)

    async def test_token_rotation_invalidates_open_ids(self):
        job_id = await self.queued()
        self.app["config"]["backends"]["mymarker"]["token"] = "rotated"

        status, _, data = await self.poll(job_id)

        self.assertEqual(status, 404)
        self.assertEqual(data["error"]["code"], "conversion_not_found")

    async def test_non_bool_delivery_claimed_never_delivers(self):
        # Only a JSON true bills; a truthy 1 or "false" string must not be
        # read as a claim, and the result is never handed out without one.
        for raw in ("1", "0", '"true"', '"false"', "null", "{}"):
            with self.subTest(delivery_claimed=raw), \
                    self.assertLogs(self.app.logger, level="ERROR"):
                job_id = await self.queued(_claim_raw=raw)
                for _ in range(2):
                    status, _, data = await self.poll(job_id)
                    self.assertEqual(status, 502)
                    self.assertNotIn("hello", str(data))
        self.assertListEqual(await self.get_events(), [])

    async def test_invalid_page_count_never_delivers(self):
        for raw in ("0", "-1", "true", "1.5", '"3"'):
            with self.subTest(page_count=raw), \
                    self.assertLogs(self.app.logger, level="ERROR"):
                job_id = await self.queued(_pages=raw)
                for _ in range(2):
                    status, _, data = await self.poll(job_id)
                    self.assertEqual(status, 502)
                    self.assertNotIn("hello", str(data))
        self.assertListEqual(await self.get_events(), [])

    async def test_unusable_backend_request_id_is_502(self):
        # The request id becomes a backend URL path segment on every poll.
        for raw in ('"a/b"', '"../healthz"', '"a?b"', '""',
                '"%s"' % ("a" * 129), '"\u0105"', "123", "null"):
            with self.subTest(request_id=raw), \
                    self.assertLogs(self.app.logger, level="ERROR"):
                status, headers, data = await self.submit(_request_id_raw=raw)
                self.assertEqual(status, 502, data)
                self.assertNotIn("Location", headers)
        self.assertListEqual(await self.get_events(), [])

    async def test_reserved_output_formats_never_shadow_the_text(self):
        # Literal copy of file-processor's _RESERVED_KEYS (datalab_compat.py),
        # independent of both the proxy's and the mock's sets: a key missing
        # from the proxy's list would hand out e.g. the status as the text.
        for fmt in ("status", "STATUS", "success", "images", "page_count",
                "output_format", "provenance", "delivery_claimed",
                "error_code", "error", "request_id"):
            with self.subTest(fmt=fmt):
                job_id = await self.queued(output_format=fmt)
                status, _, data = await self.poll(job_id)
                self.assertEqual(status, 200, data)
                self.assertEqual(data["output"], "# Converted\n\nhello")
                self.assertEqual(data["output_format"], fmt)

    async def test_result_never_reaches_another_key_or_the_logs(self):
        await self.insert_key("othertoken", "otheruser")
        other = {"Authorization": "Bearer othertoken"}
        marker = "MARKER-%s" % secrets.token_hex(8)

        with self.assertLogs(level="DEBUG") as logs:
            job_id = await self.queued(_text=marker, _polls="1")
            own_id = (await self.submit(headers=other))[2]["id"]
            refused_id = await self.queued(_text=marker, _no_claim="1")
            answers = []
            for job, hdrs in ((job_id, self.AUTH), (job_id, other),
                    (job_id, self.AUTH), (job_id, other), (own_id, other),
                    (refused_id, self.AUTH), (refused_id, other)):
                answers.append((job, hdrs, await self.poll(job, headers=hdrs)))

        owner_got = [a for _, h, a in answers
            if h is self.AUTH and a[0] == 200 and a[2]["status"] == "completed"]
        self.assertEqual([a[2]["output"] for a in owner_got], [marker])
        for job, hdrs, (status, res_headers, data) in answers:
            if hdrs is other:
                self.assertNotIn(marker, json.dumps(data))
                self.assertNotIn(marker, json.dumps(res_headers))
                if job != own_id:
                    self.assertEqual(status, 404)
        for line in logs.output:
            self.assertNotIn(marker, line)

    async def test_poll_counts_backend_metrics(self):
        job_id = await self.queued(_polls="1")
        counter = metrics.BACKEND_REQUESTS_TOTAL.labels("mymarker", "200")
        before = counter._value.get()

        await self.poll(job_id)
        await self.poll(job_id)

        self.assertEqual(counter._value.get(), before + 2)

    async def test_owner_keeps_access_after_reload_with_same_token(self):
        job_id = await self.queued(_pages="2")
        cfg = copy.deepcopy(self.app["config"])
        self.app["config"]["_path"] = "unused-test-path"
        with mock.patch("llmproxy.config.load", return_value=cfg):
            reload_config(self.app)
        # The reload swapped in fresh backend dicts (same token).
        self.assertIs(self.app["config"]["backends"], cfg["backends"])

        status, _, data = await self.poll(job_id)

        self.assertEqual(status, 200, data)
        self.assertEqual(data["status"], "completed")
        self.assertListEqual(await self.get_events(), [
            {"product": "mymarker/none/conversion", "quantity": 2},
        ])

    async def test_polls_in_flight_limited_per_api_key(self):
        await self.insert_key("othertoken", "otheruser")
        other = {"Authorization": "Bearer othertoken"}
        mine = await self.queued(_polls="10")
        theirs = (await self.submit(headers=other, _polls="10"))[2]["id"]
        self.backend.app["marker_control"]["poll_delay"] = 0.3

        for limit in (None, 1):
            with self.subTest(max_polls_in_flight=limit):
                if limit is not None:
                    self.app["config"]["files"] = {
                        "max_polls_in_flight": limit}
                allowed = limit or files.DEFAULT_MAX_POLLS_IN_FLIGHT
                seen = len(self.backend.app["marker_polls"])

                results = await asyncio.gather(
                    *(self.poll(mine) for _ in range(allowed + 1)),
                    self.poll(theirs, headers=other))

                statuses = sorted(r[0] for r in results[:-1])
                self.assertEqual(statuses, [200] * allowed + [429])
                status, headers, data = next(r for r in results if r[0] == 429)
                self.assertEqual(data["error"]["code"], "rate_limit_exceeded")
                self.assertEqual(headers["Retry-After"], "5")
                # Another key is not affected by this key's limit.
                self.assertEqual(results[-1][0], 200)
                self.assertEqual(results[-1][2]["status"], "in_progress")
                # The rejected poll never reached the backend.
                self.assertEqual(
                    len(self.backend.app["marker_polls"]) - seen, allowed + 1)
                # Every slot is released.
                self.assertEqual(files._polls_in_flight, {})

    async def test_poll_slot_released_on_errors(self):
        self.app["config"]["files"] = {"max_polls_in_flight": 1}
        job_id = await self.queued(_omit_page_count="1")
        for _ in range(3):
            status, _, _ = await self.poll(job_id)
            self.assertEqual(status, 502)
        self.backend.app["marker_control"]["poll_delay"] = 2
        status, _, _ = await self.poll(job_id)
        self.assertEqual(status, 504)
        self.assertEqual(files._polls_in_flight, {})

    async def test_oversized_poll_answer_is_502_and_not_billed(self):
        self.app["config"]["files"] = {"max_poll_response_bytes": 2000}
        job_id = await self.queued(_text="x" * 3000)

        for _ in range(2):
            with self.assertLogs(self.app.logger, level="ERROR") as logs:
                status, _, data = await self.poll(job_id)
            self.assertEqual(status, 502)
            self.assertNotIn("xxxx", str(data))
            self.assertIn("max_poll_response_bytes", logs.output[0])
        self.assertListEqual(await self.get_events(), [])

        # Within the limit the same kind of answer goes through.
        small = await self.queued(_text="x" * 500)
        status, _, data = await self.poll(small)
        self.assertEqual(status, 200)
        self.assertEqual(data["output"], "x" * 500)


class TestFilesConfig(unittest.TestCase):
    def test_files_options(self):
        config.validate({"files": {"max_polls_in_flight": 2,
            "max_poll_response_bytes": 1024}})
        for files_cfg in ({"max_polls_in_flight": 0},
                {"max_polls_in_flight": -1}, {"max_polls_in_flight": True},
                {"max_polls_in_flight": "4"},
                {"max_poll_response_bytes": 1.5}, {"max_poll_typo": 4},
                []):
            with self.subTest(files=files_cfg), \
                    self.assertRaises(config.ConfigError):
                config.validate({"files": files_cfg})
