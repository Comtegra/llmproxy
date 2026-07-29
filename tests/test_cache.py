"""Tests for the performance groundwork: process-global Mongo client (§16.1)
and the in-process auth cache (§16.2).

The auth cache is the one deliberate behavioural change in the rate-limit plan
-- it delays REVOCATION by up to auth_cache_ttl seconds -- so the tests here
pin both halves of that trade: that a hit really does avoid the database, and
that key EXPIRY is still exact despite the cache.
"""

import datetime
import unittest
import unittest.mock

import aiohttp.web

from llmproxy import auth, db


class _FakeDb:
    """Counts user_list calls so a test can prove a lookup was served from
    memory rather than from the database."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    async def user_list(self, digest, include_expired=False):
        self.calls += 1
        return list(self.rows)

    async def close(self):
        pass


class _App(dict):
    logger = unittest.mock.MagicMock()


class _Req(dict):
    """Request stub carrying just what require_auth touches: .app and
    .headers, plus dict access for the per-request db slot."""

    def __init__(self, app, headers):
        super().__init__()
        self.app = app
        self.headers = headers


def _req(token="secret", ttl=5):
    app = _App(config={"db": {"uri": "sqlite://:memory:"},
        "auth_cache_ttl": ttl})
    return _Req(app, {"Authorization": "Bearer %s" % token})


def _row(expires=None):
    return {"id": "u1", "secret": "abc", "expires": expires,
        "status": "active", "comment": ""}


class TestAuthCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        auth.flush_cache()

    def tearDown(self):
        auth.flush_cache()

    async def _auth(self, fake, **kw):
        req = _req(**kw)
        with unittest.mock.patch.object(auth, "get_db",
                unittest.mock.AsyncMock(return_value=fake)):
            return await auth.require_auth(req)

    @staticmethod
    def _digest(token="secret"):
        import hashlib
        return hashlib.sha256(token.encode()).hexdigest()

    async def test_hit_avoids_database(self):
        fake = _FakeDb([_row()])

        first = await self._auth(fake)
        second = await self._auth(fake)

        self.assertEqual(first["id"], "u1")
        self.assertEqual(second["id"], "u1")
        # Second lookup served from memory: no additional query.
        self.assertEqual(fake.calls, 1)

    async def test_disabled_ttl_always_queries(self):
        fake = _FakeDb([_row()])

        await self._auth(fake, ttl=0)
        await self._auth(fake, ttl=0)

        self.assertEqual(fake.calls, 2)

    async def test_ttl_expiry_forces_requery(self):
        fake = _FakeDb([_row()])

        await self._auth(fake)
        self.assertEqual(fake.calls, 1)

        # Age the entry out by rewriting its deadline into the past, rather
        # than patching the clock (which would leak into every other module).
        row, expires_at, _ = auth._cache[self._digest()]
        auth._cache[self._digest()] = (row, expires_at, 0.0)

        await self._auth(fake)
        self.assertEqual(fake.calls, 2)

    async def test_key_expiring_during_ttl_is_rejected(self):
        """The point of re-checking expires in Python: a key that expires while
        its row is still cached must be rejected immediately, not at TTL end.

        Seeded directly with a passed expiry and a still-live TTL -- the exact
        state a key reaches partway through its cache entry's lifetime."""
        past = datetime.datetime.now(datetime.UTC) \
            - datetime.timedelta(seconds=1)
        auth._cache_put(self._digest(), _row(), past, 300,
            auth.time.monotonic())

        fake = _FakeDb([_row()])
        with self.assertRaises(aiohttp.web.HTTPUnauthorized):
            await self._auth(fake)

        # Rejected without going back to the database...
        self.assertEqual(fake.calls, 0)
        # ...and the dead entry is dropped rather than re-checked every request.
        self.assertNotIn(self._digest(), auth._cache)

    async def test_naive_and_aware_expires_both_normalize(self):
        past = datetime.datetime.now(datetime.UTC) \
            - datetime.timedelta(hours=1)

        # SQLite hands back an ISO string; Mongo hands back a datetime.
        for value in (past.replace(tzinfo=None).isoformat(), past.isoformat(),
                past):
            auth.flush_cache()
            with self.subTest(value=value):
                self.assertLess(auth._normalize_expires(value),
                    datetime.datetime.now(datetime.UTC))

    async def test_unparseable_expires_is_not_cached(self):
        fake = _FakeDb([_row(expires="not-a-date")])

        await self._auth(fake)
        await self._auth(fake)

        # Never cached, so every request re-queries -- fail safe, not fail open.
        self.assertEqual(fake.calls, 2)

    async def test_negative_result_is_cached(self):
        fake = _FakeDb([])

        for _ in range(3):
            with self.assertRaises(aiohttp.web.HTTPUnauthorized):
                await self._auth(fake)

        self.assertEqual(fake.calls, 1)

    async def test_negative_cache_is_bounded(self):
        fake = _FakeDb([])

        for i in range(auth.CACHE_MAX_ENTRIES + 50):
            with self.assertRaises(aiohttp.web.HTTPUnauthorized):
                await self._auth(fake, token="bad-%d" % i)

        self.assertLessEqual(len(auth._cache), auth.CACHE_MAX_ENTRIES)

    async def test_flush_clears(self):
        fake = _FakeDb([_row()])

        await self._auth(fake)
        auth.flush_cache()
        await self._auth(fake)

        self.assertEqual(fake.calls, 2)

    async def test_caller_mutation_does_not_poison_cache(self):
        fake = _FakeDb([_row()])

        first = await self._auth(fake)
        first["id"] = "tampered"
        second = await self._auth(fake)

        self.assertEqual(second["id"], "u1")

    async def test_bad_scheme_never_touches_database(self):
        fake = _FakeDb([_row()])
        req = _req()
        req.headers = {"Authorization": "Basic whatever"}

        with unittest.mock.patch.object(auth, "get_db",
                unittest.mock.AsyncMock(return_value=fake)):
            with self.assertRaises(aiohttp.web.HTTPUnauthorized):
                await auth.require_auth(req)

        self.assertEqual(fake.calls, 0)


class TestMongoClientReuse(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        db._mongo_clients.clear()

    async def test_same_uri_returns_same_instance(self):
        made = []

        async def fake_connect(cls, uri):
            inst = db.MongoDatabase()
            inst.db = unittest.mock.MagicMock()
            made.append(inst)
            return inst

        with unittest.mock.patch.object(db.MongoDatabase, "_connect",
                classmethod(fake_connect)):
            a = await db.MongoDatabase.create("mongodb://x")
            b = await db.MongoDatabase.create("mongodb://x")
            c = await db.MongoDatabase.create("mongodb://y")

        self.assertIs(a, b)
        self.assertIsNot(a, c)
        self.assertEqual(len(made), 2)

    async def test_per_request_close_is_noop_shutdown_is_not(self):
        inst = db.MongoDatabase()
        inst.db = unittest.mock.MagicMock()
        db._mongo_clients["mongodb://x"] = inst

        # close() runs after every request via the close_db middleware.
        await inst.close()
        inst.db.close.assert_not_called()
        self.assertIn("mongodb://x", db._mongo_clients)

        await db.shutdown_all()
        inst.db.close.assert_called_once()
        self.assertEqual(db._mongo_clients, {})


if __name__ == "__main__":
    unittest.main()
