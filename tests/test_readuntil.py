import unittest

import aiohttp
import aiohttp.test_utils

from llmproxy.chat import readuntil


class TestReaduntil(aiohttp.test_utils.AioHTTPTestCase):
    async def get_application(self):
        app = aiohttp.web.Application()
        app.add_routes([
            aiohttp.web.get("/1", self.create_chunk_handler([b"chunk1\n\n", b"chunk2\n\n"])),
            aiohttp.web.get("/2", self.create_chunk_handler([b"chunk1\n", b"\nchunk2\n\n"])),
            aiohttp.web.get("/3", self.create_chunk_handler([b"chunk1", b"\n\nchunk2\n\n"])),
            aiohttp.web.get("/4", self.create_chunk_handler([b"chunk1\n", b"\n", b"chunk2\n\n"])),
            aiohttp.web.get("/5", self.create_chunk_handler([b"chunk1", b"\n\n", b"chunk2\n\n"])),
            aiohttp.web.get("/6", self.create_chunk_handler([b"chunk1", b"\n", b"\nchunk2\n\n"])),
        ])
        return app

    @staticmethod
    def create_chunk_handler(chunks):
        async def handler(req):
            res = aiohttp.web.StreamResponse()
            res.enable_chunked_encoding()
            await res.prepare(req)
            for chunk in chunks:
                await res.write(chunk)
            return res

        return handler

    async def test_readuntil(self):
        for endpoint in ["/1", "/2", "/3", "/4", "/5", "/6"]:
            with self.subTest(endpoint=endpoint):
                async with self.client.request("GET", endpoint) as res:
                    self.assertEqual(await readuntil(res.content, b"\n\n"), b"chunk1\n\n")
                    self.assertEqual(await readuntil(res.content, b"\n\n"), b"chunk2\n\n")


if __name__ == '__main__':
    unittest.main()
