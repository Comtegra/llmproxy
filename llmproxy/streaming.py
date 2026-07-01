import contextlib

import aiohttp.web


async def readuntil(stream, separator: bytes = b"\n") -> bytes:
    result = bytearray()
    while not result.endswith(separator):
        b = await stream.read(1)
        if not b:
            break
        result += b

    return result


async def drain(read_block, write_block, on_chunk, on_disconnect=None,
        disconnected=False):
    """Forward SSE blocks from a backend to the client without ever losing usage.

    Reads blocks via ``read_block()`` until it returns ``b""`` (backend EOF),
    feeding every block to ``on_chunk`` and forwarding it via ``write_block``.
    If the client goes away (``write_block`` raises ``OSError``) we STOP writing
    but KEEP reading and feeding ``on_chunk`` — so a mid-stream disconnect never
    costs us the usage, and therefore the billing. This is the billing moat.

    ``read_block``/``write_block`` are injected, keeping this pure w.r.t. aiohttp
    and unit-testable without a real connection. Pass ``disconnected=True`` to
    start already disconnected (e.g. the client vanished during the header
    flush): we then drain the backend for usage without writing.
    """
    while (chunk := await read_block()):
        on_chunk(chunk)
        if disconnected:
            continue
        try:
            await write_block(chunk)
        except OSError as e:
            disconnected = True
            if on_disconnect is not None:
                on_disconnect(e)


async def stream_through(f_req, b_res, on_chunk):
    """Wire a chunked backend response to the client, invoking ``on_chunk`` for
    every SSE block (including during the post-disconnect drain). Returns the
    prepared client ``StreamResponse``.

    ``on_chunk`` is a per-format usage accumulator, so the same pump/drain logic
    serves chat, Anthropic messages and OpenAI responses.
    """
    app = f_req.app
    headers = {"Content-Type":
        b_res.headers.get("Content-Type", "application/octet-stream")}
    headers["X-Request-ID"] = str(f_req["request_id"])
    f_res = aiohttp.web.StreamResponse(headers=headers)

    f_res.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    if o := app["config"].get("http_origin"):
        f_res.headers["Access-Control-Allow-Origin"] = o

    # Prepare (flush) response headers. If the client is already gone, prepare()
    # raises OSError — but we still MUST drain the backend to capture usage, so
    # treat a prepare-time disconnect as an immediate disconnected state instead
    # of letting it abort billing (matches the pre-refactor handle_resp_stream,
    # which wrapped prepare in the same try/except as the write loop).
    disconnected = False
    try:
        await f_res.prepare(f_req)
    except OSError as e:
        app.logger.info("Client disconnected: %s", e)
        disconnected = True

    async def read_block():
        return await readuntil(b_res.content, b"\n\n")

    await drain(read_block, f_res.write, on_chunk,
        on_disconnect=lambda e: app.logger.info("Client disconnected: %s", e),
        disconnected=disconnected)

    with contextlib.suppress(OSError):
        await f_res.write_eof()

    return f_res
