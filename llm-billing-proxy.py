import aiohttp.web
import yarl


# TODO: config file
BACKENDS = {
    "llama31-70b": {
        "url": yarl.URL("https://llm-server.comtegra.cgc-waw-01.comtegra.cloud"),
        "token": "MY-TOKEN1",
    },
    "llama3-sqlcoder-8b": {
        "url": yarl.URL("https://llm-server2.comtegra.cgc-waw-01.comtegra.cloud"),
        "token": "MY-TOKEN2",
    },
    "llama31-8b": {
        "url": yarl.URL("https://llm-server-bis.comtegra.cgc-waw-01.comtegra.cloud"),
        "token": "MY-TOKEN3",
    },
}

routes = aiohttp.web.RouteTableDef()


async def client_init(app):
    app["client"] = aiohttp.ClientSession()


# Frontend related variables are prefixed with f_.
# Backend related variables are prefixed with b_.
@routes.post("/v1/chat/completions")
async def chat(f_req):
    app = f_req.app

    f_body = await f_req.json()

    try:
        b_cfg = BACKENDS[f_body["model"]]
    except KeyError:
        return aiohttp.web.Response(status=400)

    b_url = b_cfg["url"] / str(f_req.rel_url)[1:]
    b_hdrs = {"Authorization": "Bearer %s" % b_cfg["token"]}

    f_res = aiohttp.web.StreamResponse()
    await f_res.prepare(f_req)

    async with app["client"].post(b_url, headers=b_hdrs, json=f_body) as b_res:
        if b_res.status != 200:
            return aiohttp.web.Response(status=b_res.status)

        async for chunk in b_res.content:
            await f_res.write(chunk)

    await f_res.write_eof()

    return f_res


if __name__ == "__main__":
    app = aiohttp.web.Application()
    app.on_startup.append(client_init)
    app.add_routes(routes)
    aiohttp.web.run_app(app)
