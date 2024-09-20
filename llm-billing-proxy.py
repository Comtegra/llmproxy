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


@routes.post("/v1/chat/completions")
async def chat(req):
    body = await req.json()

    try:
        backend = BACKENDS[body["model"]]
    except KeyError:
        return aiohttp.web.Response(status=400)

    url = backend["url"] / str(req.rel_url)[1:]
    headers = {"Authorization": "Bearer %s" % backend["token"]}

    async with req.app["client"].post(url, headers=headers, json=body) as res:
        if res.status != 200:
            return aiohttp.web.Response(status=res.status)

        return aiohttp.web.json_response(await res.json())


if __name__ == "__main__":
    app = aiohttp.web.Application()
    app.on_startup.append(client_init)
    app.add_routes(routes)
    aiohttp.web.run_app(app)
