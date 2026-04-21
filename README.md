# LLM Billing Proxy

HTTP proxy that sits between end users and LLM servers and bills them per token.

## Requirements

* Python >=3.11
* Python packages from [pyproject.toml](pyproject.toml) (dependencies)

## Quick start

```sh
python3 -m venv venv
source venv/bin/activate

sqlite3 db.sqlite < llmproxy/schema.sql

python3 -m llmproxy
```

## Configuration

See [llmproxy/config.toml](llmproxy/config.toml) for an example configuration file.
The program will update the list of configured backends from the config file
on SIGHUP.

```sh
pkill -f -HUP 'python3? .*llmproxy'

# or in Docker
docker kill -s=SIGHUP CONTAINER
```

### Backend configuration

All backends share a single `[backends.<alias>]` schema regardless of type —
the `type` field discriminates. The flat layout keeps aliases globally unique
since they are what clients send as `model` in their requests (e.g.
`/v1/chat/completions`). Field reference:

| Field | Required | Description |
|---|---|---|
| `url` | yes | Backend service URL |
| `token` | yes | Bearer token for backend auth |
| `device` | yes | GPU name; used in billing product strings |
| `model` | yes | Full HuggingFace path (e.g. `Qwen/Qwen3-8B`); exposed in response as `source_model` |
| `type` | yes | One of `chat`, `embedding`, `audio` |
| `quantization` | no | Free-form string (e.g. `FP8`, `Q4_K_M`) |
| `context_length` | no | Integer; auto-discovered from vLLM/SGLang `/v1/models` if absent. Not applicable to `type=audio` |
| `model_url` | no | Overrides the auto-generated HuggingFace model card URL; exposed in response as `card_url` |
| `verify_ssl` | no | Default `true` |

At startup the proxy probes each non-audio backend's `/v1/models` endpoint in
parallel and caches `max_model_len`. An explicit `context_length` in TOML wins
over discovery; if the two disagree, the proxy logs a warning so stale TOML
overrides don't go unnoticed. Discovery does not re-run on SIGHUP — restart
the proxy to pick up a new `max_model_len`.

### Listing models

```sh
curl -H'Authorization: Bearer token' http://localhost:8080/v1/models
```

Sample response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-8b",
      "object": "model",
      "created": null,
      "owned_by": null,
      "type": "chat",
      "source_model": "Qwen/Qwen3-8B",
      "card_url": "https://huggingface.co/Qwen/Qwen3-8B",
      "quantization": "FP8",
      "context_length": 131072
    }
  ]
}
```

`id` is the alias clients pass back as `model`. `context_length` is omitted
for `audio` backends and when neither TOML nor discovery provides a value.
`quantization` and (optional) other fields are included only when set.

## Testing

```sh
python3 -m unittest
```

## Building the image

To build the Docker image, run the following command:

```sh
docker build -t ghcr.io/comtegra/llmproxy:master .
```

## Deployment

The following instructions assume that you already have a Docker compose
repository and file.

### SQLite

1. Create an SQLite database according to
   [llmproxy/schema.sql](llmproxy/schema.sql).

```sh
sqlite3 db.sqlite < llmproxy/schema.sql
```

2. Copy [llmproxy/config.toml](llmproxy/config.toml) to your repository. A good
relative path would be `secrets/llm-billing-proxy-config.toml`.
3. Add appropriate entries to your compose file's `services` and `secrets`
sections. There's a template [compose.yml](compose.yml) in this repository.
4. Bring the new service up and verify it started correctly
(e.g `docker compose up llm-billing-proxy`).

### MongoDB

1. Create a MongoDB user with the following privileges (see section
[Database](#database) below for JS snippets):
  * `{resource: {db: "cgc", collection: "api_keys"}, actions: ["find"]}`
  * `{resource: {db: "billing", collection: "events_oneoff"}, actions: ["insert"]}`
2. Copy [llmproxy/config.toml](llmproxy/config.toml) to your repository. A good
relative path would be `secrets/llm-billing-proxy-config.toml`.
3. In the config edit `uri` in section `db` so that it includes credentials for
the Mongo user (e.g. `mongodb://myuser:mypass@host:27017/?authSource=cgc`).
4. Add appropriate entries to your compose file's `services` and `secrets`
sections. There's a template [compose.yml](compose.yml) in this repository.
5. Bring the new service up and verify it started correctly
(e.g `docker compose up llm-billing-proxy`).

## Testing

Sample data can be added to an SQLite database by using [sample.sql](sample.sql).
Ditto for MongoDB in [sample.js](sample.js).

After loading sample data you may run a query like this:

```sh
curl -v -H'Content-Type: application/json' -H'Authorization: Bearer token2' \
    -d'{"messages": [{"role": "system", "content": "You are an assistant."}, {"role": "user", "content": "Write a limerick about python exceptions"}], "model": "llama31-70b", "stream": true}' \
    'http://localhost:8080/v1/chat/completions'
```

## Database

If using SQLite see [llmproxy/schema.sql](llmproxy/schema.sql).
Otherwise read on.

This program uses MongoDB for authentication and completion logging.
The schema is compatible with Comtergra GPU Core.
The MongoDB user needs the following privileges:

```js
db.getSiblingDB("cgc").createRole({
  role: "apiKeysReader",
  privileges: [
    {
      resource: {db: "cgc", collection: "api_keys"},
      actions: ["find"],
    },
  ],
  roles: [],
});
db.getSiblingDB("billing").createRole({
  role: "completionBillingWriter",
  privileges: [
    {
      resource: {db: "billing", collection: "events_oneoff"},
      actions: ["insert"],
    },
  ],
  roles: [],
});

db.getSiblingDB("cgc").createUser({
  user: "llm-billing-proxy",
  pwd: "mypass",
  roles: [
    { role: "completionBillingWriter", db: "cgc" },
    { role: "apiKeysReader", db: "billing" },
  ],
});
```

## Authentication

Users authenticate via bearer tokens.
Tokens are stored in the database as SHA256 hashes.
They can be generated by the following command:

```sh
python3 -c 'import hashlib, secrets; print("Token:", t:=secrets.token_urlsafe(64)); print("Hash:", hashlib.sha256(t.encode()).hexdigest())'
```

If using SQLite see [llmproxy/schema.sql](llmproxy/schema.sql).
Otherwise read on.

When a user attempts to authenticate, the following query is performed to the
`cgc.api_keys` collection.

```js
{
    "access_level": "LLM",
    "secret": TOKEN-HASH,
    "$or": [
        {"date_expiry": {"$gt": new Date()}},
        {"date_expiry": null},
    ],
}
```

`TOKEN-HASH` is replaced with the SHA256 hash of the bearer token.

The documents are required to have an additional field: `user_id`.

## Completion logging

If using SQLite see [llmproxy/schema.sql](llmproxy/schema.sql).
Otherwise read on.

When a user performs a completion, two events (prompt token and completion
token counts) are inserted into the collection `billing.events_oneoff`.
These documents have the following fields:

* `date_created` -- date and time when the request finished processing
* `user_id` -- `user_id` of the API key
* `api_key_id` -- id of the API key
* `product` -- a string in the following format: `MODEL/DEVICE/TYPE`, where
    * `MODEL` is the name of the backend
    * `DEVICE` is the name of the GPU where the model runs
    * `TYPE` is `prompt` or `completion`
* `quantity` -- token count
* `request_id` -- request ID to correlate prompt/completion counts
