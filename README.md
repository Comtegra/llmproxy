# LLM Billing Proxy

HTTP proxy that sits between end users and LLM servers and bills them per token.

*Work in progress*

## Requirements

* aiohttp ~= 3.9.3

## Configuration

See [config.toml](config.toml) for an example configuration file.
The program will update the list of configured backends from the config file
on SIGHUP.

```sh
pkill -f -HUP 'python3? .*llm-billing-proxy\.py'

# or in Docker
docker kill -s=SIGHUP CONTAINER
```

## Development

Edit `config.toml` and configure backends, then run the proxy.

```sh
python3 llm-billing-proxy.py
```

Then, run a query.

```sh
curl -v -H'Content-Type: application/json' -H'Authorization: Bearer mykey1' \
    -d'{"messages": [{"role": "system", "content": "You are an assistant."}, {"role": "user", "content": "Write a limerick about python exceptions"}], "model": "llama31-70b", "stream": true}' \
    'http://localhost:8080/v1/chat/completions'
```

## Docker

```sh
docker build -t llm-billing-proxy .
docker run 
```
