FROM alpine:3.20.3

RUN apk add py3-aiohttp

COPY llm-billing-proxy.py /

CMD python3 llm-billing-proxy.py
