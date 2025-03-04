FROM alpine:3.20.3

RUN apk add \
	py3-aiohttp \
	py3-aiosqlite \
	py3-gpep517 \
	py3-installer \
	py3-mongo \
	py3-pip \
	py3-setuptools \
	py3-yarl \
	python3 \
	sqlite

# motor is not present in Alpine Linux package repositories
RUN pip install --break-system-packages motor==3.5.0

WORKDIR /root/llmproxy
COPY scripts scripts
COPY pyproject.toml .
COPY llmproxy llmproxy

RUN gpep517 build-wheel --wheel-dir .dist --output-fd 3 3>&1 >&2
RUN python3 -m installer .dist/llmproxy-*.whl

RUN rm -rf /root/llmproxy

WORKDIR /

ENTRYPOINT ["python3", "-m", "llmproxy"]
