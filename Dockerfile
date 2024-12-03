FROM alpine:3.20.3

RUN apk add python3 py3-pip sqlite

COPY requirements.txt /
RUN pip install --break-system-packages -r requirements.txt && rm requirements.txt

COPY llmproxy /llmproxy

ENTRYPOINT ["python3", "-m", "llmproxy"]
