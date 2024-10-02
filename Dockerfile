FROM alpine:3.20.3

RUN apk add python3 py3-pip

COPY requirements.txt /
RUN pip install --break-system-packages -r requirements.txt && rm requirements.txt

COPY llm-billing-proxy.py /

ENTRYPOINT ["python3", "llm-billing-proxy.py"]
