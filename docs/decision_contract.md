# Internal decision adapter contract

`decision_contract.py` defines the boundary between a decision engine and the
shared proxy service. It is an internal Python interface, not a new HTTP API.
It does not make an engine OpenAI compatible by itself.

## Responsibilities

- `DecisionAdapter` declares its backend path and `DecisionCapabilities`, and
  prepares a request under the selected backend's `decision_limits`.
- `PreparedDecision` owns the backend payload and request-specific validation
  state. It reports question and logical branch counts for metrics and limits.
  `decode()` validates a response against that request and returns a
  `DecisionResult`. It must not accept malformed or mismatched answers.
- `DecisionResult` exposes the model, named answers, actual `DecisionUsage`,
  and the native response. Answer semantics are adapter-specific; this contract
  does not pretend that every engine implements Basal's question types.
- `DecisionUsage` contains actual non-negative integer token counts. An adapter
  validates engine-specific requirements before returning it (Basal additionally
  requires positive input usage). Serializing a result is not text generation.
- `decisions.py` owns the authenticated HTTP flow, bounded reads, backend error
  handling, billing and metrics. It bills validated usage once. Invalid requests
  do not reach the backend; invalid responses are not billed.

Adapters raise `ValueError` for invalid client payloads or backend data. Decoder
error categories are `invalid_response`, `invalid_usage` and `invalid_answers`.
Malformed JSON and response size limits remain transport responsibilities.
Adapters must not send HTTP requests, write billing or store request state on
the shared adapter instance. Prepared payloads must not mutate caller input.

## Basal and configuration

The initial implementation is `basal.py`; it preserves the existing six native
question types (basal API 2.0), workload counting, calibrated answer validation and wire format.

```toml
[backends.basal-20-max]
type = "decision"
decision_adapter = "basal"
url = "https://your-basal-service.example"
model = "Remek/basal-2.0-max-GGUF"
device = "a5000"
```

`decision_adapter` defaults to `basal` for existing configurations. Unknown
adapters and use of this option with another backend type fail configuration
validation. Engine/model names do not select wire protocols implicitly.

`POST /v1/systemone` is the native entry point. It uses the selected
adapter and preserves native response bytes. `/v1/decisions`
is a separate OpenAI Decisions frontend; it does not accept native payloads.

## Adding another engine

Implement `DecisionAdapter` and a per-request `PreparedDecision`, register the
adapter in `decision_adapters.py`, and configure its name. Tests must cover its
request/answer validation, declared capabilities and limits. A local HTTP test
must verify the backend path, credentials, error handling and actual billed
usage. No conditional on a model name belongs in the shared HTTP handler.

`decision_openai.py` implements the supported OpenAI Decisions subset:
text input and predicate/choice/score questions. It translates the request to
Basal's native wire format and translates validated answers back to the public
Decisions schema before billing. It explicitly checks that the selected wire
adapter is Basal; adding a new engine to the internal registry does not imply
that it supports this frontend. Other adapters are not advertised as OpenAI
Decisions-compatible in the model catalog.

`decisions.respond()` provides bounded response reads, native validation,
rendering before billing, and shared accounting for both frontends. Neither
frontend calls the other HTTP handler recursively. There is no Basal adapter
for Chat Completions: `chat.py` accepts `type = "chat"` backends only.
