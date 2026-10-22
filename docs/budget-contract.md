# Output-budget contract (budget.v1)

This implements the corrected scope of the HTML architecture proposal. It
separates observation, request policy and backend capabilities. Billing products,
prices and the existing two token events are unchanged. It never retries or
continues generations automatically.

## Activation and configuration

The feature is **off by default**. Enable `[budget].enabled = true` and add
`output_budget` to a chat backend. With `mode = "observe"` the proxy changes no
output/reasoning limits, even if `apply_default`, `clamp` or `native_budget` are
configured. Existing transforms still run: model alias rewriting, streaming
usage inclusion and stateless Responses enforcement.

```toml
[budget]
enabled = true
echo_headers = true
echo_body = true

[backends.example.output_budget]
mode = "observe"
default_max_tokens = 4096
ceiling = 8192
apply_default = false
clamp = false
```

Switch the backend to `mode = "enforce"` to activate the independently selected
`apply_default` and `clamp` policies. A configured default must not exceed the
ceiling, and a ceiling must not exceed `max_model_len` when provided. All token
values must be positive integers (booleans are rejected). Invalid config/reload
is rejected; the previous config survives. `[budget]` and backend tables reload
on SIGHUP. Requests retain their budget config snapshot across a reload.

`echo_body = false` removes the extra JSON object; `echo_headers = false` removes
the budget headers. Neither disables policy or telemetry. The catalog remains
available. `enabled = false` disables all budget policy, catalog extensions,
response extensions and per-request budget telemetry. Existing provenance is
controlled separately; it can still reserialize non-streaming responses.

## Request policy

| Endpoint | Accepted output limit | Injected default field |
| --- | --- | --- |
| `/v1/chat/completions` | `max_tokens` or `max_completion_tokens` | `max_completion_tokens` |
| `/v1/completions` | `max_tokens` | `max_tokens` |
| `/v1/responses` | `max_output_tokens` | `max_output_tokens` |
| `/v1/messages` | `max_tokens` | `max_tokens` |

Observation forwards invalid/conflicting limits unchanged for backend validation;
the forwarded limit is unknown for ambiguous requests. Enforcement rejects
nonpositive/noninteger limits and both chat limit fields together with 400 before
forwarding or billing. Omitted limits are injected only with `apply_default = true`.
Explicit limits are clamped only with `clamp = true`. The original positive
requested value, forwarded value and whether clamping happened are reported.

The ceiling is a per-candidate output allowance, not a total cost cap when `n > 1`.
It is also not the remaining context window: prompt/template/multimodal tokens
consume context, and the proxy does not tokenize them. **Backend validation errors
are preserved, including context errors; clamping does not guarantee HTTP 200.**

## Catalog and echo

Authenticated `GET /v1/models` and `GET /v1/models/{id}` share identical items.
Unknown IDs return the existing JSON `model_not_found` error. Dynamic route
metrics use `/v1/models/{id}`, never individual requested IDs.

Enabled/configured backends advertise `extensions: ["budget.v1"]` and an
`output_budget` object with policy mode, default (null unless actually injected),
ceiling, clamp status and reasoning capability settings. An optional
`backend_default_max_tokens` is an operator-configured note, never a guaranteed
effective allowance. No flat `max_output_tokens` alias is invented for model
listings: client adoption of metadata must be tested explicitly.

Non-streaming responses can contain:

```json
{
  "llmproxy": {
    "version": 1,
    "budget": {
      "mode": "enforce",
      "parameter": "max_completion_tokens",
      "requested_max_output_tokens": 999999,
      "forwarded_max_output_tokens": 8192,
      "effective_max_output_tokens": null,
      "source": "client",
      "clamped": true
    },
    "truncation": {
      "occurred": true,
      "reason": "unknown",
      "backend_stop_reasons": [{"index": 0, "reason": "length"}],
      "details_available": true,
      "details_status": "present",
      "reasoning_tokens": 100,
      "non_reasoning_output_tokens": 20
    }
  }
}
```

`source` is `client`, `proxy`, or `unknown`. **Effective is always null**: knowing
what was forwarded does not establish the backend's sampling/context limit.
When the default is not injected and no unambiguous client limit exists, the
forwarded limit is null and `X-LLMProxy-Max-Output-Tokens` is omitted.

Budget headers (when enabled):

* `X-LLMProxy-Extensions: budget.v1`
* `X-LLMProxy-Max-Output-Tokens`: the **forwarded** allowance, when known
* `X-LLMProxy-Budget-Source: client|proxy|unknown`
* `X-LLMProxy-Budget-Clamped: true|false`
* `X-LLMProxy-Truncation: none|unknown` for JSON, `pending` for SSE

Both CORS paths expose these headers. Provenance and budget metadata share one
JSON serialization. Serialization failures retain original backend bytes and
headers; optional diagnostics cannot prevent billing.

## Truncation and reasoning usage

Stop signals are retained per choice: `length` (chat/completions), `max_tokens`
(messages), `max_output_tokens` (incomplete Responses). No cause is guessed from
aggregate counts or configured context length: `reason` remains `unknown` for
truncation, null for a known nontruncated stop. If no terminal signal is observed,
`occurred` is null. Standard backend fields are never rewritten.

Usage details are accepted only when integer counts satisfy `0 <= reasoning <=
output`. Missing/invalid details produce null partition counts. A reported zero
reasoning count alongside observed reasoning text is marked `inconsistent` and
also yields null counts. `non_reasoning_output_tokens` is **not** a measure of
visible final prose; it may include tool calls, delimiters and formatting.

Streaming bytes are forwarded untouched, with no synthetic SSE events. Budget
headers are flushed before the first event, so final diagnostics go to logs
correlated by `request_id` and Prometheus after billing commits. The observer
retains bounded stop/usage state, not reasoning text. Disconnects continue to drain
the backend and capture billing plus diagnostics.

Metrics: `llmproxy_budget_source_total{model,source,clamped}`,
`llmproxy_truncation_total{model,reason,stream}`,
`llmproxy_reasoning_tokens_total{model}` and
`llmproxy_reasoning_details_total{model,availability}`. Availability is `present`,
`inconsistent`, or `missing_or_invalid`. Models come from configured aliases;
other labels are fixed enums. Counters describe successful billed requests, not
all attempted requests. For reasoning share, filter completion totals to the
same enabled models/time periods and account for missing details.

The Grafana overview includes truncation rate and reasoning-detail coverage
panels. Example Prometheus rules alert on sustained truncation with a minimum
request volume, and on inconsistent reasoning counts. Missing details alone are
not an alert: they are expected on the verified Qwen deployment's chat endpoint.

## Native reasoning policy

Native control is supported **only on `/v1/chat/completions`**, after deployment
verification. Requests on the other three endpoints keep reasoning parameters
unchanged. Configuring other native endpoints is rejected rather than advertising
a control that may be ignored. The policy does not disable backend-native
Responses `reasoning.effort` or Anthropic `thinking` passthrough.

```toml
[backends.example.reasoning]
native_budget = true
endpoints = ["chat"]
reserve_content_tokens = 1024
effort_budgets = { low = 256, medium = 1024, high = 2048 }
# default_effort = "low" # optional, omit for no effort default
```

In enforce mode an explicit positive `thinking_token_budget` takes precedence
over the configured `reasoning_effort` map. A client-selected smaller native
budget is preserved. Otherwise the map uses the client's effort or configured
default. If reserve and a known forwarded output limit exist, the native budget
is capped at `forwarded - reserve` (or set to that value if no other budget was
selected). Reserve alone cannot operate against an unknown backend default.
An output limit at/below reserve is rejected with 400 rather than silently
changing the client's reasoning mode. `reasoning_effort=none` and explicit
`enable_thinking=false` leave reasoning unchanged. Existing template kwargs are
preserved. No mapping is inferred from template variable names.

This bounds reasoning only when the backend actually honors its sampling
parameter/parser boundaries; transition tokens, context exhaustion and output
quality still matter. It does not guarantee useful final content on every task.

## Reasoning effort dialects (`effort_enum`) — GLM-5.3-Flash

Some backends have **no token-level thinking budget at all**. GLM-5.3-Flash's
chat template controls reasoning only through the enum kwargs
`reasoning_effort` (`'low'` and `'high'`; anything else — including an
omitted value — maps to `'max'`) and `clear_thinking`, accepted via the
top-level `reasoning_effort` field or `chat_template_kwargs.reasoning_effort`.
Live check F01 (vLLM 0.31.0, direct probes) established the opposite for the
top-level parameter: **a top-level `thinking_token_budget` BINDS natively on
flash** (ttb=16 → `reasoning_tokens` exactly 16), while the
`chat_template_kwargs.thinking_token_budget` channel does **not** bind
(F01: control 23 / top-level 16 / kwargs 23). In dialect mode the proxy
forwards the client's `thinking_token_budget` verbatim — so the native
binding takes precedence (E05: reasoning exactly 16); the enum stays the
effort-shaped control and remains an approximation.

The `effort_enum` dialect keeps the entire native computation and changes
only the final injection:

```toml
[backends.glm-53f.reasoning]
native_budget = true          # required: the token-budget computation IS the native path
endpoints = ["chat"]
reserve_content_tokens = 64   # still active: the 400-edge is preserved
effort_budgets = { low = 64, high = 1024 }   # ladder thresholds; no medium on flash
dialect = "effort_enum"       # default "native" = today's behavior, byte-identical
```

`dialect = "effort_enum"` requires `native_budget = true` and a non-empty
`effort_budgets` table; anything other than `native`/`effort_enum` is
rejected. `native_budget = true` stays REQUIRED with
`dialect = "effort_enum"`: it gates the token-budget computation
(precedence, reserve math, the 400-edges and the metadata). Phase 2
therefore runs BOTH live paths — the enum for effort selection, and the
native binding for an exact client `thinking_token_budget`. The proxy
computes the token budget **exactly as in native mode**
(same precedence, same reserve math, same 400-edges) and then maps it onto the
highest configured effort level whose cap fits the budget:

```python
ordered = sorted(effort_budgets.items(), key=lambda kv: kv[1])
chosen = ordered[0][0]                       # floor = smallest configured effort
for name, cap in ordered:
    if cap <= budget:
        chosen = name
```

The selected effort is injected on **both channels** —
`reasoning_effort` top-level AND `chat_template_kwargs.reasoning_effort`
(dict-merged with any sibling kwargs; a client that already speaks the
dialect through `chat_template_kwargs.reasoning_effort` is honored as the
effort input when the top-level field is absent, and a top-level field wins
over a disagreeing kwargs value). `thinking_token_budget` is never written in
dialect mode; a client's own field is forwarded verbatim — and, being
top-level, it BINDS natively on flash (E05: reasoning exactly 16). The
proxy never injects a value outside the configured effort keys
(never `max`).

**The enum is an approximation: the model may think longer or shorter than
the level implies; there is NO exact-token guarantee for the ENUM in
dialect mode.** A client `thinking_token_budget` still binds natively on
flash: in dialect mode it is forwarded verbatim, and the native binding
takes precedence over the selected effort (E05: reasoning exactly 16).
`reserve_content_tokens` keeps exactly two hard guarantees:
(1) an output limit at/below reserve is rejected with 400 (unchanged edge),
and (2) the computed budget — hence the selected effort — is capped at
`forwarded - reserve`. Degradation is intended behavior: an effort whose cap
does not fit the affordable budget maps DOWN to the highest fitting level
(e.g. requested `high` with only 960 affordable tokens maps to `low`), which
protects the content budget instead of honoring an unaffordable effort.

Precedence (chat only, enforce only):

| Input | Effect |
| --- | --- |
| client `thinking_token_budget` (positive int) | selects the budget; the ladder then picks the level (client wins over any effort) |
| `reasoning_effort` (top-level) / `chat_template_kwargs.reasoning_effort` when the top-level field is absent | `effort_budgets[effort]` (or `default_effort`) selects the budget |
| none of the above | the budget is `forwarded - reserve` (pure reserve math) |
| then, always | the budget is capped at `forwarded - reserve`; a limit at/below reserve is a 400 |

Dialect metadata is reported on chat only, in `llmproxy.budget.reasoning`:

```json
{
  "dialect": "effort_enum",
  "parameter": "reasoning_effort",
  "requested_tokens": 16,
  "requested_effort": "low",
  "forwarded_tokens": 16,
  "forwarded_effort": "low",
  "source": "client",
  "reserve_content_tokens": 64
}
```

`forwarded_tokens` is the computed budget that SELECTED the effort level —
it is not a parameter forwarded to the backend. `source` keeps the native
semantics (`client` only for an explicit client `thinking_token_budget`,
else `proxy`). The catalog's `reasoning` block always carries a `dialect`
key (`"native"` by default); `effort_enum` there proves the dialect source
is live, while its absence proves the baseline (pre-dialect) source — the
flash round-1/round-2 live checks assert exactly that.

Round-2 flash empiria (recorded live-check results from the CGC
GLM-5.3-Flash test deployment, enforce matrix) — iter-1 recorded
numbers:

| Check | Observed result |
| --- | --- |
| Catalog E01 (dialect "effort_enum" — round-2 source live) | PASS: mode=enforce, native_budget=true, dialect "effort_enum" |
| E02 apply_default (8128 -> "high") / E03 clamp (32704 -> "high") | 8128 -> "high" / 32704 -> "high" (both PASS) |
| E04 effort map (64 -> "low") / E04b degradation (960 -> "low", content) | 64 -> "low", usage reasoning 2 — inside the recorded direct-low envelope [0, F02(c)+4] ("low" is a soft guide: spread {2, 63} observed per request); 960 -> "low" with content (PASS) |
| E05 client ttb precedence (16 forwarded verbatim -> binds natively) | requested 16, forwarded 16, source client, `reasoning_tokens` EXACTLY 16 — the native binding beats the enum |
| E06 incident mitigation (primes@128: 64 -> "low", content) vs the O02f round-1 baseline | reasoning 63, content, natural stop vs O02f 128/empty/length (PASS) |
| E07 reserve math (2936 -> "high") | 2936 -> "high", usage reasoning 131 (PASS) |
| O02f-vs-E06 contrast (same request: round-1 shape -> round-2 content) | O02f 128/empty/length (incident) -> E06 63 reasoning tokens, non-empty content, stop, 69 completion tokens |

The O02f-vs-E06 contrast is the dialect's reason to exist: the same
primes@128 request that consumed its whole budget on thinking in round 1
(observe, no control) gets a ladder-injected `low` in round 2 and is
expected to produce content — with reasoning bounded only approximately
( relational, against the recorded round-1 baseline — never an exact-token
claim).

## Verification and live evidence

Run the standard suite (no network credentials needed):

```sh
.venv/bin/python -m unittest
```

`tests/test_budget.py` runs real HTTP proxy/upstream connections and SQLite
billing. Fixed upstream SSE fixtures verify exact bytes for all four endpoints,
including separate usage chunks, disconnect draining and in-flight policy
snapshots. It covers observation, defaults/clamping, independent switches, model
retrieval/auth, missing/inconsistent usage, config/reload failures and billing.

An optional official SDK route test runs when `openai` is installed. Install
`openai>=1,<3` in a test environment to exercise JSON metadata and Chat/Responses
stream parsing. This was verified with openai-python **2.54.0**; it is not a
compatibility claim for every SDK or harness.

Opt-in live tests (use a test backend; generations are capped at 256 tokens):

```sh
export BUDGET_VLLM_URL=https://your-vllm-host
export BUDGET_VLLM_MODEL=your-model
# Set BUDGET_VLLM_TOKEN securely in the environment.
.venv/bin/python -m unittest tests.test_e2e_budget -v
```

These regression tests target a reasoning-enabled Qwen deployment. They assert
the small-budget failure reproduces; a backend which changes its default thinking
mode may legitimately require a different fixture. Live SSE tests check protocol
and billing, while byte identity is established with deterministic fixture tests,
not comparisons between independent stochastic generations.

On 2026-10-13, direct API probes and the real local proxy were tested against
`Qwen/Qwen3.5-122B-A10B-FP8`, public deployment alias `qwen35-122b`, reported vLLM
**0.26.0**, context 65,536:

| Check | Observed result |
| --- | --- |
| Chat, 256 output tokens, no native thinking cap | `length`, empty final content, 256 completion tokens |
| Same chat prompt, native thinking budget 16 | `stop`, final answer contained `323`, 21–27 completion tokens across seeded test runs |
| Chat policy, JSON and SSE | `thinking_token_budget=16`, usable final content, 27 completion tokens each (unseeded matrix) |
| Legacy completions, Responses, Messages; JSON and SSE | Output clamp echoed as 256, backend output usage 256, two unchanged billing events each |
| Direct `thinking_token_budget=16` on Responses/Messages | Still 256 tokens of reasoning, no final text: **control ineffective** |
| Chat reasoning usage details | Missing despite reasoning text; partition remains unknown |
| Responses reasoning usage details | Reported zero despite reasoning output; marked inconsistent |

The live test matrix made ten bounded generations and billed through the local
test SQLite database; it did not change or deploy the production proxy/backend.
Credentials are never committed. These results establish the tested deployment's
behavior, not guarantees for other vLLM versions/models/parser configurations.
