# OpenAI Decisions adapter

Contract snapshot: OpenAI Decisions public beta, checked on 2026-10-20 against
the [API reference](https://developers.openai.com/api/reference/resources/decisions/methods/create)
and Python SDK `openai==3.26.0`. Changes in the beta API require explicit review;
the native Basal endpoint remains available for all six question types.

## Local test

Keep the existing `basal-20-max` backend configuration and restart the proxy.
The `/v1/decisions` route is registered without a feature flag. No backend or
model deployment change is required. The public OpenAI contract remains beta.

Save this request as `dev/openai-decisions-request.json`:

```json
{
  "model": "basal-20-max",
  "input": "The customer received a damaged product.",
  "questions": [
    {
      "type": "predicate",
      "name": "complaint",
      "instructions": "Is this a complaint?"
    },
    {
      "type": "choice",
      "name": "department",
      "instructions": "Which department should handle this?",
      "choices": [{"value": "support"}, {"value": "sales"}]
    },
    {
      "type": "score",
      "name": "severity",
      "instructions": "How severe is the problem?",
      "levels": [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}]
    }
  ]
}
```

Run from the repository root, using your proxy key:

```powershell
$key = Read-Host "Wklej klucz API proxy"
curl.exe -i --max-time 310 "http://127.0.0.1:8081/v1/decisions" `
    -H "Authorization: Bearer $key" `
    -H "Content-Type: application/json" `
    --data-binary "@dev/openai-decisions-request.json"
```

The response contains `model`, an `answers` array and `usage`. There is no chat
`choices` envelope and no JSON string inside `message.content`. Predicate
answers contain a probability, not a thresholded boolean. Choice answers
include the selected typed value, confidence and per-option probabilities.
Score answers contain the expected zero-based level index (possibly fractional),
confidence and per-level probabilities, with original labels. The proxy does
not generate a natural-language summary or infer a response language.

## Supported subset and boundaries

Only the `basal` wire adapter is supported. Requests accept `model`, text-string
`input` and 1–400 `questions`, further bounded by configured `max_questions`
(default 16) and `max_branches` (default 64). All three supported question types
count as one native branch each. Choice values are unique strings or booleans;
`true` and `"true"` remain distinct. Choices and score levels require 2–255
options. Score labels must be unique strings. Question names are optional
strings; unnamed answers have `name: null`. Internal question IDs preserve
order even when external names repeat.

The adapter rejects message/multimodal input, file references, streaming,
generation parameters, cache controls, safety identifiers, unknown fields and
native-only question types with a JSON 400 error. It cannot reproduce OpenAI
model calibration or safety refusals. Native responses of an unexpected type,
missing/extra options, invalid usage or inconsistent distributions produce 502
and no billing. Supported outputs retain Basal probabilities and confidence;
the adapter does not renormalize them or round scores.

Usage is based on native input/output counts. Serialization adds no model
tokens. Required cache/reasoning detail counters are zero, indicating no
separately accounted tokens, rather than observed cache misses. Authentication,
rate limiting, timeouts and validated billing use the existing decision service.
The client API key is never forwarded as the Basal backend credential.

## Routes and migration

| Route | Contract |
|---|---|
| `/v1/decisions` | OpenAI Decisions: predicate, choice, score |
| `/v1/systemone` | Native Basal: all six question types |

SDK clients use `base_url="http://127.0.0.1:8081/v1"`; the SDK's relative
`/decisions` path reaches the OpenAI frontend. The former experimental path
is removed. Remove `[experimental] openai_decisions` from local configuration.

This changes the previous native `/v1/decisions` alias. Existing native
`{state, questions: {...}}` clients must move to `/v1/systemone`. The `/v1/basal` alias is also removed; its clients
must move to `/v1/systemone`.
Inform clients of the changed contract and migration date before rollout.
Sending a native payload to the new OpenAI route is rejected with 400.

The Basal Chat Completions adapter has been removed. Clients using native JSON
in `messages[0].content` should send that inner JSON directly to `/v1/systemone`,
or construct a supported OpenAI Decisions request. Chat Completions stays
available for ordinary chat models; decision models receive 404 with
`code: model_not_supported` before inference and billing.
