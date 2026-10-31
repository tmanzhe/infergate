# infergate

An OpenAI-compatible gateway that sits in front of model APIs to cut cost and latency: it logs tokens, dollars and latency for every request and serves repeated requests from an exact-match cache with TTL.

## Quickstart

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and a running Postgres
(tested with Postgres 17). Create the `infergate` database first, for example with
`createdb -U postgres infergate`. Tables are created idempotently at startup.

```bash
uv sync
export DATABASE_URL=postgresql://postgres@localhost:5432/infergate
export UPSTREAM_BASE_URL=https://api.openai.com/v1
export UPSTREAM_API_KEY=your-key
uv run infergate
```

For a free, deterministic local demo, set `MOCK_MODEL=true` before starting;
no upstream calls are made and logged costs are zero. Then send the same request
twice to see a cache miss followed by a hit:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}]}'
```

Clients can point their OpenAI base URL at `http://127.0.0.1:8000/v1`.
`stream=true` passes SSE chunks through as they arrive and bypasses the cache.
Usage is logged once at stream end (also on disconnect/error), using upstream
usage events when provided; retries/fallback stop once streaming begins.

| Environment variable | Default / meaning |
| --- | --- |
| `DATABASE_URL` | `postgresql://postgres@localhost:5432/infergate` |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` (include the version path) |
| `UPSTREAM_API_KEY` | Empty; sent as a Bearer token when set |
| `CACHE_TTL_SECONDS` | `3600`; fractional seconds allowed, `0` disables caching |
| `MOCK_MODEL` | `false`; enable with `true` or `1` |
| `UPSTREAM_TIMEOUT_SECONDS` | `60` |
| `HOST`, `PORT` | `127.0.0.1`, `8000` |
| `MODEL_PRICES` | JSON mapping of model names to `prompt` / `completion` USD per million tokens |

Example price override (replaces the default table):

```bash
export MODEL_PRICES='{"gpt-4o-mini":{"prompt":0.15,"completion":0.60},"my-model":{"prompt":1,"completion":3}}'
```

Illustrative defaults are `gpt-4o-mini` (0.15 / 0.60), `gpt-4o` (2.50 / 10.00),
and `mock-model` (1.00 / 2.00). Set prices to match your upstream contract.
Unknown models and responses without usage log zero cost; cache hits always log
zero cost while retaining the original response's token counts.

Optional routing config (send `"model":"auto"`; keep `MOCK_MODEL=true` for a free demo):

```bash
export MODEL_ROUTES='{"auto":{"cheap_model":"gpt-4o-mini","strong_model":"gpt-4o","min_output_chars":8}}'
export MODEL_SPEND_CAPS_USD='{"gpt-4o-mini":0.50}'
export MAX_IN_FLIGHT_PER_MODEL='{"gpt-4o-mini":2}'
```

Routes try cheap first, then strong once after retries. `fallback_on_error`,
`fallback_on_quality`, and `fallback_on_limit` default to `true`; quality checks
empty/short/refusal output (`min_output_chars` defaults to `1`, `refusal_prefixes`
is configurable). Model caps use lifetime logged USD across callers and can
overshoot by one attempt; model concurrency is per process (`0` means unlimited).
Existing `SPEND_CAPS_USD` caller caps and `MAX_IN_FLIGHT_PER_KEY` always apply.

`request_logs` records timestamp, model, prompt/completion tokens, USD cost,
latency in milliseconds (through cache persistence, before the log write), cache
hit, and HTTP status. `response_cache` stores successful (2xx) bodies with expiry.
Expired entries are ignored and overwritten on the next successful request.
The cache hashes sorted, compact JSON of the request, ignoring `user`, `stream`,
`stream_options`, and `metadata`. Message order and all generation parameters
(including upstream extensions) remain significant; omitted defaults are not
expanded. Upstream bodies and status codes pass through, including errors;
transport failures return 502 and timeouts return 504.

## Mock load test

```bash
uv run --no-env-file python scripts/loadtest.py
```

No keys, server or Postgres needed. This drives the gateway in process with a fake
upstream and an in-memory cache/log store. It prints JSON. Add
`--requests 20 --concurrency 3 --latency-scale 0` for a quick smoke run.

The default sends 100 unique prompts, then 100 replays, at concurrency 10.
Every fifth unique prompt gets a short cheap answer and falls back. Fake delays
are 180-260 ms for cheap and 650-950 ms for expensive. Usage varies from 800-1,600
prompt tokens and 160-280 output tokens; rejected answers bill one output token.
Illustrative prices per million prompt/output tokens are $0.15/$0.60 for cheap
and $2.50/$10.00 for expensive.

One measured run took 4.54 seconds:

| Metric | Result |
| --- | --- |
| Cache hits | 100/200 (50%) |
| Final route, including cache hits | 80% cheap, 20% fallback |
| Client latency p50 / p99 | 0.73 / 1,133.68 ms |
| Simulated cost / always expensive | $0.112572 / $1.040000 |
| USD saved | $0.927428 (89.18%) |

Costs include all 100 cheap attempts and 20 fallback attempts. The baseline uses
the same successful token usage on the expensive model for every request, without
caching. Latencies include both phases and vary by machine; they exclude network
and Postgres overhead. No real API calls or charges.

## Tests

```bash
DATABASE_URL=postgresql://postgres@localhost:5432/infergate uv run pytest
```

Tests use deterministic mocks or `httpx.MockTransport` and never call a paid API.
Each DB test uses a unique schema and drops only that schema afterward; the test
DB user needs schema creation privileges. DB tests skip cleanly if Postgres is
unreachable; unit tests still run. `create_app(settings, upstream_transport=...,
clock=...)` supports injected transports and a UTC datetime clock for testing.
