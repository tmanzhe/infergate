# infergate

An OpenAI-compatible chat gateway with streaming, cheap-first routing, spend and
concurrency limits, an exact-match cache, and a Postgres cost log. Mock mode lets
you try it without calling a model API.

## Quickstart

You'll need Python 3.12+, [uv](https://docs.astral.sh/uv/), and Postgres
(tested with Postgres 17). Create the database with `createdb -U postgres infergate`.
Tables are created at startup; existing tables are kept.

```bash
uv sync
export DATABASE_URL=postgresql://postgres@localhost:5432/infergate
export MOCK_MODEL=true
uv run --no-env-file infergate
```

Send this twice to get a cache miss, then a hit:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}]}'
```

Mock responses are deterministic and log zero cost. Clients can use
`http://127.0.0.1:8000/v1` as their OpenAI base URL.

For an upstream API, set `MOCK_MODEL=false`, `UPSTREAM_BASE_URL` (including the
version path), and `UPSTREAM_API_KEY` before starting.

## Streaming

Set `"stream":true` to pass SSE chunks through as they arrive. Streams skip the
cache. Usage is logged once when the stream ends, including disconnects and
errors, using upstream usage events when present. Retries and fallback stop once
streaming begins. Streaming routes can fall back on errors, limits, or an empty
stream before delivery starts.

## Routing and limits

Send `"model":"auto"` with this config to try the cheap model first:

```bash
export MODEL_ROUTES='{"auto":{"cheap_model":"gpt-4o-mini","strong_model":"gpt-4o","min_output_chars":8}}'
export MODEL_SPEND_CAPS_USD='{"gpt-4o-mini":0.50}'
export MAX_IN_FLIGHT_PER_MODEL='{"gpt-4o-mini":2}'
```

Keep `MOCK_MODEL=true` for a free demo. The route tries the strong model once if
the cheap model fails after retries, hits a model limit, or returns an empty,
short, or refusal answer. `fallback_on_error`, `fallback_on_quality`, and
`fallback_on_limit` each default to `true`. For non-streaming answers,
`min_output_chars` defaults to `1`, and `refusal_prefixes` is configurable.

Model spend caps count lifetime logged USD across callers and can overshoot by
one attempt. Caller caps (`SPEND_CAPS_USD`) still apply through retries and
fallback, even on cache hits. Concurrency limits are per process; `0` means
unlimited. Caller limits use the request's Bearer token, separate from the
upstream key.

## Configuration

| Environment variable | Default / meaning |
| --- | --- |
| `DATABASE_URL` | `postgresql://postgres@localhost:5432/infergate` |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` (include the version path) |
| `UPSTREAM_API_KEY` | Empty; sent as a Bearer token when set |
| `CACHE_TTL_SECONDS` | `3600`; fractional seconds allowed, `0` disables caching |
| `MOCK_MODEL` | `false`; enable with `true` or `1` |
| `UPSTREAM_TIMEOUT_SECONDS` | `60` |
| `HOST`, `PORT` | `127.0.0.1`, `8000` |
| `MODEL_PRICES` | JSON map of models to `prompt` / `completion` USD per million tokens |
| `MODEL_ROUTES` | `{}`; aliases with cheap and strong models |
| `SPEND_CAPS_USD` | `{}`; caller Bearer tokens mapped to USD caps |
| `MODEL_SPEND_CAPS_USD` | `{}`; model names mapped to USD caps |
| `MAX_IN_FLIGHT_PER_KEY` | `0`; concurrent requests per caller |
| `MAX_IN_FLIGHT_PER_MODEL` | `{}`; model names mapped to concurrency limits |
| `UPSTREAM_MAX_ATTEMPTS` | `1`; total attempts per model, from `1` to `10` |
| `UPSTREAM_RETRY_BASE_SECONDS` | `0.25`; base delay for exponential backoff with jitter |
| `UPSTREAM_RETRY_MAX_SECONDS` | `5`; maximum backoff delay |

Retries cover upstream 429s, 5xx responses, and timeouts. Each attempt gets a log
entry.

To replace the default price table:

```bash
export MODEL_PRICES='{"gpt-4o-mini":{"prompt":0.15,"completion":0.60},"my-model":{"prompt":1,"completion":3}}'
```

The built-in example rates are `gpt-4o-mini` (0.15 / 0.60), `gpt-4o` (2.50 / 10.00),
and `mock-model` (1.00 / 2.00), in USD per million prompt / completion tokens.
Set them to match your upstream contract. Unknown models and responses without
usage log zero cost. Cache hits log zero cost and keep the original token counts.

## Cache and cost log

`request_logs` stores timestamp, model, prompt and completion tokens, USD cost,
latency in milliseconds, cache hit, HTTP status, and a hash of the caller key.
Non-streaming attempt latency stops before cache persistence and the log write;
stream latency runs through the end of delivery.

`response_cache` stores successful (2xx) responses with an expiry. Routed answers
must also pass the enabled quality check. Expired entries are ignored and replaced
on the next cacheable response.

Cache keys use sorted, compact request JSON, ignoring `user`, `stream`,
`stream_options`, and `metadata`. Message order and generation parameters,
including upstream extensions, matter. Omitted defaults stay omitted. Route
settings are part of the key too.

Upstream bodies and status codes pass through, including errors. Connection
failures return 502; timeouts return 504.

## Mock load test

```bash
uv run --no-env-file python scripts/loadtest.py
```

No keys, server, or Postgres needed. The script runs the gateway in process with
a fake upstream and an in-memory cache and log store, then prints JSON. For a
quick smoke run, add `--requests 20 --concurrency 3 --latency-scale 0`.

The default run sends 100 unique prompts, then 100 replays, at concurrency 10.
Every fifth unique prompt gets a short cheap answer and falls back. Fake delays
are 180-260 ms for cheap and 650-950 ms for expensive. Usage ranges from 800-1,600
prompt tokens and 160-280 output tokens; rejected answers bill one output token.
Example prices per million prompt / output tokens are $0.15/$0.60 for cheap and
$2.50/$10.00 for expensive.

One measured run took 4.54 seconds:

| Metric | Result |
| --- | --- |
| Cache hits | 100/200 (50%) |
| Final route, including cache hits | 80% cheap, 20% fallback |
| Client latency p50 / p99 | 0.73 / 1,133.68 ms |
| Simulated cost / always expensive | $0.112572 / $1.040000 |
| USD saved | $0.927428 (89.18%) |

Cost includes all 100 cheap attempts and 20 fallback attempts. The baseline uses
the same successful token usage on the expensive model for all 200 requests,
without caching. Latencies cover both phases and vary by machine. They exclude
network and Postgres overhead. All calls and costs are simulated.

## Tests

```bash
DATABASE_URL=postgresql://postgres@localhost:5432/infergate uv run --no-env-file pytest
```

Tests use deterministic mocks or `httpx.MockTransport`. Each DB test creates a
unique schema and drops it afterward, so the test DB user needs schema creation
privileges. DB tests skip if Postgres is unreachable; unit tests still run.
`create_app(settings, upstream_transport=..., clock=...)` accepts a test transport
and a UTC datetime clock.
