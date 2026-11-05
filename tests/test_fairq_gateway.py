"""Control ASGI receive/send and upstream events without timing-based assertions."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from infergate.app import create_app
from infergate.config import Settings
from infergate.database import Database


PATH = "/v1/chat/completions"
FIRST = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
END = b'data: {"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\ndata: [DONE]\n\n'


class MemoryDatabase(Database):
    def __init__(self):
        self.logs = []
        self.cache = {}

    async def initialize(self):
        pass

    async def log_request(self, **row):
        self.logs.append(row)

    async def get_cached(self, key, now):
        return self.cache.get(key)

    async def put_cached(self, key, response, expires_at):
        self.cache[key] = response


class Call:
    def __init__(self, app, group, label, tenant="", *, stream=False, hold_send=None):
        self.incoming = asyncio.Queue()
        self.incoming.put_nowait({
            "type": "http.request", "more_body": False,
            "body": json.dumps({"model": "test", "messages": [], "user": label, "stream": stream}).encode(),
        })
        self.waiting = asyncio.Event()
        self.chunk = asyncio.Event()
        self.sending = asyncio.Event()
        self.messages = []
        self.read_body = False

        async def receive():
            if self.read_body:
                # The gate has enqueued this request before starting its watcher.
                self.waiting.set()
            message = await self.incoming.get()
            self.read_body = True
            return message

        async def send(message):
            self.messages.append(message)
            if message["type"] == "http.response.body":
                self.sending.set()
                if hold_send is not None:
                    await hold_send.wait()
                if message.get("body") and message.get("more_body"):
                    self.chunk.set()

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "scheme": "http", "method": "POST", "path": PATH, "raw_path": PATH.encode(),
            "query_string": b"", "headers": [(b"authorization", f"Bearer {tenant}".encode())],
            "server": ("test", 80), "client": ("test", 1234),
        }
        self.task = group.create_task(app(scope, receive, send))

    @property
    def status(self):
        return next(message["status"] for message in self.messages if message["type"] == "http.response.start")

    @property
    def body(self):
        return b"".join(message.get("body", b"") for message in self.messages)

    def disconnect(self):
        self.incoming.put_nowait({"type": "http.disconnect"})


@pytest.fixture
def gateway():
    @asynccontextmanager
    async def make(upstream, **overrides):
        database = MemoryDatabase()
        settings = Settings(**{
            "fairq_slots": 1, "cache_ttl_seconds": 0, **overrides,
        })
        app = create_app(settings, upstream_transport=httpx.MockTransport(upstream), database=database)
        async with app.router.lifespan_context(app), asyncio.timeout(5), asyncio.TaskGroup() as group:
            yield app, database, group

    return make


async def test_weighted_tenants_interleave_under_load(gateway):
    admitted = asyncio.Queue()
    releases = {label: asyncio.Event() for label in ["blocker"] + [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(3)]}

    async def upstream(request):
        label = json.loads(request.content)["user"]
        admitted.put_nowait(label)
        await releases[label].wait()
        return httpx.Response(200, json={})

    async with gateway(upstream, fairq_weights={"alpha": 2, "beta": 1}) as (app, _, group):
        gate = app.state.fairgate
        blocker = Call(app, group, "blocker")
        assert await admitted.get() == "blocker"
        pending = []
        for tenant, prefix, count in [("alpha", "a", 6), ("beta", "b", 3)]:
            for index in range(count):
                call = Call(app, group, f"{prefix}{index}", tenant)
                await call.waiting.wait()
                pending.append(call)
        assert gate.depth == 9
        releases["blocker"].set()
        expected = ["a0", "a1", "b0", "a2", "a3", "b1", "a4", "a5", "b2"]
        for remaining, label in zip(reversed(range(9)), expected):
            assert await admitted.get() == label
            assert gate.active == 1
            assert gate.depth == remaining
            releases[label].set()
        await asyncio.gather(blocker.task, *(call.task for call in pending))
        assert all(call.status == 200 for call in pending)
        assert gate.active == gate.depth == 0


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("disconnect", [True, False])
async def test_queued_disconnect_or_task_cancellation_frees_depth(gateway, stream, disconnect):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def upstream(request):
        calls.append(json.loads(request.content)["user"])
        entered.set()
        await release.wait()
        return httpx.Response(200, json={})

    async with gateway(upstream, fairq_max_depth=1) as (app, _, group):
        first = Call(app, group, "first")
        await entered.wait()
        doomed = Call(app, group, "doomed", stream=stream)
        await doomed.waiting.wait()
        assert app.state.fairgate.depth == 1
        if disconnect:
            doomed.disconnect()
            await doomed.task
            assert doomed.status == 499
        else:
            doomed.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await doomed.task
        assert app.state.fairgate.depth == 0
        survivor = Call(app, group, "survivor")
        await survivor.waiting.wait()
        assert app.state.fairgate.depth == 1
        release.set()
        await asyncio.gather(first.task, survivor.task)
        assert survivor.status == 200
        assert calls == ["first", "survivor"]
        assert app.state.fairgate.active == app.state.fairgate.depth == 0


@pytest.mark.parametrize("stream", [False, True])
async def test_full_queue_returns_gateway_429(gateway, stream):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def upstream(request):
        calls.append(json.loads(request.content)["user"])
        entered.set()
        await release.wait()
        return httpx.Response(200, json={})

    async with gateway(upstream, fairq_max_depth=1) as (app, database, group):
        first = Call(app, group, "first")
        await entered.wait()
        queued = Call(app, group, "queued")
        await queued.waiting.wait()
        rejected = Call(app, group, "rejected", stream=stream)
        await rejected.task
        assert rejected.status == 429
        assert json.loads(rejected.body)["error"]["type"] == "gateway_error"
        assert "queue" in json.loads(rejected.body)["error"]["message"]
        assert (b"retry-after", b"1") in rejected.messages[0]["headers"]
        assert calls == ["first"]
        assert app.state.fairgate.depth == 1
        assert database.logs[0]["status"] == 429
        release.set()
        await asyncio.gather(first.task, queued.task)
        assert queued.status == 200


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, release):
        self.release = release
        self.closed = False

    async def __aiter__(self):
        yield FIRST
        await self.release.wait()
        yield END

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("disconnect", [False, True])
async def test_stream_holds_slot_until_end_or_disconnect(gateway, disconnect):
    release = asyncio.Event()
    streams, calls = [], []

    def upstream(request):
        calls.append(json.loads(request.content)["user"])
        stream = ByteStream(release)
        streams.append(stream)
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    async with gateway(upstream) as (app, database, group):
        first = Call(app, group, "first", stream=True)
        await first.chunk.wait()
        assert first.body == FIRST
        assert app.state.fairgate.active == 1
        second = Call(app, group, "second", stream=True)
        await second.waiting.wait()
        assert app.state.fairgate.depth == 1
        assert calls == ["first"]
        assert not streams[0].closed
        if disconnect:
            first.disconnect()
            await first.task
            assert database.logs[0]["status"] == 499
        release.set()
        await asyncio.gather(first.task, second.task)
        assert second.body == FIRST + END
        assert first.body == (FIRST if disconnect else FIRST + END)
        assert all(stream.closed for stream in streams)
        assert app.state.fairgate.active == app.state.fairgate.depth == 0
        assert database.logs[-1]["prompt_tokens"] == 10


async def test_cache_hits_queue_and_slot_lasts_through_response_send(gateway):
    release = asyncio.Event()
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={"answer": "cached"})

    async with gateway(upstream, cache_ttl_seconds=60) as (app, database, group):
        first = Call(app, group, "first", hold_send=release)
        await first.sending.wait()
        assert app.state.fairgate.active == 1
        cached = Call(app, group, "cached")
        await cached.waiting.wait()
        assert app.state.fairgate.depth == 1
        assert len(database.logs) == 1
        release.set()
        await asyncio.gather(first.task, cached.task)
        assert len(calls) == 1
        assert cached.body == first.body
        assert database.logs[-1]["cache_hit"]
        assert app.state.fairgate.active == 0


async def test_unconfigured_keys_share_default_tenant(gateway):
    admitted = asyncio.Queue()
    releases = {label: asyncio.Event() for label in ("first", "a0", "a1", "b0")}

    async def upstream(request):
        label = json.loads(request.content)["user"]
        admitted.put_nowait(label)
        await releases[label].wait()
        return httpx.Response(200, json={})

    async with gateway(upstream, fairq_weights={"default": 1}) as (app, _, group):
        first = Call(app, group, "first")
        assert await admitted.get() == "first"
        pending = []
        # Shared default FIFO keeps a1 ahead of b0; separate tenants would rotate.
        for label, tenant in [("a0", "unknown-a"), ("a1", "unknown-a"), ("b0", "unknown-b")]:
            call = Call(app, group, label, tenant)
            await call.waiting.wait()
            pending.append(call)
        releases["first"].set()
        for label in ("a0", "a1", "b0"):
            assert await admitted.get() == label
            releases[label].set()
        await asyncio.gather(first.task, *(call.task for call in pending))


async def test_disabled_gate_preserves_overlapping_requests(gateway):
    entered = asyncio.Queue()
    release = asyncio.Event()

    async def upstream(request):
        entered.put_nowait(None)
        await release.wait()
        return httpx.Response(200, json={})

    async with gateway(upstream, fairq_enabled=False, fairq_max_depth=1) as (app, _, group):
        pending = [Call(app, group, str(index)) for index in range(3)]
        for _ in pending:
            await entered.get()
        assert app.state.fairgate.active == app.state.fairgate.depth == 0
        release.set()
        await asyncio.gather(*(call.task for call in pending))
        assert all(call.status == 200 for call in pending)


def test_fairq_environment_configuration(monkeypatch):
    monkeypatch.setenv("FAIRQ_ENABLED", "false")
    monkeypatch.setenv("FAIRQ_WEIGHTS", " alpha=3, default=2 ")
    monkeypatch.setenv("FAIRQ_MAX_DEPTH", "7")
    monkeypatch.setenv("FAIRQ_SLOTS", "2")
    settings = Settings.from_env()
    assert not settings.fairq_enabled
    assert settings.fairq_weights == {"alpha": 3, "default": 2}
    assert settings.fairq_max_depth == 7
    assert settings.fairq_slots == 2
    assert Settings().fairq_enabled


@pytest.mark.parametrize("options", [
    {"fairq_enabled": "true"}, {"fairq_max_depth": 0}, {"fairq_max_depth": True},
    {"fairq_slots": -1}, {"fairq_slots": 1.5}, {"fairq_weights": {"": 1}},
    {"fairq_weights": {"alpha": 0}}, {"fairq_weights": {"alpha": True}},
])
def test_invalid_fairq_configuration(options):
    with pytest.raises(ValueError, match="FAIRQ_"):
        Settings(**options)
