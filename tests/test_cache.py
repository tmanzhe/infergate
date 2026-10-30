import pytest

from infergate.cache import cache_key, canonical_request


def test_canonical_request_ignores_order_and_bookkeeping():
    first = {
        "model": "test",
        "messages": [{"role": "user", "content": "héllo"}],
        "temperature": 0.2,
        "response_format": {"type": "json_schema", "json_schema": {"strict": True, "name": "test"}},
        "user": "alice",
        "stream": False,
        "stream_options": None,
        "metadata": {"trace": "one"},
    }
    second = {
        "response_format": {"json_schema": {"name": "test", "strict": True}, "type": "json_schema"},
        "temperature": 0.2,
        "messages": [{"content": "héllo", "role": "user"}],
        "model": "test",
        "user": "bob",
    }
    assert canonical_request(first) == canonical_request(second)
    assert cache_key(first) == cache_key(second)
    assert len(cache_key(first)) == 64


@pytest.mark.parametrize("field,value", [
    ("model", "different"),
    ("messages", [{"role": "user", "content": "different"}]),
    ("temperature", 0.9),
    ("seed", 123),
    ("max_completion_tokens", 20),
    ("tools", [{"type": "function", "function": {"name": "test"}}]),
    ("provider_specific_option", True),
])
def test_generation_changes_affect_key(field, value):
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2}
    assert cache_key(body) != cache_key({**body, field: value})


def test_message_order_is_significant():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert cache_key({"model": "test", "messages": messages}) != cache_key(
        {"model": "test", "messages": list(reversed(messages))}
    )
