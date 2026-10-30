"""Deterministic, network-free upstream for development and tests."""

import json
from typing import Any

from infergate.cache import cache_key


def mock_completion(request: dict[str, Any]) -> dict[str, Any]:
    digest = cache_key(request)
    content = f"Mock response {digest[:16]}"
    # A stable approximation, deliberately independent of a paid tokenizer/model.
    prompt_tokens = max(1, len(json.dumps(request.get("messages", []), sort_keys=True).encode()) // 4)
    completion_tokens = len(content.split())
    return {
        "id": f"chatcmpl-mock-{digest[:24]}",
        "object": "chat.completion",
        "created": 0,
        "model": request.get("model", "mock-model"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
