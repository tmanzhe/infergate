"""Exact request identity: preserve generation options, discard bookkeeping."""

import hashlib
import json
from typing import Any


IGNORED_FIELDS = frozenset({"user", "stream", "stream_options", "metadata"})


def canonical_request(request: dict[str, Any]) -> str:
    # Keep unknown options too: upstream extensions may affect generation.
    normalized = {key: value for key, value in request.items() if key not in IGNORED_FIELDS}
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def cache_key(request: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_request(request).encode("utf-8")).hexdigest()
