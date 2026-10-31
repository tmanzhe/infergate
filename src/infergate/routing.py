"""Small, deterministic output checks for opt-in two-model routing."""

import json

from fastapi.responses import Response

from infergate.config import ModelRoute


def fails_quality(response: Response, route: ModelRoute) -> bool:
    try:
        payload = json.loads(response.body)
    except (ValueError, UnicodeDecodeError):
        return True
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return True
    for choice in choices:
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict) or message.get("refusal"):
            return True
        content = message.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if text.casefold().startswith(tuple(prefix.strip().casefold() for prefix in route.refusal_prefixes)):
            return True
        # Tool calls are valid assistant output even when content is null.
        if not text and message.get("tool_calls"):
            continue
        if not text or len(text) < route.min_output_chars:
            return True
    return False


def should_fallback(response: Response, route: ModelRoute, reason: str | None) -> bool:
    if reason == "caller_cap":
        return False
    if reason == "model_limit":
        return route.fallback_on_limit
    if not 200 <= response.status_code < 300:
        return route.fallback_on_error
    return route.fallback_on_quality and fails_quality(response, route)
