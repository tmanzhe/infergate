"""Caller identity is separate from the configured upstream credential."""

import hashlib


def bearer_key(authorization: str) -> str:
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return ""


def key_hash(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()
