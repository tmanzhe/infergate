"""Environment configuration. Prices are USD per million tokens."""

import json
import math
import os
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass(frozen=True)
class ModelPrice:
    prompt: Decimal
    completion: Decimal

    def __post_init__(self) -> None:
        for value in (self.prompt, self.completion):
            if not value.is_finite() or value < 0:
                raise ValueError("Model prices must be finite and non-negative")


def default_prices() -> dict[str, ModelPrice]:
    # Illustrative defaults; override MODEL_PRICES to match your upstream contract.
    return {
        "gpt-4o-mini": ModelPrice(Decimal("0.15"), Decimal("0.60")),
        "gpt-4o": ModelPrice(Decimal("2.50"), Decimal("10.00")),
        "mock-model": ModelPrice(Decimal("1.00"), Decimal("2.00")),
    }


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://postgres@localhost:5432/infergate"
    database_schema: str = "public"
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str = ""
    cache_ttl_seconds: float = 3600
    mock_model: bool = False
    upstream_timeout_seconds: float = 60
    model_prices: dict[str, ModelPrice] = field(default_factory=default_prices)

    def __post_init__(self) -> None:
        if not math.isfinite(self.cache_ttl_seconds) or self.cache_ttl_seconds < 0:
            raise ValueError("CACHE_TTL_SECONDS must be finite and non-negative")
        if (
            not math.isfinite(self.upstream_timeout_seconds)
            or self.upstream_timeout_seconds <= 0
        ):
            raise ValueError("UPSTREAM_TIMEOUT_SECONDS must be finite and positive")

    @classmethod
    def from_env(cls) -> "Settings":
        prices = default_prices()
        if raw := os.getenv("MODEL_PRICES"):
            prices = {
                model: ModelPrice(Decimal(str(rates["prompt"])), Decimal(str(rates["completion"])))
                for model, rates in json.loads(raw).items()
            }
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            upstream_base_url=os.getenv("UPSTREAM_BASE_URL", cls.upstream_base_url),
            upstream_api_key=os.getenv("UPSTREAM_API_KEY", ""),
            cache_ttl_seconds=float(os.getenv("CACHE_TTL_SECONDS", "3600")),
            mock_model=os.getenv("MOCK_MODEL", "false").lower() in {"1", "true", "yes", "on"},
            upstream_timeout_seconds=float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "60")),
            model_prices=prices,
        )
