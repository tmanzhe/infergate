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
class ModelRoute:
    cheap_model: str
    strong_model: str
    fallback_on_error: bool = True
    fallback_on_quality: bool = True
    fallback_on_limit: bool = True
    min_output_chars: int = 1
    refusal_prefixes: tuple[str, ...] = (
        "i cannot", "i can't", "i'm sorry", "i’m sorry", "i am sorry", "sorry, i",
    )

    def __post_init__(self) -> None:
        if any(not isinstance(model, str) or not model.strip()
               for model in (self.cheap_model, self.strong_model)):
            raise ValueError("MODEL_ROUTES requires non-empty cheap_model and strong_model")
        if self.cheap_model == self.strong_model:
            raise ValueError("MODEL_ROUTES models must be different")
        for flag in (self.fallback_on_error, self.fallback_on_quality, self.fallback_on_limit):
            if type(flag) is not bool:
                raise ValueError("MODEL_ROUTES fallback flags must be booleans")
        if type(self.min_output_chars) is not int or self.min_output_chars < 0:
            raise ValueError("MODEL_ROUTES min_output_chars must be a non-negative integer")
        if not isinstance(self.refusal_prefixes, (list, tuple)) or any(
            not isinstance(prefix, str) or not prefix.strip() for prefix in self.refusal_prefixes
        ):
            raise ValueError("MODEL_ROUTES refusal_prefixes must be a list of non-empty strings")


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://postgres@localhost:5432/infergate"
    database_schema: str = "public"
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str = ""
    cache_ttl_seconds: float = 3600
    mock_model: bool = False
    upstream_timeout_seconds: float = 60
    max_in_flight_per_key: int = 0
    upstream_max_attempts: int = 1
    upstream_retry_base_seconds: float = 0.25
    upstream_retry_max_seconds: float = 5
    spend_caps_usd: dict[str, Decimal] = field(default_factory=dict)
    model_routes: dict[str, ModelRoute] = field(default_factory=dict)
    model_spend_caps_usd: dict[str, Decimal] = field(default_factory=dict)
    max_in_flight_per_model: dict[str, int] = field(default_factory=dict)
    model_prices: dict[str, ModelPrice] = field(default_factory=default_prices)

    def __post_init__(self) -> None:
        if not math.isfinite(self.cache_ttl_seconds) or self.cache_ttl_seconds < 0:
            raise ValueError("CACHE_TTL_SECONDS must be finite and non-negative")
        if (
            not math.isfinite(self.upstream_timeout_seconds)
            or self.upstream_timeout_seconds <= 0
        ):
            raise ValueError("UPSTREAM_TIMEOUT_SECONDS must be finite and positive")
        if type(self.max_in_flight_per_key) is not int or self.max_in_flight_per_key < 0:
            raise ValueError("MAX_IN_FLIGHT_PER_KEY must be a non-negative integer")
        if type(self.upstream_max_attempts) is not int or not 1 <= self.upstream_max_attempts <= 10:
            raise ValueError("UPSTREAM_MAX_ATTEMPTS must be an integer from 1 to 10")
        for name, value in (
            ("UPSTREAM_RETRY_BASE_SECONDS", self.upstream_retry_base_seconds),
            ("UPSTREAM_RETRY_MAX_SECONDS", self.upstream_retry_max_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.upstream_retry_max_seconds < self.upstream_retry_base_seconds:
            raise ValueError("UPSTREAM_RETRY_MAX_SECONDS must be >= UPSTREAM_RETRY_BASE_SECONDS")
        for key, cap in self.spend_caps_usd.items():
            if not isinstance(key, str) or not cap.is_finite() or cap < 0:
                raise ValueError("SPEND_CAPS_USD must map API keys to finite non-negative USD amounts")
        for model, route in self.model_routes.items():
            if not isinstance(model, str) or not model.strip() or not isinstance(route, ModelRoute):
                raise ValueError("MODEL_ROUTES must map model names or aliases to ModelRoute values")
        for model, cap in self.model_spend_caps_usd.items():
            if not isinstance(model, str) or not model.strip() or not cap.is_finite() or cap < 0:
                raise ValueError("MODEL_SPEND_CAPS_USD must map models to finite non-negative USD amounts")
        for model, maximum in self.max_in_flight_per_model.items():
            if (not isinstance(model, str) or not model.strip()
                    or type(maximum) is not int or maximum < 0):
                raise ValueError("MAX_IN_FLIGHT_PER_MODEL must map models to non-negative integers")

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
            max_in_flight_per_key=int(os.getenv("MAX_IN_FLIGHT_PER_KEY", "0")),
            upstream_max_attempts=int(os.getenv("UPSTREAM_MAX_ATTEMPTS", "1")),
            upstream_retry_base_seconds=float(os.getenv("UPSTREAM_RETRY_BASE_SECONDS", "0.25")),
            upstream_retry_max_seconds=float(os.getenv("UPSTREAM_RETRY_MAX_SECONDS", "5")),
            spend_caps_usd={
                key: Decimal(str(cap))
                for key, cap in json.loads(os.getenv("SPEND_CAPS_USD", "{}")).items()
            },
            model_routes={
                model: ModelRoute(**route)
                for model, route in json.loads(os.getenv("MODEL_ROUTES", "{}")).items()
            },
            model_spend_caps_usd={
                model: Decimal(str(cap))
                for model, cap in json.loads(os.getenv("MODEL_SPEND_CAPS_USD", "{}")).items()
            },
            max_in_flight_per_model=json.loads(os.getenv("MAX_IN_FLIGHT_PER_MODEL", "{}")),
            model_prices=prices,
        )
