from decimal import Decimal

import pytest

from infergate.config import ModelPrice, ModelRoute, Settings


def test_environment_configuration(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://local/test")
    monkeypatch.setenv("UPSTREAM_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("UPSTREAM_API_KEY", "secret")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0.125")
    monkeypatch.setenv("UPSTREAM_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("MOCK_MODEL", "true")
    monkeypatch.setenv("MODEL_PRICES", '{"test":{"prompt":0.1,"completion":0.7}}')
    settings = Settings.from_env()
    assert settings.database_url == "postgresql://local/test"
    assert settings.upstream_base_url == "http://localhost:9999/v1"
    assert settings.upstream_api_key == "secret"
    assert settings.cache_ttl_seconds == 0.125
    assert settings.upstream_timeout_seconds == 2
    assert settings.mock_model
    assert settings.model_prices == {"test": ModelPrice(Decimal("0.1"), Decimal("0.7"))}


@pytest.mark.parametrize("ttl", [-1, float("nan"), float("inf")])
def test_invalid_ttl(ttl):
    with pytest.raises(ValueError, match="CACHE_TTL_SECONDS"):
        Settings(cache_ttl_seconds=ttl)


@pytest.mark.parametrize("price", ["-1", "NaN", "Infinity"])
def test_invalid_prices(price):
    with pytest.raises(ValueError, match="Model prices"):
        ModelPrice(Decimal(price), Decimal(0))


def test_routing_environment_configuration(monkeypatch):
    monkeypatch.setenv("MODEL_ROUTES", '{"auto":{"cheap_model":"cheap","strong_model":"strong",'
                       '"fallback_on_error":false,"min_output_chars":20,"refusal_prefixes":["declined"]}}')
    monkeypatch.setenv("MODEL_SPEND_CAPS_USD", '{"cheap":"0.05","strong":2}')
    monkeypatch.setenv("MAX_IN_FLIGHT_PER_MODEL", '{"cheap":1,"strong":0}')
    settings = Settings.from_env()
    assert settings.model_routes == {"auto": ModelRoute("cheap", "strong", fallback_on_error=False,
                                                        min_output_chars=20, refusal_prefixes=["declined"])}
    assert settings.model_spend_caps_usd == {"cheap": Decimal("0.05"), "strong": Decimal(2)}
    assert settings.max_in_flight_per_model == {"cheap": 1, "strong": 0}


@pytest.mark.parametrize("options", [
    {"cheap_model": ""}, {"strong_model": " "}, {"cheap_model": None},
    {"strong_model": "cheap"}, {"fallback_on_error": "false"},
    {"fallback_on_quality": 1}, {"fallback_on_limit": None},
    {"min_output_chars": -1}, {"min_output_chars": True}, {"min_output_chars": 1.5},
    {"refusal_prefixes": "refusal"}, {"refusal_prefixes": [""]}, {"refusal_prefixes": [None]},
])
def test_invalid_model_route(options):
    with pytest.raises(ValueError, match="MODEL_ROUTES"):
        ModelRoute(**{"cheap_model": "cheap", "strong_model": "strong", **options})


@pytest.mark.parametrize("cap", ["-1", "NaN", "Infinity"])
def test_invalid_model_spend_caps(cap):
    with pytest.raises(ValueError, match="MODEL_SPEND_CAPS_USD"):
        Settings(model_spend_caps_usd={"cheap": Decimal(cap)})


@pytest.mark.parametrize("maximum", [-1, True, 0.5, "2"])
def test_invalid_model_concurrency_limits(maximum):
    with pytest.raises(ValueError, match="MAX_IN_FLIGHT_PER_MODEL"):
        Settings(max_in_flight_per_model={"cheap": maximum})


@pytest.mark.parametrize("routes", [{"": ModelRoute("cheap", "strong")}, {"auto": {}}])
def test_invalid_route_map(routes):
    with pytest.raises(ValueError, match="MODEL_ROUTES"):
        Settings(model_routes=routes)
