from decimal import Decimal

import pytest

from infergate.config import ModelPrice, Settings


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
