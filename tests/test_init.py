"""Tests for LLM Home Controller __init__."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.llm_home_controller import (
    async_get_models,
    async_migrate_entry,
    async_setup_entry,
    async_unload_entry,
)
from custom_components.llm_home_controller.const import (
    API_TYPE_OPENAI,
    CONF_API_KEY,
    CONF_API_TYPE,
    CONF_API_URL,
    DOMAIN,
)


def _make_mock_response(data: dict | None = None, status: int = 200) -> AsyncMock:
    """Create a mock aiohttp response as a context manager."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value=data or {"data": []})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)
    return mock_response


async def test_async_get_models_success() -> None:
    """Test fetching models successfully."""
    mock_response = _make_mock_response({"data": [{"id": "model-b"}, {"id": "model-a"}]})

    session = AsyncMock()
    session.get = MagicMock(return_value=mock_response)

    models = await async_get_models(session, "http://localhost:8080/v1", "test-key")
    assert models == ["model-a", "model-b"]

    session.get.assert_called_once()
    call_args = session.get.call_args
    assert call_args[0][0] == "http://localhost:8080/v1/models"
    assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"


async def test_async_get_models_no_api_key() -> None:
    """Test fetching models without an API key."""
    mock_response = _make_mock_response({"data": [{"id": "model-a"}]})

    session = AsyncMock()
    session.get = MagicMock(return_value=mock_response)

    models = await async_get_models(session, "http://localhost:8080/v1")
    assert models == ["model-a"]

    call_args = session.get.call_args
    assert "Authorization" not in call_args[1]["headers"]


async def test_async_get_models_strips_trailing_slash() -> None:
    """Test that trailing slash is stripped from API URL."""
    mock_response = _make_mock_response({"data": []})

    session = AsyncMock()
    session.get = MagicMock(return_value=mock_response)

    await async_get_models(session, "http://localhost:8080/v1/")
    call_args = session.get.call_args
    assert call_args[0][0] == "http://localhost:8080/v1/models"


async def test_setup_entry_success(hass: HomeAssistant) -> None:
    """Test successful entry setup."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_URL: "http://localhost:8080/v1", CONF_API_KEY: "test-key"},
    )
    entry.add_to_hass(hass)

    mock_session = AsyncMock()

    with (
        patch(
            "custom_components.llm_home_controller.async_create_clientsession",
            return_value=mock_session,
        ),
        patch(
            "custom_components.llm_home_controller.async_get_models",
            return_value=["model-a"],
        ),
        patch.object(hass.config_entries, "async_forward_entry_setups") as mock_forward,
    ):
        result = await async_setup_entry(hass, entry)

    assert result is True
    assert entry.runtime_data is mock_session
    mock_forward.assert_called_once()


async def test_setup_entry_auth_failure(hass: HomeAssistant) -> None:
    """Test entry setup with authentication failure raises ConfigEntryAuthFailed."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_URL: "http://localhost:8080/v1", CONF_API_KEY: "bad-key"},
    )
    entry.add_to_hass(hass)

    mock_session = AsyncMock()

    with (
        patch(
            "custom_components.llm_home_controller.async_create_clientsession",
            return_value=mock_session,
        ),
        patch(
            "custom_components.llm_home_controller.async_get_models",
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=401,
                message="Unauthorized",
            ),
        ),
        pytest.raises(ConfigEntryAuthFailed),
    ):
        await async_setup_entry(hass, entry)


async def test_setup_entry_connection_error(hass: HomeAssistant) -> None:
    """Test entry setup with connection error raises ConfigEntryNotReady."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_URL: "http://localhost:8080/v1"},
    )
    entry.add_to_hass(hass)

    mock_session = AsyncMock()

    with (
        patch(
            "custom_components.llm_home_controller.async_create_clientsession",
            return_value=mock_session,
        ),
        patch(
            "custom_components.llm_home_controller.async_get_models",
            side_effect=aiohttp.ClientError("Connection refused"),
        ),
        pytest.raises(ConfigEntryNotReady),
    ):
        await async_setup_entry(hass, entry)


async def test_unload_entry_success(hass: HomeAssistant) -> None:
    """Test successful entry unload."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_API_URL: "http://localhost:8080/v1"})
    entry.add_to_hass(hass)
    entry.runtime_data = AsyncMock()

    with patch.object(hass.config_entries, "async_unload_platforms", return_value=True):
        result = await async_unload_entry(hass, entry)

    assert result is True


async def test_unload_entry_failure(hass: HomeAssistant) -> None:
    """Test unload failure returns False."""
    entry = MockConfigEntry(domain=DOMAIN, data={CONF_API_URL: "http://localhost:8080/v1"})
    entry.add_to_hass(hass)
    entry.runtime_data = AsyncMock()

    with patch.object(hass.config_entries, "async_unload_platforms", return_value=False):
        result = await async_unload_entry(hass, entry)

    assert result is False


async def test_migrate_v1_to_v2(hass: HomeAssistant) -> None:
    """Test migration from v1 to v2 adds CONF_API_TYPE."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_URL: "http://localhost:8080/v1", CONF_API_KEY: "key"},
        version=1,
    )
    entry.add_to_hass(hass)

    result = await async_migrate_entry(hass, entry)

    assert result is True
    assert entry.data[CONF_API_TYPE] == API_TYPE_OPENAI
    assert entry.version == 2


async def test_migrate_future_version(hass: HomeAssistant) -> None:
    """Test migration fails for unknown future version."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_API_URL: "http://localhost:8080/v1"},
        version=3,
    )
    entry.add_to_hass(hass)

    result = await async_migrate_entry(hass, entry)

    assert result is False
