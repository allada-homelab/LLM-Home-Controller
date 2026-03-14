"""Tests for LLM Home Controller config flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import aiohttp
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.llm_home_controller.const import (
    API_TYPE_ANTHROPIC,
    API_TYPE_OPENAI,
    API_TYPE_OPENAI_RESPONSES,
    CONF_API_KEY,
    CONF_API_TYPE,
    CONF_API_URL,
    CONF_MODEL,
    DOMAIN,
)


async def test_user_flow_success(hass: HomeAssistant) -> None:
    """Test the complete user flow with successful API connection."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        return_value=["model-a", "model-b"],
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_URL: "http://localhost:8080/v1",
                CONF_API_KEY: "test-key",
            },
        )

    # Should advance to model selection step
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_model"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_MODEL: "model-b"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "http://localhost:8080/v1"
    assert result["data"][CONF_API_URL] == "http://localhost:8080/v1"
    assert result["data"][CONF_API_KEY] == "test-key"
    assert result["data"][CONF_API_TYPE] == API_TYPE_OPENAI
    # Should have created a subentry with the selected model
    assert len(result.get("subentries", [])) == 1
    subentry = result["subentries"][0]
    assert subentry["subentry_type"] == "conversation"
    assert subentry["data"][CONF_MODEL] == "model-b"


async def test_user_flow_anthropic_type(hass: HomeAssistant) -> None:
    """Test user flow with Anthropic API type."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        return_value=["claude-3-5-sonnet", "claude-3-haiku"],
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_TYPE: API_TYPE_ANTHROPIC,
                CONF_API_URL: "https://api.anthropic.com/v1",
                CONF_API_KEY: "sk-ant-test",
            },
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_model"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_MODEL: "claude-3-5-sonnet"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_API_TYPE] == API_TYPE_ANTHROPIC
    subentry = result["subentries"][0]
    assert subentry["data"][CONF_MODEL] == "claude-3-5-sonnet"


async def test_user_flow_openai_responses_type(hass: HomeAssistant) -> None:
    """Test user flow with OpenAI Responses API type."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        return_value=["gpt-4o", "gpt-4o-mini"],
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_TYPE: API_TYPE_OPENAI_RESPONSES,
                CONF_API_URL: "https://api.openai.com/v1",
                CONF_API_KEY: "sk-test",
            },
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_model"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_MODEL: "gpt-4o"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_API_TYPE] == API_TYPE_OPENAI_RESPONSES
    subentry = result["subentries"][0]
    assert subentry["data"][CONF_MODEL] == "gpt-4o"


async def test_user_flow_no_models(hass: HomeAssistant) -> None:
    """Test user flow when API returns no models — manual input."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        return_value=[],
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_URL: "http://localhost:8080/v1"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "pick_model"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_MODEL: "my-custom-model"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    subentry = result["subentries"][0]
    assert subentry["data"][CONF_MODEL] == "my-custom-model"


async def test_user_flow_cannot_connect(hass: HomeAssistant) -> None:
    """Test user flow when API is unreachable."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        side_effect=aiohttp.ClientError("Connection refused"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_URL: "http://localhost:8080/v1"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


async def test_user_flow_invalid_auth(hass: HomeAssistant) -> None:
    """Test user flow with invalid authentication."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=401,
            message="Unauthorized",
        ),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_URL: "http://localhost:8080/v1",
                CONF_API_KEY: "bad-key",
            },
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_auth"}


async def test_user_flow_unknown_error(hass: HomeAssistant) -> None:
    """Test user flow with unexpected error."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        side_effect=RuntimeError("Unexpected"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_URL: "http://localhost:8080/v1"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "unknown"}


async def test_user_flow_non_auth_http_error(hass: HomeAssistant) -> None:
    """Test user flow with non-auth HTTP error (e.g., 500)."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=500,
            message="Internal Server Error",
        ),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_URL: "http://localhost:8080/v1"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


async def test_user_flow_timeout(hass: HomeAssistant) -> None:
    """Test user flow with connection timeout."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        side_effect=TimeoutError("Connection timed out"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_URL: "http://localhost:8080/v1"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


# --- Reconfigure flow tests ---


async def test_reconfigure_flow_success(hass: HomeAssistant) -> None:
    """Test successful reconfigure flow updates entry and aborts."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_API_TYPE: API_TYPE_OPENAI,
            CONF_API_URL: "http://localhost:8080/v1",
            CONF_API_KEY: "old-key",
        },
    )
    entry.add_to_hass(hass)

    result = await entry.start_reconfigure_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        return_value=["model-a"],
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_TYPE: API_TYPE_OPENAI,
                CONF_API_URL: "http://new-host:8080/v1",
                CONF_API_KEY: "new-key",
            },
        )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data[CONF_API_URL] == "http://new-host:8080/v1"
    assert entry.data[CONF_API_KEY] == "new-key"


async def test_reconfigure_flow_auth_error(hass: HomeAssistant) -> None:
    """Test reconfigure flow with invalid auth shows error."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_API_TYPE: API_TYPE_OPENAI,
            CONF_API_URL: "http://localhost:8080/v1",
            CONF_API_KEY: "old-key",
        },
    )
    entry.add_to_hass(hass)

    result = await entry.start_reconfigure_flow(hass)

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=401,
            message="Unauthorized",
        ),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_URL: "http://localhost:8080/v1",
                CONF_API_KEY: "bad-key",
            },
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_auth"}


async def test_reconfigure_flow_connection_error(hass: HomeAssistant) -> None:
    """Test reconfigure flow with connection error shows error."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_API_TYPE: API_TYPE_OPENAI,
            CONF_API_URL: "http://localhost:8080/v1",
        },
    )
    entry.add_to_hass(hass)

    result = await entry.start_reconfigure_flow(hass)

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        side_effect=aiohttp.ClientError("Connection refused"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_URL: "http://unreachable:8080/v1"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "cannot_connect"}


async def test_reconfigure_flow_unknown_error(hass: HomeAssistant) -> None:
    """Test reconfigure flow with unexpected error shows error."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_API_TYPE: API_TYPE_OPENAI,
            CONF_API_URL: "http://localhost:8080/v1",
        },
    )
    entry.add_to_hass(hass)

    result = await entry.start_reconfigure_flow(hass)

    with patch(
        "custom_components.llm_home_controller.config_flow.async_get_models",
        side_effect=RuntimeError("Unexpected"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_URL: "http://localhost:8080/v1"},
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "unknown"}
