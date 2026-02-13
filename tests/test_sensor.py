"""Tests for LLM Home Controller token usage sensor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.llm_home_controller.const import CONF_MODEL, DOMAIN
from custom_components.llm_home_controller.sensor import TokenUsageSensor


def _make_sensor() -> TokenUsageSensor:
    """Create a test token usage sensor."""
    entry = MagicMock()
    subentry = MagicMock()
    subentry.subentry_id = "test-sub-id"
    subentry.title = "Test Agent"
    subentry.data = {CONF_MODEL: "test-model"}

    return TokenUsageSensor(entry, subentry)


def test_sensor_initial_state() -> None:
    """Test sensor starts at 0."""
    sensor = _make_sensor()
    assert sensor.native_value == 0
    attrs = sensor.extra_state_attributes
    assert attrs["total_input_tokens"] == 0
    assert attrs["total_output_tokens"] == 0
    assert attrs["request_count"] == 0
    assert attrs["last_input_tokens"] == 0
    assert attrs["last_output_tokens"] == 0


def test_sensor_unique_id() -> None:
    """Test unique ID is based on subentry ID."""
    sensor = _make_sensor()
    assert sensor.unique_id == "test-sub-id_token_usage"


def test_sensor_device_info() -> None:
    """Test device info matches conversation entity."""
    sensor = _make_sensor()
    assert sensor.device_info is not None
    assert (DOMAIN, "test-sub-id") in sensor.device_info["identifiers"]


def test_sensor_name() -> None:
    """Test sensor name."""
    sensor = _make_sensor()
    assert sensor.name == "Token Usage"


def test_handle_usage_update() -> None:
    """Test that usage updates accumulate correctly."""
    sensor = _make_sensor()
    sensor.hass = MagicMock(spec=HomeAssistant)
    sensor.async_write_ha_state = MagicMock()

    sensor._handle_usage_update(100, 50)
    assert sensor.native_value == 150
    assert sensor.extra_state_attributes["total_input_tokens"] == 100
    assert sensor.extra_state_attributes["total_output_tokens"] == 50
    assert sensor.extra_state_attributes["request_count"] == 1
    assert sensor.extra_state_attributes["last_input_tokens"] == 100
    assert sensor.extra_state_attributes["last_output_tokens"] == 50

    sensor._handle_usage_update(200, 100)
    assert sensor.native_value == 450
    assert sensor.extra_state_attributes["total_input_tokens"] == 300
    assert sensor.extra_state_attributes["total_output_tokens"] == 150
    assert sensor.extra_state_attributes["request_count"] == 2
    assert sensor.extra_state_attributes["last_input_tokens"] == 200
    assert sensor.extra_state_attributes["last_output_tokens"] == 100


def test_cumulative_tracking() -> None:
    """Test that multiple updates are cumulative."""
    sensor = _make_sensor()
    sensor.hass = MagicMock(spec=HomeAssistant)
    sensor.async_write_ha_state = MagicMock()

    for _i in range(5):
        sensor._handle_usage_update(10, 5)

    assert sensor.native_value == 75  # 5 * (10 + 5)
    assert sensor.extra_state_attributes["request_count"] == 5


@pytest.mark.asyncio
async def test_restore_state() -> None:
    """Test that sensor restores state across HA restarts."""
    sensor = _make_sensor()
    sensor.hass = MagicMock(spec=HomeAssistant)
    sensor.entity_id = "sensor.test_agent_token_usage"
    sensor.async_write_ha_state = MagicMock()

    # Mock restore state
    last_state = MagicMock()
    last_state.attributes = {
        "total_input_tokens": 500,
        "total_output_tokens": 250,
        "request_count": 10,
    }

    with (
        patch.object(sensor, "async_get_last_state", new_callable=AsyncMock, return_value=last_state),
        patch("custom_components.llm_home_controller.sensor.async_dispatcher_connect") as mock_connect,
    ):
        mock_connect.return_value = MagicMock()
        sensor.async_on_remove = MagicMock()
        await sensor.async_added_to_hass()

    assert sensor._total_input_tokens == 500
    assert sensor._total_output_tokens == 250
    assert sensor._request_count == 10
    assert sensor.native_value == 750


@pytest.mark.asyncio
async def test_restore_state_none() -> None:
    """Test that sensor handles no previous state."""
    sensor = _make_sensor()
    sensor.hass = MagicMock(spec=HomeAssistant)
    sensor.entity_id = "sensor.test_agent_token_usage"

    with (
        patch.object(sensor, "async_get_last_state", new_callable=AsyncMock, return_value=None),
        patch("custom_components.llm_home_controller.sensor.async_dispatcher_connect") as mock_connect,
    ):
        mock_connect.return_value = MagicMock()
        sensor.async_on_remove = MagicMock()
        await sensor.async_added_to_hass()

    assert sensor.native_value == 0
