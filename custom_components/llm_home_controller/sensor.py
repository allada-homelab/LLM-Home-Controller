"""Token usage sensor for LLM Home Controller."""

from __future__ import annotations

import logging

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from . import LLMHomeControllerConfigEntry
from .const import CONF_MODEL, DOMAIN

_LOGGER = logging.getLogger(__name__)

SIGNAL_USAGE = f"{DOMAIN}_usage"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LLMHomeControllerConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up token usage sensors."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue
        async_add_entities(
            [TokenUsageSensor(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class TokenUsageSensor(RestoreEntity, SensorEntity):
    """Sensor tracking token usage for an LLM conversation agent."""

    _attr_has_entity_name = True
    _attr_name = "Token Usage"
    _attr_native_unit_of_measurement = "tokens"
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_icon = "mdi:counter"

    def __init__(
        self,
        entry: LLMHomeControllerConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the sensor."""
        self._subentry_id = subentry.subentry_id
        self._attr_unique_id = f"{subentry.subentry_id}_token_usage"
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="LLM Home Controller",
            model=subentry.data.get(CONF_MODEL, "unknown"),
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._request_count: int = 0
        self._last_input_tokens: int = 0
        self._last_output_tokens: int = 0

    @property
    def native_value(self) -> int:
        """Return the total number of tokens used."""
        return self._total_input_tokens + self._total_output_tokens

    @property
    def extra_state_attributes(self) -> dict[str, int]:
        """Return detailed usage breakdown."""
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "request_count": self._request_count,
            "last_input_tokens": self._last_input_tokens,
            "last_output_tokens": self._last_output_tokens,
        }

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to usage updates."""
        await super().async_added_to_hass()

        # Restore previous state
        if (last_state := await self.async_get_last_state()) is not None:
            attrs = last_state.attributes
            self._total_input_tokens = attrs.get("total_input_tokens", 0)
            self._total_output_tokens = attrs.get("total_output_tokens", 0)
            self._request_count = attrs.get("request_count", 0)

        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_USAGE}_{self._subentry_id}",
                self._handle_usage_update,
            )
        )

    @callback
    def _handle_usage_update(self, input_tokens: int, output_tokens: int) -> None:
        """Handle a usage update from the conversation entity."""
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._request_count += 1
        self._last_input_tokens = input_tokens
        self._last_output_tokens = output_tokens
        self.async_write_ha_state()
