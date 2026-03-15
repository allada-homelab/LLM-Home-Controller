"""LLM Home Controller integration for Home Assistant."""

from __future__ import annotations

import logging

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers.aiohttp_client import async_create_clientsession

from .const import API_TYPE_OPENAI, CONF_API_KEY, CONF_API_TYPE, CONF_API_URL
from .providers import get_provider

_LOGGER = logging.getLogger(__name__)

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION, Platform.SENSOR)

type LLMHomeControllerConfigEntry = ConfigEntry[aiohttp.ClientSession]


async def async_get_models(
    session: aiohttp.ClientSession,
    api_url: str,
    api_key: str | None = None,
    api_type: str = API_TYPE_OPENAI,
) -> list[str]:
    """Fetch available models from the API endpoint."""
    provider = get_provider(api_type)
    return await provider.get_models(session, api_url, api_key)


async def async_setup_entry(hass: HomeAssistant, entry: LLMHomeControllerConfigEntry) -> bool:
    """Set up LLM Home Controller from a config entry."""
    session = async_create_clientsession(hass)

    api_url = entry.data[CONF_API_URL]
    api_key = entry.data.get(CONF_API_KEY)
    api_type = entry.data.get(CONF_API_TYPE, API_TYPE_OPENAI)

    try:
        await async_get_models(session, api_url, api_key, api_type)
    except aiohttp.ClientResponseError as err:
        if err.status in (401, 403):
            raise ConfigEntryAuthFailed(f"Authentication failed for {api_url}: {err}") from err
        raise ConfigEntryNotReady(f"Cannot connect to {api_url}: {err}") from err
    except (aiohttp.ClientError, TimeoutError) as err:
        raise ConfigEntryNotReady(f"Cannot connect to {api_url}: {err}") from err

    entry.runtime_data = session

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_migrate_entry(hass: HomeAssistant, entry: LLMHomeControllerConfigEntry) -> bool:
    """Migrate config entry to current version."""
    if entry.version > 2:
        return False

    if entry.version == 1:
        _LOGGER.debug("Migrating config entry from version 1 to 2")
        new_data = {**entry.data, CONF_API_TYPE: API_TYPE_OPENAI}
        hass.config_entries.async_update_entry(entry, data=new_data, version=2)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: LLMHomeControllerConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
