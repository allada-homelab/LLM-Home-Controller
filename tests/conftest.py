"""Test fixtures for LLM Home Controller."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import chat_session
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.llm_home_controller.const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
)

TEST_API_URL = "http://localhost:8080/v1"
TEST_API_KEY = "test-api-key"
TEST_MODEL = "test-model"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Automatically enable custom integrations in all tests."""


@pytest.fixture(autouse=True)
async def setup_ha_components(hass: HomeAssistant) -> None:
    """Set up required HA components for integration tests."""
    from homeassistant.setup import async_setup_component

    await async_setup_component(hass, "homeassistant", {})
    await hass.async_block_till_done()


@dataclass
class MockChatLog(conversation.ChatLog):
    """Mock ChatLog that allows controlling tool results."""

    _mock_tool_results: dict = field(default_factory=dict)

    def mock_tool_results(self, results: dict) -> None:
        """Set mock tool results."""
        self._mock_tool_results = results

    @property
    def llm_api(self):
        """Return the LLM API."""
        return self._llm_api

    @llm_api.setter
    def llm_api(self, value):
        """Set LLM API and patch async_call_tool."""
        self._llm_api = value
        if not value:
            return

        async def async_call_tool(tool_input):
            if tool_input.id not in self._mock_tool_results:
                raise ValueError(f"Tool {tool_input.id} not found in mock results")
            return self._mock_tool_results[tool_input.id]

        self._llm_api.async_call_tool = async_call_tool


@pytest.fixture
async def mock_chat_log(hass: HomeAssistant) -> Generator[MockChatLog]:
    """Return a mock chat log."""
    with (
        patch(
            "homeassistant.components.conversation.chat_log.ChatLog",
            MockChatLog,
        ),
        chat_session.async_get_chat_session(hass, "mock-conversation-id") as session,
        conversation.async_get_chat_log(hass, session) as chat_log,
    ):
        yield chat_log


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Return a mock config entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Test LLM Controller",
        data={
            CONF_API_URL: TEST_API_URL,
            CONF_API_KEY: TEST_API_KEY,
        },
        subentries_data=[
            {
                "data": {
                    CONF_MODEL: TEST_MODEL,
                    CONF_PROMPT: DEFAULT_PROMPT,
                    CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
                    CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
                    CONF_TOP_P: DEFAULT_TOP_P,
                },
                "subentry_type": "conversation",
                "title": "Test Agent",
                "unique_id": None,
            }
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_subentry(mock_config_entry: MockConfigEntry) -> ConfigSubentry:
    """Return the first conversation subentry from the mock config entry."""
    for subentry in mock_config_entry.subentries.values():
        if subentry.subentry_type == "conversation":
            return subentry
    raise ValueError("No conversation subentry found")


@pytest.fixture
def mock_aiohttp_session() -> AsyncGenerator[AsyncMock]:
    """Return a mock aiohttp ClientSession."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    session.closed = False
    return session


@pytest.fixture
async def mock_setup_entry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_aiohttp_session: AsyncMock,
) -> MockConfigEntry:
    """Set up a mock config entry with a mock session as runtime_data."""
    mock_config_entry.runtime_data = mock_aiohttp_session
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry
