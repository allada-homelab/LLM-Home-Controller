"""Tests for LLM Home Controller AI Task entity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from custom_components.llm_home_controller.ai_task import (
    LLMHomeControllerAITaskEntity,
)
from custom_components.llm_home_controller.const import (
    CONF_API_KEY,
    CONF_API_URL,
    CONF_MODEL,
    DOMAIN,
)
from custom_components.llm_home_controller.providers.openai import OpenAIProvider


def _make_entry_and_subentry() -> tuple[MagicMock, MagicMock]:
    """Create mock entry and subentry for testing."""
    subentry = MagicMock()
    subentry.subentry_id = "test-ai-task-id"
    subentry.title = "Test AI Task"
    subentry.data = {CONF_MODEL: "test-model"}

    entry = MagicMock()
    entry.data = {
        CONF_API_URL: "http://localhost:8080/v1",
        CONF_API_KEY: "test-key",
    }
    entry.runtime_data = AsyncMock()

    return entry, subentry


def test_supported_features() -> None:
    """Test that GENERATE_DATA and SUPPORT_ATTACHMENTS are set."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerAITaskEntity(entry, subentry)
    assert entity.supported_features & ai_task.AITaskEntityFeature.GENERATE_DATA
    assert entity.supported_features & ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS


def test_unique_id() -> None:
    """Test unique ID is set from subentry ID."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerAITaskEntity(entry, subentry)
    assert entity.unique_id == "test-ai-task-id"


def test_device_info() -> None:
    """Test device info is populated correctly."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerAITaskEntity(entry, subentry)
    assert entity.device_info is not None
    assert (DOMAIN, "test-ai-task-id") in entity.device_info["identifiers"]


def test_provider_is_openai_by_default() -> None:
    """Test that OpenAI provider is selected by default."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerAITaskEntity(entry, subentry)
    assert isinstance(entity._provider, OpenAIProvider)


@pytest.mark.asyncio
async def test_generate_data_text() -> None:
    """Test _async_generate_data returns text when no structure."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerAITaskEntity(entry, subentry)
    entity.entity_id = "ai_task.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    task = MagicMock(spec=ai_task.GenDataTask)
    task.structure = None

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "conv-123"
    assistant_content = conversation.AssistantContent(agent_id="test", content="Here is the summary.")
    chat_log.content = [MagicMock(), MagicMock(), assistant_content]

    entity._async_handle_chat_log = AsyncMock()

    result = await entity._async_generate_data(task, chat_log)

    entity._async_handle_chat_log.assert_called_once_with(chat_log)
    assert isinstance(result, ai_task.GenDataTaskResult)
    assert result.data == "Here is the summary."
    assert result.conversation_id == "conv-123"


@pytest.mark.asyncio
async def test_generate_data_structured() -> None:
    """Test _async_generate_data parses JSON when structure is set."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerAITaskEntity(entry, subentry)
    entity.entity_id = "ai_task.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    task = MagicMock(spec=ai_task.GenDataTask)
    task.structure = MagicMock()  # non-None triggers JSON parsing

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "conv-456"
    assistant_content = conversation.AssistantContent(agent_id="test", content='{"items": ["milk", "eggs"]}')
    chat_log.content = [MagicMock(), MagicMock(), assistant_content]

    entity._async_handle_chat_log = AsyncMock()

    result = await entity._async_generate_data(task, chat_log)

    assert result.data == {"items": ["milk", "eggs"]}


@pytest.mark.asyncio
async def test_generate_data_invalid_json() -> None:
    """Test _async_generate_data raises on invalid JSON with structure."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerAITaskEntity(entry, subentry)
    entity.entity_id = "ai_task.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    task = MagicMock(spec=ai_task.GenDataTask)
    task.structure = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "conv-789"
    assistant_content = conversation.AssistantContent(agent_id="test", content="not valid json")
    chat_log.content = [MagicMock(), MagicMock(), assistant_content]

    entity._async_handle_chat_log = AsyncMock()

    with pytest.raises(HomeAssistantError, match="invalid JSON"):
        await entity._async_generate_data(task, chat_log)


@pytest.mark.asyncio
async def test_generate_data_no_assistant_content() -> None:
    """Test _async_generate_data raises when last content is not assistant."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerAITaskEntity(entry, subentry)
    entity.entity_id = "ai_task.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    task = MagicMock(spec=ai_task.GenDataTask)
    task.structure = None

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "conv-000"
    chat_log.content = [MagicMock(), MagicMock()]  # no AssistantContent

    entity._async_handle_chat_log = AsyncMock()

    with pytest.raises(HomeAssistantError, match="AssistantContent"):
        await entity._async_generate_data(task, chat_log)
