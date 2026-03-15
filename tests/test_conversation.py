"""Tests for LLM Home Controller conversation entity."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components import conversation
from homeassistant.components.conversation import ConversationEntityFeature
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant

from custom_components.llm_home_controller.const import (
    API_TYPE_ANTHROPIC,
    CONF_API_KEY,
    CONF_API_TYPE,
    CONF_API_URL,
    CONF_ENTITY_CONTEXT_TEMPLATE,
    CONF_MODEL,
    CONF_PROMPT,
    DEFAULT_PROMPT,
    DOMAIN,
)
from custom_components.llm_home_controller.conversation import (
    LLMHomeControllerConversationEntity,
)
from custom_components.llm_home_controller.providers.anthropic import AnthropicProvider
from custom_components.llm_home_controller.providers.openai import OpenAIProvider


def _make_entry_and_subentry(
    *,
    llm_api: list[str] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Create mock entry and subentry for testing."""
    subentry_data: dict[str, Any] = {
        CONF_MODEL: "test-model",
        CONF_PROMPT: DEFAULT_PROMPT,
    }
    if llm_api is not None:
        subentry_data[CONF_LLM_HASS_API] = llm_api

    subentry = MagicMock()
    subentry.subentry_id = "test-subentry-id"
    subentry.title = "Test Agent"
    subentry.data = subentry_data

    entry = MagicMock()
    entry.data = {
        CONF_API_URL: "http://localhost:8080/v1",
        CONF_API_KEY: "test-key",
    }
    entry.runtime_data = AsyncMock()

    return entry, subentry


def test_supported_languages() -> None:
    """Test that all languages are supported."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    assert entity.supported_languages == MATCH_ALL


def test_supports_streaming() -> None:
    """Test that streaming is supported."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    assert entity.supports_streaming is True


def test_supported_features_with_llm_api() -> None:
    """Test that CONTROL feature is set when LLM API is configured."""
    entry, subentry = _make_entry_and_subentry(llm_api=["assist"])
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    assert entity.supported_features == ConversationEntityFeature.CONTROL


def test_supported_features_without_llm_api() -> None:
    """Test that CONTROL feature is not set without LLM API."""
    entry, subentry = _make_entry_and_subentry(llm_api=None)
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    assert entity.supported_features == ConversationEntityFeature(0)


def test_unique_id() -> None:
    """Test unique ID is set from subentry ID."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    assert entity.unique_id == "test-subentry-id"


def test_device_info() -> None:
    """Test device info is populated correctly."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    assert entity.device_info is not None
    assert (DOMAIN, "test-subentry-id") in entity.device_info["identifiers"]
    assert entity.device_info["name"] == "Test Agent"
    assert entity.device_info["model"] == "test-model"


def test_provider_selection_openai() -> None:
    """Test that OpenAI provider is selected by default."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    assert isinstance(entity._provider, OpenAIProvider)


def test_provider_selection_anthropic() -> None:
    """Test that Anthropic provider is selected when configured."""
    entry, subentry = _make_entry_and_subentry()
    entry.data[CONF_API_TYPE] = API_TYPE_ANTHROPIC
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    assert isinstance(entity._provider, AnthropicProvider)


@pytest.mark.asyncio
async def test_async_handle_message_converse_error() -> None:
    """Test that ConverseError is returned as ConversationResult."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    mock_error = conversation.ConverseError("Test error", "conv-123", MagicMock())
    chat_log.async_provide_llm_data = AsyncMock(side_effect=mock_error)

    result = await entity._async_handle_message(user_input, chat_log)

    assert isinstance(result, conversation.ConversationResult)


@pytest.mark.asyncio
async def test_async_handle_message_calls_handle_chat_log() -> None:
    """Test that _async_handle_message calls _async_handle_chat_log."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock) as mock_handle,
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log"
        ) as mock_result,
    ):
        mock_result.return_value = MagicMock(spec=conversation.ConversationResult)
        await entity._async_handle_message(user_input, chat_log)

    mock_handle.assert_called_once_with(chat_log)
    mock_result.assert_called_once_with(user_input, chat_log)


@pytest.mark.asyncio
async def test_prompt_template_rendered() -> None:
    """Test that HA templates in the prompt are rendered."""
    entry, subentry = _make_entry_and_subentry()
    subentry.data = {**subentry.data, CONF_PROMPT: "The time is {{ now().isoformat() }}"}
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
        patch("custom_components.llm_home_controller.conversation.Template") as mock_tpl_cls,
    ):
        mock_tpl = MagicMock()
        mock_tpl.async_render.return_value = "The time is 2025-01-01T00:00:00"
        mock_tpl_cls.return_value = mock_tpl
        await entity._async_handle_message(user_input, chat_log)

    # Verify the rendered prompt was passed (3rd arg to async_provide_llm_data)
    call_args = chat_log.async_provide_llm_data.call_args
    assert call_args[0][2] == "The time is 2025-01-01T00:00:00"


@pytest.mark.asyncio
async def test_prompt_plain_text_passthrough() -> None:
    """Test that a plain text prompt passes through unchanged."""
    entry, subentry = _make_entry_and_subentry()
    subentry.data = {**subentry.data, CONF_PROMPT: "You are a helpful assistant."}
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
        patch("custom_components.llm_home_controller.conversation.Template") as mock_tpl_cls,
    ):
        mock_tpl = MagicMock()
        mock_tpl.async_render.return_value = "You are a helpful assistant."
        mock_tpl_cls.return_value = mock_tpl
        await entity._async_handle_message(user_input, chat_log)

    call_args = chat_log.async_provide_llm_data.call_args
    assert call_args[0][2] == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_prompt_template_error_fallback() -> None:
    """Test that invalid template falls back to raw string."""
    entry, subentry = _make_entry_and_subentry()
    subentry.data = {**subentry.data, CONF_PROMPT: "{{ invalid.syntax }"}
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
        patch(
            "custom_components.llm_home_controller.conversation.Template",
            side_effect=Exception("Template error"),
        ),
    ):
        await entity._async_handle_message(user_input, chat_log)

    # Fallback: raw prompt is used
    call_args = chat_log.async_provide_llm_data.call_args
    assert call_args[0][2] == "{{ invalid.syntax }"


@pytest.mark.asyncio
async def test_prompt_none_passthrough() -> None:
    """Test that None prompt passes through as None."""
    entry, subentry = _make_entry_and_subentry()
    subentry.data = {CONF_MODEL: "test-model"}  # No CONF_PROMPT
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
    ):
        await entity._async_handle_message(user_input, chat_log)

    call_args = chat_log.async_provide_llm_data.call_args
    assert call_args[0][2] is None


# --- Feature: Voice mode ---


@pytest.mark.asyncio
async def test_voice_mode_appends_suffix() -> None:
    """Test that voice mode appends conciseness instruction to prompt."""
    from custom_components.llm_home_controller.const import CONF_VOICE_MODE, VOICE_MODE_SUFFIX

    entry, subentry = _make_entry_and_subentry()
    subentry.data = {**subentry.data, CONF_VOICE_MODE: True}
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
    ):
        await entity._async_handle_message(user_input, chat_log)

    call_args = chat_log.async_provide_llm_data.call_args
    rendered_prompt = call_args[0][2]
    assert rendered_prompt.endswith(VOICE_MODE_SUFFIX)


@pytest.mark.asyncio
async def test_voice_mode_disabled_no_suffix() -> None:
    """Test that voice mode disabled does not modify the prompt."""
    entry, subentry = _make_entry_and_subentry()
    # voice_mode not set
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
    ):
        await entity._async_handle_message(user_input, chat_log)

    call_args = chat_log.async_provide_llm_data.call_args
    rendered_prompt = call_args[0][2]
    assert rendered_prompt == DEFAULT_PROMPT


# --- Feature: Conversation memory ---


@pytest.mark.asyncio
async def test_memory_saves_and_restores() -> None:
    """Test that conversation memory saves and restores history."""
    from custom_components.llm_home_controller.const import CONF_MEMORY_ENABLED

    entry, subentry = _make_entry_and_subentry()
    subentry.data = {**subentry.data, CONF_MEMORY_ENABLED: True}
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    # First message — no history to inject
    user_content_1 = conversation.UserContent(content="Hello")
    system_content = MagicMock()
    system_content.role = "system"

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "conv-123"
    chat_log.content = [system_content, user_content_1]
    chat_log.async_provide_llm_data = AsyncMock()

    # After the call, chat_log.content will have assistant response added
    assistant_content = conversation.AssistantContent(agent_id="test", content="Hi there!")
    chat_log.content = [system_content, user_content_1, assistant_content]

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
    ):
        await entity._async_handle_message(user_input, chat_log)

    # Verify history was saved (single rolling buffer, not keyed by conv_id)
    assert len(entity._conversation_history) == 2  # user + assistant
    assert entity._conversation_history[0]["type"] == "user"
    assert entity._conversation_history[1]["type"] == "assistant"


@pytest.mark.asyncio
async def test_memory_injects_history_into_fresh_log() -> None:
    """Test that saved history is injected into a fresh chat log."""
    from custom_components.llm_home_controller.const import CONF_MEMORY_ENABLED

    entry, subentry = _make_entry_and_subentry()
    subentry.data = {**subentry.data, CONF_MEMORY_ENABLED: True}
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    # Pre-populate history (single rolling buffer)
    entity._conversation_history = [
        {"type": "user", "content": "Previous question"},
        {"type": "assistant", "agent_id": "test", "content": "Previous answer"},
    ]

    system_content = MagicMock()
    system_content.role = "system"
    current_user = conversation.UserContent(content="New question")

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "conv-456"
    chat_log.content = [system_content, current_user]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
    ):
        await entity._async_handle_message(user_input, chat_log)

    # Should have: system + previous_user + previous_assistant + current_user
    assert len(chat_log.content) == 4
    assert chat_log.content[1].content == "Previous question"
    assert chat_log.content[2].content == "Previous answer"
    assert chat_log.content[3].content == "New question"


# --- Feature: Serialization ---


def test_serialize_deserialize_roundtrip() -> None:
    """Test that content serialization and deserialization is lossless."""
    from custom_components.llm_home_controller.conversation import _deserialize_content, _serialize_content

    original = [
        conversation.UserContent(content="Hello"),
        conversation.AssistantContent(agent_id="test", content="Hi there!"),
        conversation.UserContent(content="Turn on lights"),
    ]
    serialized = _serialize_content(original)
    deserialized = _deserialize_content(serialized)

    assert len(deserialized) == 3
    assert deserialized[0].content == "Hello"
    assert isinstance(deserialized[0], conversation.UserContent)
    assert deserialized[1].content == "Hi there!"
    assert isinstance(deserialized[1], conversation.AssistantContent)
    assert deserialized[2].content == "Turn on lights"


# --- Feature: Update listener (hot-reload) ---


@pytest.mark.asyncio
async def test_update_listener_registered() -> None:
    """Test that an update listener is registered on async_added_to_hass."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.hass = MagicMock(spec=HomeAssistant)
    entity.entity_id = "conversation.test"

    with (
        patch("custom_components.llm_home_controller.conversation.conversation.async_set_agent"),
        patch.object(entity, "async_on_remove") as mock_on_remove,
    ):
        await entity.async_added_to_hass()

    # async_on_remove should be called with the update listener unsub
    mock_on_remove.assert_called()
    entry.add_update_listener.assert_called_once()


# --- Feature: Entity context template ---


def test_apply_custom_entity_context_replaces_static_context() -> None:
    """Test that custom entity context replaces the default Static Context section."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.hass = MagicMock(spec=HomeAssistant)

    preamble = "When controlling Home Assistant always call the intent tools."
    entity_yaml = "- names: Living Room Light\n  domain: light\n"
    api_prompt = (
        f"{preamble}\n"
        "Static Context: An overview of the areas and the devices in this smart home:\n"
        f"{entity_yaml}"
    )
    full_system = f"You are helpful.\n{api_prompt}\nThe current time is 12:00."

    # Build a mock chat_log with SystemContent
    system_content = conversation.SystemContent(content=full_system)
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.content = [system_content]
    mock_llm_api = MagicMock()
    mock_llm_api.api_prompt = api_prompt
    chat_log.llm_api = mock_llm_api

    with patch("custom_components.llm_home_controller.conversation.Template") as mock_tpl_cls:
        mock_tpl = MagicMock()
        mock_tpl.async_render.return_value = "Entities: light.living_room=on"
        mock_tpl_cls.return_value = mock_tpl

        entity._apply_custom_entity_context(chat_log, "{{ states.light }}")

    new_system = chat_log.content[0].content
    # Preamble should be preserved
    assert preamble in new_system
    # Custom context should be present
    assert "Entities: light.living_room=on" in new_system
    # Default YAML should be gone
    assert "Static Context:" not in new_system
    assert entity_yaml not in new_system
    # Date/time suffix should be preserved
    assert "The current time is 12:00." in new_system


def test_apply_custom_entity_context_no_marker_appends() -> None:
    """Test that custom context is appended when Static Context marker not found."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.hass = MagicMock(spec=HomeAssistant)

    original_system = "You are helpful.\nSome other context."
    system_content = conversation.SystemContent(content=original_system)
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.content = [system_content]
    chat_log.llm_api = None

    with patch("custom_components.llm_home_controller.conversation.Template") as mock_tpl_cls:
        mock_tpl = MagicMock()
        mock_tpl.async_render.return_value = "Custom entities here"
        mock_tpl_cls.return_value = mock_tpl

        entity._apply_custom_entity_context(chat_log, "template")

    new_system = chat_log.content[0].content
    assert new_system == original_system + "\nCustom entities here"


def test_apply_custom_entity_context_template_error_keeps_default() -> None:
    """Test that template rendering error preserves the original system prompt."""
    entry, subentry = _make_entry_and_subentry()
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.hass = MagicMock(spec=HomeAssistant)

    marker = "Static Context: An overview of the areas and the devices in this smart home:"
    original_system = f"You are helpful.\n{marker}\n- light\n"
    system_content = conversation.SystemContent(content=original_system)
    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.content = [system_content]

    with patch(
        "custom_components.llm_home_controller.conversation.Template",
        side_effect=Exception("Bad template"),
    ):
        entity._apply_custom_entity_context(chat_log, "{{ bad }}")

    # System prompt should be unchanged
    assert chat_log.content[0].content == original_system


@pytest.mark.asyncio
async def test_entity_context_template_applied_in_handle_message() -> None:
    """Test that entity context template is applied during message handling."""
    entry, subentry = _make_entry_and_subentry()
    subentry.data = {
        **subentry.data,
        CONF_ENTITY_CONTEXT_TEMPLATE: "{{ states.light }}",
    }
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch.object(entity, "_apply_custom_entity_context") as mock_apply,
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
    ):
        await entity._async_handle_message(user_input, chat_log)

    mock_apply.assert_called_once_with(chat_log, "{{ states.light }}")


@pytest.mark.asyncio
async def test_entity_context_template_not_applied_when_empty() -> None:
    """Test that entity context replacement is skipped when template is not set."""
    entry, subentry = _make_entry_and_subentry()
    # No CONF_ENTITY_CONTEXT_TEMPLATE
    entity = LLMHomeControllerConversationEntity(entry, subentry)
    entity.entity_id = "conversation.test"
    entity.hass = MagicMock(spec=HomeAssistant)

    user_input = MagicMock(spec=conversation.ConversationInput)
    user_input.extra_system_prompt = None
    user_input.as_llm_context.return_value = MagicMock()

    chat_log = MagicMock(spec=conversation.ChatLog)
    chat_log.conversation_id = "test-conv-id"
    chat_log.content = [MagicMock(), MagicMock()]
    chat_log.async_provide_llm_data = AsyncMock()

    with (
        patch.object(entity, "_async_handle_chat_log", new_callable=AsyncMock),
        patch.object(entity, "_apply_custom_entity_context") as mock_apply,
        patch(
            "custom_components.llm_home_controller.conversation.conversation.async_get_result_from_chat_log",
            return_value=MagicMock(spec=conversation.ConversationResult),
        ),
    ):
        await entity._async_handle_message(user_input, chat_log)

    mock_apply.assert_not_called()
