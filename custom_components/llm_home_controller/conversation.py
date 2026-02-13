"""Conversation entity for LLM Home Controller."""

from __future__ import annotations

import logging
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.template import Template

from . import LLMHomeControllerConfigEntry
from .const import (
    API_TYPE_OPENAI,
    CONF_API_KEY,
    CONF_API_TYPE,
    CONF_API_URL,
    CONF_CUSTOM_HEADERS,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_MAX_MESSAGES,
    CONF_PROMPT,
    CONF_VOICE_MODE,
    DEFAULT_MEMORY_MAX_MESSAGES,
    DOMAIN,
    VOICE_MODE_SUFFIX,
)
from .entity import LLMHomeControllerBaseLLMEntity
from .providers import get_provider

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LLMHomeControllerConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "conversation":
            continue
        async_add_entities(
            [LLMHomeControllerConversationEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


def _serialize_content(content_list: list[conversation.Content]) -> list[dict[str, Any]]:
    """Serialize ChatLog content items for memory persistence."""
    serialized: list[dict[str, Any]] = []
    for item in content_list:
        if isinstance(item, conversation.ToolResultContent):
            serialized.append(
                {
                    "type": "tool_result",
                    "agent_id": item.agent_id,
                    "tool_call_id": item.tool_call_id,
                    "tool_name": item.tool_name,
                    "tool_result": item.tool_result,
                }
            )
        elif isinstance(item, conversation.AssistantContent):
            entry: dict[str, Any] = {
                "type": "assistant",
                "agent_id": item.agent_id,
                "content": item.content,
            }
            if item.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.id, "tool_name": tc.tool_name, "tool_args": tc.tool_args} for tc in item.tool_calls
                ]
            serialized.append(entry)
        elif isinstance(item, conversation.UserContent):
            serialized.append(
                {
                    "type": "user",
                    "content": item.content,
                }
            )
    return serialized


def _deserialize_content(serialized: list[dict[str, Any]]) -> list[conversation.Content]:
    """Deserialize saved memory back into Content objects."""
    from homeassistant.helpers import llm

    result: list[conversation.Content] = []
    for item in serialized:
        item_type = item.get("type")
        if item_type == "user":
            result.append(conversation.UserContent(content=item["content"]))
        elif item_type == "assistant":
            tool_calls = None
            if "tool_calls" in item:
                tool_calls = [
                    llm.ToolInput(
                        id=tc["id"],
                        tool_name=tc["tool_name"],
                        tool_args=tc["tool_args"],
                    )
                    for tc in item["tool_calls"]
                ]
            result.append(
                conversation.AssistantContent(
                    agent_id=item.get("agent_id", ""),
                    content=item.get("content"),
                    tool_calls=tool_calls,
                )
            )
        elif item_type == "tool_result":
            result.append(
                conversation.ToolResultContent(
                    agent_id=item.get("agent_id", ""),
                    tool_call_id=item["tool_call_id"],
                    tool_name=item["tool_name"],
                    tool_result=item["tool_result"],
                )
            )
    return result


class LLMHomeControllerConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    LLMHomeControllerBaseLLMEntity,
):
    """LLM Home Controller conversation agent entity."""

    _attr_supports_streaming = True

    def __init__(
        self,
        entry: LLMHomeControllerConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the conversation entity."""
        provider = get_provider(entry.data.get(CONF_API_TYPE, API_TYPE_OPENAI))
        super().__init__(
            entry=subentry,
            subentry=subentry,
            session=entry.runtime_data,
            api_url=entry.data[CONF_API_URL],
            api_key=entry.data.get(CONF_API_KEY),
            provider=provider,
            custom_headers_raw=entry.data.get(CONF_CUSTOM_HEADERS),
        )
        self.entry = entry
        if subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL

        # Conversation memory: rolling buffer of recent messages across all sessions
        self._conversation_history: list[dict[str, Any]] = []

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process the user input and call the API."""
        options = self.subentry.data

        # --- Memory: inject saved history into fresh chat logs ---
        memory_enabled = options.get(CONF_MEMORY_ENABLED, False)
        max_messages = int(options.get(CONF_MEMORY_MAX_MESSAGES, DEFAULT_MEMORY_MAX_MESSAGES))

        if memory_enabled and self._conversation_history:
            # Check if this is a fresh log (system prompt + current user message only)
            is_fresh = len(chat_log.content) <= 2
            if is_fresh:
                saved = self._conversation_history[-max_messages:]
                history_items = _deserialize_content(saved)
                if history_items:
                    # Insert after system prompt (index 0), before current user message
                    current_user = chat_log.content[-1]
                    chat_log.content[1:] = [*history_items, current_user]

        # --- Render prompt template ---
        raw_prompt = options.get(CONF_PROMPT)
        rendered_prompt = raw_prompt
        if raw_prompt:
            try:
                tpl = Template(raw_prompt, self.hass)
                tpl.hass = self.hass
                rendered_prompt = tpl.async_render()
            except Exception:
                _LOGGER.warning("Failed to render prompt template, using raw prompt")

        # --- Voice mode: append conciseness instruction ---
        if options.get(CONF_VOICE_MODE, False) and rendered_prompt:
            rendered_prompt += VOICE_MODE_SUFFIX

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                rendered_prompt,
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        # --- Memory: append this session's content to rolling history ---
        if memory_enabled:
            # Save all non-system content (skip index 0 which is the system prompt)
            # Exclude previously injected history — only save new messages from this session
            new_content = [c for c in chat_log.content[1:] if not isinstance(c, conversation.SystemContent)]
            self._conversation_history.extend(_serialize_content(new_content))
            # Trim to max
            self._conversation_history = self._conversation_history[-max_messages:]

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def async_added_to_hass(self) -> None:
        """Register as a conversation agent when added to HA."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)
        # Listen for config entry updates (model changes, etc.)
        self.async_on_remove(self.entry.add_update_listener(self._async_entry_updated))

    async def async_will_remove_from_hass(self) -> None:
        """Unregister as a conversation agent when removed from HA."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @staticmethod
    async def _async_entry_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle config entry update — reload to pick up changes."""
        await hass.config_entries.async_reload(entry.entry_id)
