"""AI Task entity for LLM Home Controller."""

from __future__ import annotations

import logging
from json import JSONDecodeError

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from . import LLMHomeControllerConfigEntry
from .const import (
    API_TYPE_OPENAI,
    CONF_API_KEY,
    CONF_API_TYPE,
    CONF_API_URL,
    CONF_CUSTOM_HEADERS,
)
from .entity import LLMHomeControllerBaseLLMEntity
from .providers import get_provider

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: LLMHomeControllerConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != "ai_task_data":
            continue
        async_add_entities(
            [LLMHomeControllerAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class LLMHomeControllerAITaskEntity(
    ai_task.AITaskEntity,
    LLMHomeControllerBaseLLMEntity,
):
    """LLM Home Controller AI Task entity."""

    def __init__(
        self,
        entry: LLMHomeControllerConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the AI task entity."""
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
        self._attr_supported_features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(chat_log)

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError("Last content in chat log is not an AssistantContent")

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            _LOGGER.error(
                "Failed to parse structured response: %s. Response: %s",
                err,
                text,
            )
            raise HomeAssistantError("LLM returned invalid JSON for structured task") from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
