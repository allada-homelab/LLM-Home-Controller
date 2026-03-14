"""Config flow for LLM Home Controller."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from . import async_get_models
from .const import (
    API_TYPE_ANTHROPIC,
    API_TYPE_OPENAI,
    API_TYPE_OPENAI_RESPONSES,
    CONF_API_KEY,
    CONF_API_TYPE,
    CONF_API_URL,
    CONF_CUSTOM_HEADERS,
    CONF_CUSTOM_TOOLS,
    CONF_EXTENDED_THINKING,
    CONF_FALLBACK_MODEL,
    CONF_JSON_SCHEMA,
    CONF_MAX_CONTEXT_TOKENS,
    CONF_MAX_RETRIES,
    CONF_MAX_TOKENS,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_MAX_MESSAGES,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_PROMPT_PRESET,
    CONF_RESPONSE_FORMAT,
    CONF_SEED,
    CONF_STOP_SEQUENCES,
    CONF_TEMPERATURE,
    CONF_THINKING_BUDGET,
    CONF_TOP_P,
    CONF_VOICE_MODE,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MEMORY_MAX_MESSAGES,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    DEFAULT_TOP_P,
    DOMAIN,
    PROMPT_PRESET_CUSTOM,
    PROMPT_PRESETS,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_TYPE, default=API_TYPE_OPENAI): SelectSelector(
            SelectSelectorConfig(
                options=[
                    {"value": API_TYPE_OPENAI, "label": "OpenAI Compatible"},
                    {"value": API_TYPE_OPENAI_RESPONSES, "label": "OpenAI Responses API"},
                    {"value": API_TYPE_ANTHROPIC, "label": "Anthropic Messages API"},
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Required(CONF_API_URL): TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
        vol.Optional(CONF_API_KEY): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
        vol.Optional(CONF_CUSTOM_HEADERS): TextSelector(TextSelectorConfig(multiline=True)),
    }
)

RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_MODEL: DEFAULT_MODEL,
    CONF_PROMPT: DEFAULT_PROMPT,
    CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
    CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
    CONF_TOP_P: DEFAULT_TOP_P,
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
}


class LLMHomeControllerConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for LLM Home Controller."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._user_input: dict[str, Any] = {}
        self._models: list[str] = []

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle the initial step — API connection."""
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=STEP_USER_DATA_SCHEMA,
            )

        errors: dict[str, str] = {}

        api_url = user_input[CONF_API_URL]
        api_key = user_input.get(CONF_API_KEY)
        api_type = user_input.get(CONF_API_TYPE, API_TYPE_OPENAI)

        try:
            async with aiohttp.ClientSession() as session:
                models = await async_get_models(session, api_url, api_key, api_type)
        except aiohttp.ClientResponseError as err:
            if err.status in (401, 403):
                errors["base"] = "invalid_auth"
            else:
                errors["base"] = "cannot_connect"
        except (aiohttp.ClientError, TimeoutError):
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected error during validation")
            errors["base"] = "unknown"
        else:
            user_input.setdefault(CONF_API_TYPE, API_TYPE_OPENAI)
            self._user_input = user_input
            self._models = models
            return await self.async_step_pick_model()

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_pick_model(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle the second step — model selection."""
        if user_input is not None:
            selected_model = user_input.get(CONF_MODEL, DEFAULT_MODEL)
            default_subentry_data = {
                **RECOMMENDED_CONVERSATION_OPTIONS,
                CONF_MODEL: selected_model,
            }

            return self.async_create_entry(
                title=self._user_input[CONF_API_URL],
                data=self._user_input,
                subentries=[
                    {
                        "subentry_type": "conversation",
                        "data": default_subentry_data,
                        "title": selected_model,
                        "unique_id": None,
                    }
                ],
            )

        # Build model selector
        if self._models:
            model_selector = SelectSelector(
                SelectSelectorConfig(
                    options=self._models,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            )
            default_model = self._models[0]
        else:
            model_selector = TextSelector(TextSelectorConfig())
            default_model = DEFAULT_MODEL

        schema = vol.Schema(
            {
                vol.Required(CONF_MODEL, default=default_model): model_selector,
            }
        )

        return self.async_show_form(
            step_id="pick_model",
            data_schema=schema,
            description_placeholders={"model_count": str(len(self._models))},
        )

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Handle reconfiguration of the API connection."""
        if user_input is None:
            return self.async_show_form(
                step_id="reconfigure",
                data_schema=self.add_suggested_values_to_schema(
                    STEP_USER_DATA_SCHEMA,
                    self._get_reconfigure_entry().data,
                ),
            )

        errors: dict[str, str] = {}

        api_url = user_input[CONF_API_URL]
        api_key = user_input.get(CONF_API_KEY)
        api_type = user_input.get(CONF_API_TYPE, API_TYPE_OPENAI)

        try:
            async with aiohttp.ClientSession() as session:
                await async_get_models(session, api_url, api_key, api_type)
        except aiohttp.ClientResponseError as err:
            if err.status in (401, 403):
                errors["base"] = "invalid_auth"
            else:
                errors["base"] = "cannot_connect"
        except (aiohttp.ClientError, TimeoutError):
            errors["base"] = "cannot_connect"
        except Exception:
            _LOGGER.exception("Unexpected error during validation")
            errors["base"] = "unknown"
        else:
            user_input.setdefault(CONF_API_TYPE, API_TYPE_OPENAI)
            return self.async_update_reload_and_abort(
                self._get_reconfigure_entry(),
                data=user_input,
            )

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_DATA_SCHEMA,
                user_input,
            ),
            errors=errors,
        )

    @classmethod
    @callback
    def async_get_supported_subentry_types(cls, config_entry: ConfigEntry) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentry types supported for this config entry."""
        return {"conversation": ConversationSubentryFlowHandler}


class ConversationSubentryFlowHandler(ConfigSubentryFlow):
    """Handle subentry flow for conversation agents."""

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Handle creation of a new conversation subentry."""
        return await self._async_step_form(user_input, step_id="user")

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        """Handle reconfiguration of an existing conversation subentry."""
        return await self._async_step_form(user_input, step_id="reconfigure")

    async def _async_step_form(self, user_input: dict[str, Any] | None, step_id: str) -> SubentryFlowResult:
        """Common form handler for user and reconfigure steps."""
        entry = self.hass.config_entries.async_get_entry(self.handler[0])
        if entry is None:
            return self.async_abort(reason="entry_not_found")

        api_type = entry.data.get(CONF_API_TYPE, API_TYPE_OPENAI)

        # Fetch available models for the dropdown
        models: list[str] = []
        try:
            session = entry.runtime_data
            models = await async_get_models(
                session,
                entry.data[CONF_API_URL],
                entry.data.get(CONF_API_KEY),
                api_type,
            )
        except Exception:
            _LOGGER.warning("Could not fetch models, allowing manual input")

        if user_input is not None:
            # Handle prompt preset: populate prompt if a preset was selected
            preset = user_input.pop(CONF_PROMPT_PRESET, PROMPT_PRESET_CUSTOM)
            if preset != PROMPT_PRESET_CUSTOM and preset in PROMPT_PRESETS:
                current_prompt = user_input.get(CONF_PROMPT, "")
                if not current_prompt or current_prompt == DEFAULT_PROMPT:
                    user_input[CONF_PROMPT] = PROMPT_PRESETS[preset]

            title = user_input.get(CONF_MODEL, "Conversation Agent")
            if step_id == "reconfigure":
                subentry = self._get_reconfigure_subentry()
                return self.async_update_and_abort(
                    entry,
                    subentry,
                    data=user_input,
                    title=title,
                )
            return self.async_create_entry(title=title, data=user_input)

        # Build suggested values from existing subentry data if reconfiguring
        suggested_values: dict[str, Any] = {}
        if step_id == "reconfigure":
            subentry = self._get_reconfigure_subentry()
            suggested_values = dict(subentry.data)

        # Model selector — dropdown if models fetched, text input otherwise
        if models:
            # Ensure the current model is in the options list even if the
            # API no longer reports it (e.g. model was removed from server)
            current_model = suggested_values.get(CONF_MODEL)
            if current_model and current_model not in models:
                models = [current_model, *models]
            model_selector = SelectSelector(
                SelectSelectorConfig(
                    options=models,
                    mode=SelectSelectorMode.DROPDOWN,
                    custom_value=True,
                )
            )
        else:
            model_selector = TextSelector(TextSelectorConfig())

        schema_fields: dict[Any, Any] = {
            vol.Optional(CONF_PROMPT_PRESET, default=PROMPT_PRESET_CUSTOM): SelectSelector(
                SelectSelectorConfig(
                    options=[{"value": k, "label": k.replace("_", " ").title()} for k in PROMPT_PRESETS],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(CONF_MODEL, default=DEFAULT_MODEL): model_selector,
            vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): TemplateSelector(),
            vol.Optional(CONF_LLM_HASS_API): SelectSelector(
                SelectSelectorConfig(
                    options=[api.id for api in llm.async_get_apis(self.hass)],
                    mode=SelectSelectorMode.DROPDOWN,
                    multiple=True,
                )
            ),
            vol.Optional(CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE): NumberSelector(
                NumberSelectorConfig(min=0.0, max=2.0, step=0.1, mode=NumberSelectorMode.SLIDER)
            ),
            vol.Optional(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS): NumberSelector(
                NumberSelectorConfig(min=1, max=128000, step=1, mode=NumberSelectorMode.BOX)
            ),
            vol.Optional(CONF_TOP_P, default=DEFAULT_TOP_P): NumberSelector(
                NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode=NumberSelectorMode.SLIDER)
            ),
            vol.Optional(CONF_MAX_CONTEXT_TOKENS, default=DEFAULT_MAX_CONTEXT_TOKENS): NumberSelector(
                NumberSelectorConfig(min=0, max=1000000, step=1000, mode=NumberSelectorMode.BOX)
            ),
            vol.Optional(CONF_STOP_SEQUENCES): TextSelector(TextSelectorConfig()),
            vol.Optional(CONF_MAX_RETRIES, default=DEFAULT_MAX_RETRIES): NumberSelector(
                NumberSelectorConfig(min=0, max=10, step=1, mode=NumberSelectorMode.BOX)
            ),
            vol.Optional(CONF_FALLBACK_MODEL): TextSelector(TextSelectorConfig()),
            vol.Optional(CONF_VOICE_MODE, default=False): BooleanSelector(),
            vol.Optional(CONF_MEMORY_ENABLED, default=False): BooleanSelector(),
            vol.Optional(CONF_MEMORY_MAX_MESSAGES, default=DEFAULT_MEMORY_MAX_MESSAGES): NumberSelector(
                NumberSelectorConfig(min=2, max=100, step=1, mode=NumberSelectorMode.BOX)
            ),
            vol.Optional(CONF_CUSTOM_TOOLS): TextSelector(TextSelectorConfig(multiline=True)),
        }

        # Add provider-specific fields
        if api_type in (API_TYPE_OPENAI, API_TYPE_OPENAI_RESPONSES):
            schema_fields[vol.Optional(CONF_RESPONSE_FORMAT)] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": "text", "label": "Text (default)"},
                        {"value": "json_object", "label": "JSON Object"},
                        {"value": "json_schema", "label": "JSON Schema"},
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
            schema_fields[vol.Optional(CONF_JSON_SCHEMA)] = TextSelector(TextSelectorConfig(multiline=True))
        elif api_type == API_TYPE_ANTHROPIC:
            # Anthropic: JSON schema via system prompt injection
            schema_fields[vol.Optional(CONF_RESPONSE_FORMAT)] = SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": "text", "label": "Text (default)"},
                        {"value": "json_schema", "label": "JSON Schema"},
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            )
            schema_fields[vol.Optional(CONF_JSON_SCHEMA)] = TextSelector(TextSelectorConfig(multiline=True))

        if api_type == API_TYPE_OPENAI:
            schema_fields[vol.Optional(CONF_SEED)] = NumberSelector(
                NumberSelectorConfig(min=0, max=2147483647, step=1, mode=NumberSelectorMode.BOX)
            )

        # Add extended thinking fields for Anthropic and Responses API
        if api_type in (API_TYPE_ANTHROPIC, API_TYPE_OPENAI_RESPONSES):
            schema_fields[vol.Optional(CONF_EXTENDED_THINKING, default=False)] = BooleanSelector()
            schema_fields[vol.Optional(CONF_THINKING_BUDGET, default=DEFAULT_THINKING_BUDGET)] = NumberSelector(
                NumberSelectorConfig(min=1000, max=128000, step=1000, mode=NumberSelectorMode.BOX)
            )

        schema = vol.Schema(schema_fields)

        if suggested_values:
            schema = self.add_suggested_values_to_schema(schema, suggested_values)

        return self.async_show_form(step_id=step_id, data_schema=schema)
