# HomeAssistant Custom Conversation Agent Integration Research

## Table of Contents

1. [Overview](#1-overview)
2. [HomeAssistant Custom Integration Fundamentals](#2-homeassistant-custom-integration-fundamentals)
3. [Integration File Structure](#3-integration-file-structure)
4. [The manifest.json File](#4-the-manifestjson-file)
5. [The Conversation Entity API](#5-the-conversation-entity-api)
6. [The Assist Pipeline Architecture](#6-the-assist-pipeline-architecture)
7. [Config Flow (UI Configuration)](#7-config-flow-ui-configuration)
8. [Options Flow (Runtime Configuration)](#8-options-flow-runtime-configuration)
9. [Strings and Translations](#9-strings-and-translations)
10. [Analysis of Existing OpenAI-Compatible Integrations](#10-analysis-of-existing-openai-compatible-integrations)
11. [Recommended Architecture for Our Integration](#11-recommended-architecture-for-our-integration)
12. [Complete Implementation Reference](#12-complete-implementation-reference)
13. [References and Links](#13-references-and-links)

---

## 1. Overview

HomeAssistant supports custom integrations (also called custom components) that can extend
its functionality. A **Conversation Agent** is a specific type of integration that plugs into
HomeAssistant's **Assist** voice/text pipeline to process natural language input and return
responses. This is how integrations like OpenAI Conversation, Google Generative AI, and Ollama
provide LLM-powered conversation capabilities within HomeAssistant.

Our goal is to build a custom integration that acts as a conversation agent, connecting to any
**OpenAI API-compatible endpoint** (such as llama-swap, Ollama, vLLM, LiteLLM, or any other
OpenAI-compatible server). This will allow HomeAssistant's Assist to use a self-hosted LLM
for natural language processing and device control.

### Key Concepts

- **Custom Component**: A Python package placed in `<config_dir>/custom_components/<domain>/`
- **Conversation Entity**: An entity type that processes text input and returns text responses
- **Assist Pipeline**: HomeAssistant's voice/text processing pipeline (Wake Word -> STT -> Intent/Conversation -> TTS)
- **Config Flow**: UI-based setup wizard for integrations
- **Options Flow**: UI-based settings editor for already-configured integrations

---

## 2. HomeAssistant Custom Integration Fundamentals

### How Custom Integrations Work

Custom integrations live in the `<config_dir>/custom_components/` directory. HomeAssistant
discovers them at startup by scanning for directories containing a valid `manifest.json` file.

The minimum requirements for a custom integration are:

1. A directory under `custom_components/` with your domain name
2. A `manifest.json` file describing the integration
3. An `__init__.py` file with setup logic
4. A `version` key in `manifest.json` (required for custom integrations, not for core)

### Setup Lifecycle

HomeAssistant calls these functions during integration setup:

```python
# For YAML-configured integrations (legacy):
async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration from YAML configuration."""
    return True

# For UI-configured integrations (config flow):
async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry
) -> bool:
    """Set up from a config entry (created by config flow)."""
    # Initialize your integration here
    # Forward setup to platforms (conversation, sensor, etc.)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True

async def async_unload_entry(
    hass: HomeAssistant,
    entry: ConfigEntry
) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
```

### Platform Concept

An integration can provide entities on one or more **platforms**. For a conversation agent,
the platform is `Platform.CONVERSATION`. This tells HomeAssistant to look for a
`conversation.py` file in your integration that provides conversation entities.

```python
from homeassistant.const import Platform

PLATFORMS = [Platform.CONVERSATION]
```

---

## 3. Integration File Structure

A conversation agent integration should have this file structure:

```
custom_components/
  llm_home_controller/          # Your domain name
    __init__.py                  # Integration setup
    config_flow.py               # UI configuration wizard
    const.py                     # Constants (domain name, defaults, config keys)
    conversation.py              # Conversation entity implementation
    manifest.json                # Integration metadata
    strings.json                 # UI text and translations
    translations/
      en.json                    # English translations (copy of strings.json)
```

### File Purposes

| File | Purpose |
|------|---------|
| `__init__.py` | Entry point. Contains `async_setup_entry()` and `async_unload_entry()`. Forwards setup to conversation platform. |
| `config_flow.py` | Defines the UI setup wizard. Handles API URL, API key, model selection validation. |
| `const.py` | Constants: DOMAIN, config keys, default values for model, temperature, etc. |
| `conversation.py` | The `ConversationEntity` subclass that handles message processing. Core logic. |
| `manifest.json` | Metadata: name, version, dependencies, requirements, etc. |
| `strings.json` | All user-facing text for the config flow, options, errors, etc. |
| `translations/en.json` | Copy of `strings.json` for English locale. |

---

## 4. The manifest.json File

### Complete Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `domain` | Yes | Unique identifier (lowercase, underscores). e.g., `"llm_home_controller"` |
| `name` | Yes | Display name. e.g., `"LLM Home Controller"` |
| `version` | Yes (custom) | SemVer or CalVer version string. e.g., `"1.0.0"` |
| `documentation` | Yes | URL to documentation |
| `codeowners` | Yes | GitHub usernames of maintainers |
| `dependencies` | Yes | Required HA integrations. e.g., `["conversation"]` |
| `after_dependencies` | No | Integrations to load before this one |
| `requirements` | Yes | Python package dependencies (pip format) |
| `config_flow` | No | Boolean: `true` if integration has a config flow UI |
| `integration_type` | Yes | One of: `device`, `entity`, `hardware`, `helper`, `hub`, `service`, `system`, `virtual` |
| `iot_class` | Yes | How it connects: `cloud_polling`, `cloud_push`, `local_polling`, `local_push`, `calculated` |
| `issue_tracker` | No | URL for bug reports |
| `loggers` | No | Logger names used by requirements |
| `single_config_entry` | No | Boolean: restrict to one config entry |
| `quality_scale` | No | Quality rating (core only) |

### Example manifest.json for Our Integration

```json
{
  "domain": "llm_home_controller",
  "name": "LLM Home Controller",
  "version": "0.1.0",
  "documentation": "https://github.com/your-repo/llm-home-controller",
  "issue_tracker": "https://github.com/your-repo/llm-home-controller/issues",
  "codeowners": ["@your-github-username"],
  "dependencies": ["conversation"],
  "after_dependencies": ["assist_pipeline", "intent"],
  "requirements": ["aiohttp>=3.8.0"],
  "config_flow": true,
  "integration_type": "service",
  "iot_class": "local_polling"
}
```

### Key Design Decisions

- **`integration_type: "service"`**: This is a service integration, not a device or hub.
  It provides a single service (conversation) rather than managing physical devices.
- **`iot_class: "local_polling"`**: If connecting to a local LLM server.
  Use `"cloud_polling"` if the API endpoint is remote.
- **`dependencies: ["conversation"]`**: Required because we implement a conversation entity.
- **`after_dependencies: ["assist_pipeline", "intent"]`**: Ensures the Assist pipeline and
  intent system are ready before our integration loads.
- **`requirements`**: We use `aiohttp` (already bundled with HA) for HTTP requests to the
  OpenAI-compatible API. If using the `openai` Python library instead, specify
  `["openai>=1.0.0"]`.

---

## 5. The Conversation Entity API

### Core Classes and Data Structures

The conversation framework is defined in `homeassistant.components.conversation`. The key
classes are:

#### ConversationEntity (Base Class)

Source: `homeassistant/components/conversation/entity.py`

```python
from homeassistant.components.conversation import ConversationEntity

class MyConversationEntity(ConversationEntity):
    """My custom conversation agent."""

    # Required: declare supported languages
    @property
    def supported_languages(self) -> list[str] | str:
        """Return list of supported languages or '*' for all."""
        return "*"  # Support all languages

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Process a user message and return a response."""
        # Your LLM API call goes here
        ...
```

The `ConversationEntity` extends `RestoreEntity` and provides:
- Automatic state management (tracks last activity timestamp)
- Chat log/history integration
- Streaming support (opt-in)
- Entity registration with the conversation component

#### ConversationInput

Source: `homeassistant/components/conversation/models.py`

```python
@dataclass
class ConversationInput:
    """Represents user input to a conversation agent."""

    text: str                              # The user's message text
    context: Context                       # HA context for actions
    conversation_id: str | None            # For multi-turn conversations
    device_id: str | None                  # Device that initiated the conversation
    satellite_id: str | None               # Satellite device ID (voice assistants)
    language: str                          # Language code (e.g., "en")
    agent_id: str | None                   # Which agent to process this
    extra_system_prompt: str | None        # Additional LLM instructions

    def as_dict(self) -> dict:
        """Return dictionary representation."""
        ...

    def as_llm_context(self, ...) -> llm.LLMContext:
        """Create an LLM context object."""
        ...
```

#### ConversationResult

Source: `homeassistant/components/conversation/models.py`

```python
@dataclass
class ConversationResult:
    """Result of processing user input."""

    response: intent.IntentResponse        # The response object
    conversation_id: str | None = None     # For conversation continuity
    continue_conversation: bool = False    # Whether to keep listening
```

#### ChatLog

Source: `homeassistant/components/conversation/chat_log.py`

The `ChatLog` class manages conversation history. Key methods:

```python
class ChatLog:
    content: list[...]                     # Message history
    conversation_id: str                   # Unique conversation ID
    llm_api: llm.APIInstance | None        # LLM API for tool execution

    async def async_add_user_content(self, content: UserContent) -> None:
        """Add user message to the log."""

    async def async_add_assistant_content_without_tools(
        self, content: AssistantContent
    ) -> None:
        """Add assistant response without tool calls."""

    async def async_provide_llm_data(
        self,
        user_input: ConversationInput,
        llm_api: ...,
        user_prompt: str,
        ...,
    ) -> None:
        """Set up system prompt and LLM context."""

    @property
    def continue_conversation(self) -> bool:
        """Whether the last message expects a follow-up."""
```

#### IntentResponse

Used to construct the response:

```python
from homeassistant.helpers import intent

response = intent.IntentResponse(language=user_input.language)
response.async_set_speech("Hello! How can I help you?")
```

### Supported Features

Features are declared via the `ConversationEntityFeature` enum:

```python
from homeassistant.components.conversation import ConversationEntityFeature

class MyConversationEntity(ConversationEntity):
    _attr_supported_features = ConversationEntityFeature.CONTROL
```

- **`CONTROL`**: Indicates the agent can control HomeAssistant devices and automations.
  This is important for agents that should be able to turn lights on/off, etc.

### Streaming Support

To enable streaming responses:

```python
class MyConversationEntity(ConversationEntity):
    _attr_supports_streaming = True
```

### Complete Minimal Implementation

```python
"""Conversation entity for LLM Home Controller."""

from homeassistant.components.conversation import (
    ChatLog,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.helpers import intent

from .const import DOMAIN


class LLMConversationEntity(ConversationEntity):
    """LLM-powered conversation entity."""

    _attr_has_entity_name = True
    _attr_name = None  # Uses device name
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, entry, api_url, api_key, model):
        """Initialize."""
        self._entry = entry
        self._api_url = api_url
        self._api_key = api_key
        self._model = model
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> str:
        """Return supported languages."""
        return "*"  # All languages

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Handle a user message."""
        # 1. Build messages for the API
        messages = self._build_messages(user_input, chat_log)

        # 2. Call the OpenAI-compatible API
        response_text = await self._call_api(messages)

        # 3. Add response to chat log
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=user_input.agent_id,
                content=response_text,
            )
        )

        # 4. Build and return the result
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)

        return ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
        )

    def _build_messages(self, user_input, chat_log):
        """Convert chat log to OpenAI message format."""
        messages = []

        # Add system prompt
        messages.append({
            "role": "system",
            "content": "You are a helpful home assistant."
        })

        # Convert chat log entries to messages
        for entry in chat_log.content:
            if hasattr(entry, 'role'):
                messages.append({
                    "role": entry.role,
                    "content": entry.content,
                })

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_input.text,
        })

        return messages

    async def _call_api(self, messages):
        """Call the OpenAI-compatible API."""
        import aiohttp

        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": messages,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._api_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
```

---

## 6. The Assist Pipeline Architecture

### Pipeline Components

The Assist pipeline processes voice/text input through sequential stages:

```
[Wake Word] -> [STT] -> [Conversation Agent] -> [TTS]
     |            |              |                  |
  optional    Speech-to-     Our agent          Text-to-
  trigger     Text engine    processes           Speech
                             the text            output
```

Each stage is optional. For text-only input (e.g., typing in the UI), the pipeline
skips Wake Word and STT, going directly to the Conversation Agent.

### How Our Agent Fits In

1. User types or speaks a message
2. If voice, STT converts speech to text
3. The **Conversation Agent** (our integration) receives the text as a `ConversationInput`
4. Our agent calls the LLM API and returns a `ConversationResult`
5. If voice, TTS converts the response text to speech
6. The response is delivered to the user

### Pipeline Configuration

Users configure which conversation agent to use in:
- **Settings -> Voice Assistants -> Assist Pipeline** in the HA UI
- Each pipeline can use a different conversation agent
- Our agent appears in the dropdown once registered as a conversation entity

### WebSocket API

The pipeline communicates via WebSocket events:

| Event | Description |
|-------|-------------|
| `run-start` | Pipeline initialized with metadata |
| `stt-start` | Speech-to-text processing begins |
| `stt-end` | STT complete, text available |
| `intent-start` | Conversation agent processing begins |
| `intent-end` | Agent response available |
| `tts-start` | Text-to-speech processing begins |
| `tts-end` | TTS complete, audio available |
| `run-end` | Pipeline complete |
| `error` | Error occurred with fault code |

### Agent Registration

When our conversation entity is added to HomeAssistant, it automatically registers
as an available conversation agent. The `ConversationEntity` base class handles this
in `async_internal_added_to_hass()`.

For the legacy `AbstractConversationAgent` approach (deprecated in favor of entity-based):

```python
# In __init__.py (OLD approach - use ConversationEntity instead)
from homeassistant.components.conversation import async_set_agent

async def async_setup_entry(hass, entry):
    agent = MyAgent(hass, entry)
    async_set_agent(hass, entry, agent)
```

The modern approach is to use `ConversationEntity` which handles registration automatically.

---

## 7. Config Flow (UI Configuration)

### Overview

Config flows provide a step-by-step UI wizard for setting up an integration. For our
integration, the config flow will collect:

1. **API URL** (e.g., `http://192.168.1.100:8080`)
2. **API Key** (optional, for authenticated endpoints)
3. **Model name** (e.g., `llama-3.1-8b`)

### Implementation

Create `config_flow.py`:

```python
"""Config flow for LLM Home Controller."""

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY
from homeassistant.helpers.selector import (
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_API_URL,
    CONF_MODEL,
    DEFAULT_API_URL,
    DEFAULT_MODEL,
    DOMAIN,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_URL, default=DEFAULT_API_URL): TextSelector(
            TextSelectorConfig(type=TextSelectorType.URL)
        ),
        vol.Optional(CONF_API_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
        vol.Required(CONF_MODEL, default=DEFAULT_MODEL): TextSelector(),
    }
)


async def validate_input(hass, data):
    """Validate the user input by testing the API connection."""
    url = data[CONF_API_URL].rstrip("/")
    headers = {"Content-Type": "application/json"}

    api_key = data.get(CONF_API_KEY)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession() as session:
        # Test by listing models
        async with session.get(
            f"{url}/v1/models",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 401:
                raise InvalidAuth
            if resp.status != 200:
                raise CannotConnect
            return await resp.json()


class LLMHomeControllerConfigFlow(
    config_entries.ConfigFlow, domain=DOMAIN
):
    """Handle a config flow for LLM Home Controller."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            try:
                await validate_input(self.hass, user_input)
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Exception:  # noqa: BLE001
                errors["base"] = "unknown"
            else:
                # Validation passed, create the entry
                return self.async_create_entry(
                    title=f"LLM ({user_input[CONF_MODEL]})",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors,
        )


class InvalidAuth(Exception):
    """Error to indicate invalid authentication."""


class CannotConnect(Exception):
    """Error to indicate connection failure."""
```

### Config Flow Key Methods

| Method | Purpose |
|--------|---------|
| `async_step_user(user_input)` | First step when user adds integration |
| `async_step_reauth(entry_data)` | Handle credential expiration |
| `async_step_reconfigure(user_input)` | Allow user to change config |
| `self.async_show_form(...)` | Display a form to the user |
| `self.async_create_entry(...)` | Save config and finish setup |
| `self.async_abort(reason=...)` | Cancel the flow with a reason |

### Data Schema and Selectors

HomeAssistant provides built-in selectors for form fields:

```python
from homeassistant.helpers.selector import (
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    BooleanSelector,
    TemplateSelector,
)

# Examples:
vol.Required("api_url"): TextSelector(
    TextSelectorConfig(type=TextSelectorType.URL)
)
vol.Optional("api_key"): TextSelector(
    TextSelectorConfig(type=TextSelectorType.PASSWORD)
)
vol.Required("model"): TextSelector()
vol.Optional("temperature", default=1.0): NumberSelector(
    NumberSelectorConfig(min=0.0, max=2.0, step=0.1, mode=NumberSelectorMode.SLIDER)
)
vol.Optional("max_tokens", default=3000): NumberSelector(
    NumberSelectorConfig(min=1, max=128000, step=1, mode=NumberSelectorMode.BOX)
)
vol.Optional("system_prompt"): TemplateSelector()
```

---

## 8. Options Flow (Runtime Configuration)

### Overview

Options flow lets users change settings after initial setup (e.g., changing the model,
adjusting temperature, editing the system prompt).

### Implementation

Add to `config_flow.py`:

```python
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    OptionsFlow,
)

class LLMHomeControllerConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Get the options flow handler."""
        return LLMOptionsFlowHandler()

    # ... async_step_user etc.


class LLMOptionsFlowHandler(OptionsFlow):
    """Handle options flow."""

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Get current options or defaults
        options = self.config_entry.options

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_SYSTEM_PROMPT,
                        default=options.get(CONF_SYSTEM_PROMPT, DEFAULT_PROMPT),
                    ): TemplateSelector(),
                    vol.Optional(
                        CONF_MODEL,
                        default=options.get(CONF_MODEL, DEFAULT_MODEL),
                    ): TextSelector(),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0.0, max=2.0, step=0.1,
                            mode=NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=1, max=128000, step=1,
                            mode=NumberSelectorMode.BOX,
                        )
                    ),
                }
            ),
        )
```

### Using OptionsFlowWithReload

If your integration needs to reload when options change (recommended for conversation agents
since options affect behavior):

```python
from homeassistant.config_entries import OptionsFlowWithReload

class LLMOptionsFlowHandler(OptionsFlowWithReload):
    """Handle options with automatic reload."""

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        # Same as above
        ...
```

This avoids the need for manual update listeners.

### Accessing Options at Runtime

In your conversation entity:

```python
class LLMConversationEntity(ConversationEntity):
    def __init__(self, entry: ConfigEntry):
        self._entry = entry

    @property
    def _model(self):
        # Options override data, with fallback to data then defaults
        return self._entry.options.get(
            CONF_MODEL,
            self._entry.data.get(CONF_MODEL, DEFAULT_MODEL)
        )

    @property
    def _temperature(self):
        return self._entry.options.get(
            CONF_TEMPERATURE,
            DEFAULT_TEMPERATURE
        )
```

---

## 9. Strings and Translations

### strings.json Format

The `strings.json` file provides all user-facing text. HomeAssistant uses this for the
config flow UI, options flow, and error messages.

```json
{
  "config": {
    "step": {
      "user": {
        "title": "Connect to LLM Server",
        "description": "Enter your OpenAI-compatible API server details.",
        "data": {
          "api_url": "API URL",
          "api_key": "API Key (optional)",
          "model": "Model"
        },
        "data_description": {
          "api_url": "The base URL of your OpenAI-compatible API server (e.g., http://192.168.1.100:8080)",
          "api_key": "API key for authentication. Leave empty if not required.",
          "model": "The model identifier to use for chat completions."
        }
      }
    },
    "error": {
      "cannot_connect": "Failed to connect to the API server. Check the URL and ensure the server is running.",
      "invalid_auth": "Invalid API key. Please check your credentials.",
      "unknown": "An unexpected error occurred."
    },
    "abort": {
      "already_configured": "This integration is already configured."
    }
  },
  "options": {
    "step": {
      "init": {
        "title": "LLM Home Controller Options",
        "data": {
          "system_prompt": "System Prompt",
          "model": "Model",
          "temperature": "Temperature",
          "max_tokens": "Maximum Tokens"
        },
        "data_description": {
          "system_prompt": "Instructions for the AI assistant's behavior.",
          "temperature": "Controls randomness. Lower values are more focused, higher values are more creative.",
          "max_tokens": "Maximum number of tokens in the response."
        }
      }
    }
  }
}
```

### Translation Files

For custom integrations, copy `strings.json` to `translations/en.json`. You can add
additional language files (e.g., `translations/de.json` for German).

For core integrations, translations are generated automatically from `strings.json` using
`python3 -m script.translations develop`.

---

## 10. Analysis of Existing OpenAI-Compatible Integrations

### 10.1 Official OpenAI Conversation (Core Integration)

**Source**: `homeassistant/components/openai_conversation/`
**URL**: https://github.com/home-assistant/core/tree/dev/homeassistant/components/openai_conversation

#### File Structure
```
openai_conversation/
  __init__.py          # Setup, service registration, migration
  ai_task.py           # AI task entity
  config_flow.py       # Multi-step config flow with subentries
  const.py             # Constants and defaults
  conversation.py      # ConversationEntity implementation
  entity.py            # Base LLM entity with OpenAI API logic
  icons.json           # Icons
  manifest.json        # Metadata
  quality_scale.yaml   # Quality metrics
  services.yaml        # Service definitions
  strings.json         # UI text
  tts.py               # Text-to-speech entity
```

#### Key Architecture Decisions
- Uses `ConfigSubentry` pattern for managing multiple conversation/TTS/AI task instances
  under a single API key config entry
- The `OpenAIConversationEntity` class extends both `ConversationEntity` and
  `OpenAIBaseLLMEntity` (which contains the API call logic)
- Supports streaming responses (`_attr_supports_streaming = True`)
- Supports the `CONTROL` feature for device management
- Uses the `openai` Python library (version 2.15.0)
- Registers/unregisters agent in `async_added_to_hass`/`async_will_remove_from_hass`

#### Conversation Entity (conversation.py) Pattern

```python
class OpenAIConversationEntity(
    conversation.ConversationEntity,
    OpenAIBaseLLMEntity,
):
    _attr_supports_streaming = True

    @property
    def supported_languages(self) -> list[str]:
        return MATCH_ALL  # "*" - all languages

    @property
    def supported_features(self) -> ConversationEntityFeature:
        # Enable CONTROL if LLM API is configured
        if llm_api_configured:
            return ConversationEntityFeature.CONTROL
        return ConversationEntityFeature(0)

    async def async_added_to_hass(self) -> None:
        """Register as conversation agent when added."""
        await super().async_added_to_hass()
        # Agent registration happens automatically via ConversationEntity

    async def async_will_remove_from_hass(self) -> None:
        """Unregister as conversation agent when removed."""
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Process the message."""
        # 1. Prepare LLM context (system prompt, tools)
        # 2. Call OpenAI API via _async_handle_chat_log()
        # 3. Return ConversationResult
```

#### API Communication (entity.py) Pattern

The `OpenAIBaseLLMEntity._async_handle_chat_log()` method:
1. Converts `ChatLog` entries to OpenAI message format
2. Configures model parameters (temperature, max_tokens, reasoning, etc.)
3. Sets up tools (web search, code interpreter, image generation)
4. Calls `client.chat.completions.create()` with streaming
5. Iterates on tool calls (max 10 iterations)
6. Handles errors (rate limits, authentication, quota)

#### manifest.json

```json
{
  "domain": "openai_conversation",
  "name": "OpenAI",
  "after_dependencies": ["assist_pipeline", "intent"],
  "codeowners": [],
  "config_flow": true,
  "dependencies": ["conversation"],
  "documentation": "https://www.home-assistant.io/integrations/openai_conversation",
  "integration_type": "service",
  "iot_class": "cloud_polling",
  "quality_scale": "bronze",
  "requirements": ["openai==2.15.0"]
}
```

### 10.2 Extended OpenAI Conversation (Community Custom Component)

**Source**: https://github.com/jekalmin/extended_openai_conversation
**Version**: 2.0.0

#### Key Differences from Official
- Supports **multiple API providers** (OpenAI, Azure, custom endpoints)
- Configurable `base_url` for OpenAI-compatible APIs
- More extensive function/tool calling configuration
- Supports custom functions defined by users
- Uses subentry pattern for multiple conversation personalities
- Dependencies include: conversation, energy, history, recorder, rest, scrape

#### Config Flow Pattern
- Step 1: API credentials (name, API key, base URL, API version, organization,
  skip authentication toggle, API provider selection)
- Step 2: Conversation options per subentry (model, temperature, top_p, max_tokens,
  function calling config, context management)
- Validates by instantiating an authenticated client and testing connectivity

#### manifest.json

```json
{
  "domain": "extended_openai_conversation",
  "name": "Extended OpenAI Conversation",
  "version": "2.0.0",
  "config_flow": true,
  "dependencies": ["conversation", "energy", "history", "recorder", "rest", "scrape"],
  "documentation": "https://github.com/jekalmin/extended_openai_conversation",
  "integration_type": "service",
  "iot_class": "cloud_polling",
  "issue_tracker": "https://github.com/jekalmin/extended_openai_conversation/issues",
  "requirements": ["openai~=2.8.0"],
  "codeowners": ["@jekalmin"]
}
```

### 10.3 AI Conversation Agent (Community Custom Component)

**Source**: https://github.com/hasscc/ai-conversation
**Version**: 0.1.0

#### Key Architecture Patterns
- **Direct HTTP calls** using `aiohttp` instead of the `openai` Python library
- Implements `BasicEntity` base class with `async_chat_completions()` method
- Posts directly to `/chat/completions` endpoint
- Uses Bearer token authentication
- Supports multiple OpenAI-format providers
- Handles tool calling with iterative loops (MAX_TOOL_ITERATIONS)
- Supports streaming delta content

#### API Communication Pattern

```python
class HassEntry:
    """Manages API connection."""

    async def async_post(self, path, **kwargs):
        """POST to the API."""
        url = f"{self.base_url}/{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.session.post(url, headers=headers, **kwargs) as resp:
            return await resp.json()

    async def async_chat_completions(self, **kwargs):
        """Call chat/completions endpoint."""
        return await self.async_post("chat/completions", json=kwargs)
```

#### manifest.json

```json
{
  "domain": "ai_conversation",
  "name": "AI Conversation Agent",
  "version": "0.1.0",
  "config_flow": true,
  "dependencies": ["conversation"],
  "after_dependencies": ["assist_pipeline", "intent"],
  "documentation": "https://github.com/hasscc/ai-conversation",
  "integration_type": "service",
  "iot_class": "cloud_polling",
  "requirements": ["voluptuous-openapi>=0.1.0", "sentence-stream>=1.0.0"],
  "codeowners": ["@al-one"]
}
```

---

## 11. Recommended Architecture for Our Integration

### Design Principles

1. **Use `aiohttp` directly** instead of the `openai` Python library. This avoids
   unnecessary dependencies and gives us full control over the HTTP requests. The
   OpenAI chat completions API is simple enough to call directly.

2. **Use `ConversationEntity`** (modern approach) rather than `AbstractConversationAgent`
   (legacy approach). ConversationEntity provides automatic entity registration, state
   management, and chat log integration.

3. **Keep it simple** for the first version. Start with basic chat completions support,
   then add tool calling and streaming in later iterations.

4. **Make the API URL configurable** so it works with any OpenAI-compatible server
   (llama-swap, Ollama, vLLM, LiteLLM, text-generation-webui, etc.).

### Proposed File Structure

```
custom_components/
  llm_home_controller/
    __init__.py          # async_setup_entry, async_unload_entry
    config_flow.py       # Config flow + Options flow
    const.py             # DOMAIN, config keys, defaults
    conversation.py      # ConversationEntity with API client
    manifest.json        # Integration metadata
    strings.json         # UI text
    translations/
      en.json            # English (copy of strings.json)
```

### Proposed const.py

```python
"""Constants for LLM Home Controller."""

DOMAIN = "llm_home_controller"

# Config keys
CONF_API_URL = "api_url"
CONF_MODEL = "model"
CONF_SYSTEM_PROMPT = "system_prompt"
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
CONF_TOP_P = "top_p"

# Defaults
DEFAULT_API_URL = "http://localhost:8080"
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_SYSTEM_PROMPT = """You are a helpful home assistant AI. You can help control
smart home devices and answer questions. Be concise and helpful."""
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOP_P = 1.0
```

### Proposed __init__.py

```python
"""LLM Home Controller integration."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN

PLATFORMS = [Platform.CONVERSATION]

type LLMConfigEntry = ConfigEntry


async def async_setup_entry(hass: HomeAssistant, entry: LLMConfigEntry) -> bool:
    """Set up LLM Home Controller from a config entry."""
    # Store the entry data for access by the conversation entity
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = entry.data

    # Forward setup to the conversation platform
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: LLMConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok
```

### Proposed conversation.py

```python
"""Conversation entity for LLM Home Controller."""

import logging

import aiohttp

from homeassistant.components.conversation import (
    AssistantContent,
    ChatLog,
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_API_URL,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entity."""
    async_add_entities([
        LLMConversationEntity(config_entry)
    ])


class LLMConversationEntity(ConversationEntity):
    """LLM-powered conversation entity."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the entity."""
        self._entry = entry
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> str:
        """Return supported languages."""
        return "*"

    @property
    def _api_url(self) -> str:
        return self._entry.data.get(CONF_API_URL, "").rstrip("/")

    @property
    def _api_key(self) -> str | None:
        return self._entry.data.get("api_key")

    @property
    def _model(self) -> str:
        return self._entry.options.get(
            CONF_MODEL, self._entry.data.get(CONF_MODEL, DEFAULT_MODEL)
        )

    @property
    def _system_prompt(self) -> str:
        return self._entry.options.get(CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT)

    @property
    def _temperature(self) -> float:
        return self._entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)

    @property
    def _max_tokens(self) -> int:
        return self._entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)

    @property
    def _top_p(self) -> float:
        return self._entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Process a message from the user."""
        try:
            # Build the messages payload
            messages = self._build_messages(user_input, chat_log)

            # Call the OpenAI-compatible API
            response_text = await self._async_call_api(messages)

            # Add the response to the chat log
            chat_log.async_add_assistant_content_without_tools(
                AssistantContent(
                    agent_id=user_input.agent_id,
                    content=response_text,
                )
            )

        except Exception as err:
            _LOGGER.error("Error calling LLM API: %s", err)
            response_text = f"Sorry, I encountered an error: {err}"

        # Build the intent response
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response_text)

        return ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
        )

    def _build_messages(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> list[dict]:
        """Build messages array for the API call."""
        messages = []

        # System prompt
        system_prompt = self._system_prompt
        if user_input.extra_system_prompt:
            system_prompt += f"\n\n{user_input.extra_system_prompt}"

        messages.append({
            "role": "system",
            "content": system_prompt,
        })

        # Convert chat log to message format
        for entry in chat_log.content:
            if hasattr(entry, "content") and entry.content:
                role = "assistant"
                if hasattr(entry, "role"):
                    role = entry.role
                messages.append({
                    "role": role,
                    "content": str(entry.content),
                })

        # Current user message
        messages.append({
            "role": "user",
            "content": user_input.text,
        })

        return messages

    async def _async_call_api(self, messages: list[dict]) -> str:
        """Call the OpenAI-compatible chat/completions endpoint."""
        headers = {"Content-Type": "application/json"}

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
        }

        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._api_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status == 401:
                    raise Exception("Authentication failed. Check your API key.")
                resp.raise_for_status()
                data = await resp.json()

        if "choices" not in data or not data["choices"]:
            raise Exception("No response from the LLM API.")

        return data["choices"][0]["message"]["content"]
```

### Proposed config_flow.py

```python
"""Config flow for LLM Home Controller."""

import logging

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_API_URL,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_SYSTEM_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_API_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


class LLMHomeControllerConfigFlow(
    config_entries.ConfigFlow, domain=DOMAIN
):
    """Handle a config flow."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial configuration step."""
        errors = {}

        if user_input is not None:
            try:
                await self._test_connection(user_input)
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except Exception:
                _LOGGER.exception("Unexpected error")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(
                    title=f"LLM ({user_input[CONF_MODEL]})",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_API_URL, default=DEFAULT_API_URL
                    ): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.URL)
                    ),
                    vol.Optional(CONF_API_KEY): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                    vol.Required(
                        CONF_MODEL, default=DEFAULT_MODEL
                    ): TextSelector(),
                }
            ),
            errors=errors,
        )

    async def _test_connection(self, data):
        """Test if we can connect to the API."""
        url = data[CONF_API_URL].rstrip("/")
        headers = {"Content-Type": "application/json"}
        api_key = data.get(CONF_API_KEY)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(
                    f"{url}/v1/models", headers=headers
                ) as resp:
                    if resp.status == 401:
                        raise InvalidAuth
                    if resp.status != 200:
                        raise CannotConnect
            except aiohttp.ClientError as err:
                raise CannotConnect from err

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow handler."""
        return LLMOptionsFlowHandler()


class LLMOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow."""

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        options = self.config_entry.options

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_SYSTEM_PROMPT,
                        default=options.get(
                            CONF_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT
                        ),
                    ): TemplateSelector(),
                    vol.Optional(
                        CONF_MODEL,
                        default=options.get(CONF_MODEL, DEFAULT_MODEL),
                    ): TextSelector(),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        default=options.get(
                            CONF_TEMPERATURE, DEFAULT_TEMPERATURE
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0.0, max=2.0, step=0.1,
                            mode=NumberSelectorMode.SLIDER,
                        )
                    ),
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        default=options.get(
                            CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS
                        ),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=1, max=128000, step=1,
                            mode=NumberSelectorMode.BOX,
                        )
                    ),
                    vol.Optional(
                        CONF_TOP_P,
                        default=options.get(CONF_TOP_P, DEFAULT_TOP_P),
                    ): NumberSelector(
                        NumberSelectorConfig(
                            min=0.0, max=1.0, step=0.05,
                            mode=NumberSelectorMode.SLIDER,
                        )
                    ),
                }
            ),
        )


class CannotConnect(Exception):
    """Error to indicate we cannot connect."""


class InvalidAuth(Exception):
    """Error to indicate there is invalid auth."""
```

### Proposed manifest.json

```json
{
  "domain": "llm_home_controller",
  "name": "LLM Home Controller",
  "version": "0.1.0",
  "documentation": "https://github.com/your-repo/llm-home-controller",
  "issue_tracker": "https://github.com/your-repo/llm-home-controller/issues",
  "codeowners": [],
  "dependencies": ["conversation"],
  "after_dependencies": ["assist_pipeline", "intent"],
  "requirements": [],
  "config_flow": true,
  "integration_type": "service",
  "iot_class": "local_polling"
}
```

### Proposed strings.json

```json
{
  "config": {
    "step": {
      "user": {
        "title": "Connect to LLM Server",
        "description": "Enter the connection details for your OpenAI-compatible API server.",
        "data": {
          "api_url": "API URL",
          "api_key": "API Key",
          "model": "Model"
        },
        "data_description": {
          "api_url": "Base URL of your OpenAI-compatible API (e.g., http://192.168.1.100:8080)",
          "api_key": "API key for authentication. Leave empty if not required.",
          "model": "Model identifier for chat completions (e.g., llama-3.1-8b)"
        }
      }
    },
    "error": {
      "cannot_connect": "Unable to connect to the API server. Check the URL and ensure the server is running.",
      "invalid_auth": "Authentication failed. Check your API key.",
      "unknown": "An unexpected error occurred."
    },
    "abort": {
      "already_configured": "This server is already configured."
    }
  },
  "options": {
    "step": {
      "init": {
        "title": "LLM Home Controller Options",
        "data": {
          "system_prompt": "System Prompt",
          "model": "Model",
          "temperature": "Temperature",
          "max_tokens": "Maximum Tokens",
          "top_p": "Top P"
        },
        "data_description": {
          "system_prompt": "Instructions that define the AI assistant's behavior and personality.",
          "model": "Model identifier for chat completions.",
          "temperature": "Controls randomness (0.0 = deterministic, 2.0 = very random).",
          "max_tokens": "Maximum number of tokens in the response.",
          "top_p": "Nucleus sampling parameter (1.0 = consider all tokens)."
        }
      }
    }
  }
}
```

---

## 12. Complete Implementation Reference

### How It All Fits Together

```
User speaks/types
       |
       v
[Assist Pipeline]
       |
       v
[ConversationEntity._async_handle_message()]
       |
       v
[Build messages array from ChatLog]
       |
       v
[POST /v1/chat/completions to LLM server]
       |
       v
[Parse response]
       |
       v
[Add to ChatLog + Build ConversationResult]
       |
       v
[Return IntentResponse with speech text]
       |
       v
[Assist Pipeline continues with TTS if voice]
```

### Entity Registration Flow

```
1. User adds integration via UI (config flow)
2. async_setup_entry() is called
3. Forwards to conversation platform
4. conversation.py async_setup_entry() creates entity
5. ConversationEntity base class registers with conversation component
6. Agent appears in Assist Pipeline dropdown
7. User selects agent in voice assistant settings
```

### OpenAI Chat Completions API Format

The endpoint we need to call:

**Request:**
```
POST {base_url}/v1/chat/completions
Content-Type: application/json
Authorization: Bearer {api_key}  (optional)

{
  "model": "llama-3.1-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Turn on the living room light"},
    {"role": "assistant", "content": "I'll turn on the living room light for you."},
    {"role": "user", "content": "Thanks, also dim it to 50%"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "top_p": 1.0,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "llama-3.1-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Done! I've dimmed the living room light to 50%."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 15,
    "total_tokens": 71
  }
}
```

### Future Enhancements

1. **Tool/Function Calling**: Allow the LLM to call HomeAssistant services
   (turn on lights, check sensor values, etc.) via OpenAI-compatible tool calling.
2. **Streaming Responses**: Implement streaming for faster perceived response times.
3. **Model Discovery**: Fetch available models from `/v1/models` and show them in a
   dropdown selector.
4. **Session Management**: Use `aiohttp.ClientSession` at the integration level (stored
   in `hass.data`) instead of creating new sessions per request.
5. **Conversation History**: Leverage ChatLog more deeply for multi-turn conversations.
6. **LLM API Integration**: Use HomeAssistant's built-in `llm.APIInstance` for tool
   execution to natively control devices through the Assist framework.
7. **HACS Compatibility**: Publish to HACS for easy community installation.

---

## 13. References and Links

### Official Documentation

- [Creating Your First Integration](https://developers.home-assistant.io/docs/creating_component_index/)
- [Conversation Entity Developer Docs](https://developers.home-assistant.io/docs/core/entity/conversation/)
- [Custom Conversation Agent](https://developers.home-assistant.io/docs/core/conversation/custom_agent/)
- [Integration Manifest](https://developers.home-assistant.io/docs/creating_integration_manifest/)
- [Config Flow Handler](https://developers.home-assistant.io/docs/config_entries_config_flow_handler/)
- [Options Flow Handler](https://developers.home-assistant.io/docs/config_entries_options_flow_handler/)
- [Assist Pipelines](https://developers.home-assistant.io/docs/voice/pipelines/)
- [Integration File Structure](https://github.com/home-assistant/developers.home-assistant/blob/master/docs/creating_integration_file_structure.md)

### Source Code References

- [Official OpenAI Conversation Integration](https://github.com/home-assistant/core/tree/dev/homeassistant/components/openai_conversation)
  - [__init__.py](https://github.com/home-assistant/core/blob/dev/homeassistant/components/openai_conversation/__init__.py)
  - [conversation.py](https://github.com/home-assistant/core/blob/dev/homeassistant/components/openai_conversation/conversation.py)
  - [entity.py](https://github.com/home-assistant/core/blob/dev/homeassistant/components/openai_conversation/entity.py)
  - [config_flow.py](https://github.com/home-assistant/core/blob/dev/homeassistant/components/openai_conversation/config_flow.py)
  - [const.py](https://github.com/home-assistant/core/blob/dev/homeassistant/components/openai_conversation/const.py)
- [Conversation Framework - entity.py](https://github.com/home-assistant/core/blob/dev/homeassistant/components/conversation/entity.py)
- [Conversation Framework - models.py](https://github.com/home-assistant/core/blob/dev/homeassistant/components/conversation/models.py)
- [Conversation Framework - chat_log.py](https://github.com/home-assistant/core/blob/dev/homeassistant/components/conversation/chat_log.py)

### Community Custom Integrations

- [Extended OpenAI Conversation](https://github.com/jekalmin/extended_openai_conversation) - Multi-provider OpenAI-compatible conversation agent with function calling
- [AI Conversation Agent](https://github.com/hasscc/ai-conversation) - Lightweight OpenAI-format compatible conversation agent
- [Custom Conversation](https://github.com/michelle-avery/custom-conversation) - Highly customizable conversation agent
- [Fallback Conversation Agent](https://community.home-assistant.io/t/custom-component-fallback-conversation-agent-use-multiple-conversation-agents-with-a-single-assist/688936) - Multi-agent fallback support

### Community Guides

- [Building a HA Custom Component Part 1: Project Structure](https://aarongodfrey.dev/home%20automation/building_a_home_assistant_custom_component_part_1/)
- [Building a HA Custom Component Part 3: Config Flow](https://aarongodfrey.dev/home%20automation/building_a_home_assistant_custom_component_part_3/)
- [Building a HA Custom Component Part 4: Options Flow](https://aarongodfrey.dev/home%20automation/building_a_home_assistant_custom_component_part_4/)
- [Developing Custom Integrations for HA - Getting Started](https://helgeklein.com/blog/developing-custom-integrations-for-home-assistant-getting-started/)
- [HA Custom Integrations Developer's Guide](https://thehomesmarthome.com/home-assistant-custom-integrations-developers-guide/)
- [AI Agents for the Smart Home (HA Blog)](https://www.home-assistant.io/blog/2024/06/07/ai-agents-for-the-smart-home/)
