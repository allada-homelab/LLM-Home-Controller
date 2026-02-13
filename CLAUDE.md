# LLM Home Controller

## Project Overview
- HomeAssistant custom integration, domain = `llm_home_controller`
- Conversation Agent for Assist that connects to OpenAI API-compatible endpoints
- Uses config subentry pattern: parent entry holds API URL + key, child subentries hold per-agent config (model, prompt, temperature, etc.)
- Compatible backends: llama-swap, Ollama, vLLM, LiteLLM, or any OpenAI-compatible API

## Architecture
- **Config subentry pattern**: One API connection (parent entry) → multiple conversation agents (child subentries)
- **Conversation entity**: Triple inheritance — `ConversationEntity + AbstractConversationAgent + BaseLLMEntity`
- **Tool calling loop**: Up to 10 iterations per message via `chat_log.async_add_delta_content_stream()` which auto-executes HA tools
- **Streaming**: `_attr_supports_streaming = True` with SSE → HA delta dict transformation
- **`ConversationEntity` extends `RestoreEntity`**, not `Entity` directly

## File Structure
```
custom_components/llm_home_controller/
  __init__.py          # Entry setup, session management, type alias
  config_flow.py       # Config flow + subentry flow (no separate options flow)
  const.py             # Constants and defaults
  conversation.py      # ConversationEntity (triple inheritance)
  entity.py            # Base LLM entity with _async_handle_chat_log()
  manifest.json        # Integration metadata
  strings.json         # UI strings
  translations/en.json # English translations
tests/
  conftest.py          # Test fixtures (MockChatLog, mock aiohttp, mock config entries)
  test_config_flow.py  # Config flow + subentry flow tests
  test_conversation.py # Conversation entity tests
  test_entity.py       # Base LLM entity tests
  test_init.py         # Entry setup/teardown tests
```

## Coding Conventions
- Override `_async_handle_message`, NEVER override `async_process` or `internal_async_process`
- Method chain: `chat_log.async_provide_llm_data()` → `_async_handle_chat_log()` → `async_get_result_from_chat_log()`
- Import `CONF_LLM_HASS_API` from `homeassistant.const` (NOT from `llm.py`)
- Import conversation types from `homeassistant.components.conversation` (all re-exported)
- Type alias: `type LLMHomeControllerConfigEntry = ConfigEntry[aiohttp.ClientSession]`
- Register agent via `conversation.async_set_agent(hass, config_entry, agent)` in `async_added_to_hass`
- Subentry settings modified via reconfigure flow, NOT separate OptionsFlow
- Use `entry.runtime_data` for shared `aiohttp.ClientSession`
- `async_get_chat_log` is a synchronous `@contextmanager`, not async
- `AssistantContent` has `thinking_content` and `native` fields
- `ToolInput` has an `external` bool field controlling auto-execution

## Key HA API Reference
| API | Import | Notes |
|-----|--------|-------|
| `ConversationEntity` | `homeassistant.components.conversation` | Base class, extends RestoreEntity |
| `AbstractConversationAgent` | `homeassistant.components.conversation` | Required mixin for agent registration |
| `ChatLog` | `homeassistant.components.conversation` | Manages conversation state + tool execution |
| `async_set_agent` | `homeassistant.components.conversation` | Register entity as conversation agent |
| `async_get_result_from_chat_log` | `homeassistant.components.conversation` | Extract ConversationResult from ChatLog |
| `ConverseError` | `homeassistant.components.conversation` | Error with `.as_conversation_result()` |
| `CONF_LLM_HASS_API` | `homeassistant.const` | Config key for LLM API selection |
| `ConfigSubentryFlow` | `homeassistant.config_entries` | Base class for subentry flows |
| `llm.Tool` | `homeassistant.helpers.llm` | Tool definition (name, description, parameters) |
| `llm.ToolInput` | `homeassistant.helpers.llm` | Tool call (tool_name, tool_args, id, external) |

## Functional Testing (HA Core Dev Container)
Run the integration inside a real Home Assistant instance from source with full debugger support.

### Setup
1. In VS Code: **Reopen in Container** → select **"HA Core Functional Test"**
2. First-time setup runs automatically (~5 min) — clones HA core, installs deps, symlinks the integration
3. Press **F5** to launch HA with debugger, or run `hass -c /workspace/ha-config`
4. Open **http://localhost:8123** → complete onboarding → add the integration

### Services
| Service | URL (from container) | URL (from host) |
|---------|---------------------|-----------------|
| Home Assistant | — | http://localhost:8123 |
| Ollama API | http://ollama:11434 | http://localhost:11434 |

When configuring the integration, use `http://ollama:11434/v1` as the API URL.

### Key paths inside the container
- HA Core source: `/workspace/ha-core`
- HA config dir: `/workspace/ha-config`
- Integration symlink: `/workspace/ha-config/custom_components/llm_home_controller` → your working copy
- Python venv: `/home/vscode/.local/ha-venv`

### Day-to-day workflow
- Code changes take effect after restarting HA (no container rebuild)
- `hass -c /workspace/ha-config --skip-pip` for faster restarts
- Update HA core: `cd /workspace/ha-core && git pull && uv pip install -e .`
- Reset config: `rm -rf /workspace/ha-config && bash scripts/setup-ha-core.sh`
- Pull an Ollama model from host: `docker exec <ollama-container> ollama pull llama3.2`

### VS Code debug configurations (`.vscode/launch.json`)
- **Home Assistant** — full launch with debugger (`justMyCode: false` to step into HA core)
- **Home Assistant (skip pip)** — faster restarts, skips dependency checks
- **Debug Current Test File** — run/debug the currently open test file

## Task Tracking
All tasks tracked in `.tasks/` directory — see `.tasks/progress.md` for current status.

## Testing
- pytest + pytest-homeassistant-custom-component + pytest-asyncio
- Mock `aiohttp.ClientSession` for API calls
- Mock `ChatLog` via `MockChatLog` (from HA core test pattern) for conversation tests
- Test all error paths: connection refused, auth failure, timeout, malformed response
- Run: `uv run pytest tests/ -v`
- Lint: `uv run ruff check .`
