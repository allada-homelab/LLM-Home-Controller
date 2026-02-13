# LLM Home Controller

A Home Assistant custom integration that provides a **Conversation Agent** for Assist, connecting to any OpenAI API-compatible endpoint for local or remote LLM inference.

Control your smart home with natural language through your own LLM — no cloud required.

## Features

- **OpenAI API Compatible** — Works with any endpoint that implements the `/v1/chat/completions` API
- **Streaming Responses** — Real-time token-by-token output via Server-Sent Events (SSE)
- **Tool Calling / Function Calling** — Full support for HA tool execution with multi-turn loops (up to 10 iterations)
- **Multiple Agents** — Configure multiple conversation agents per API connection, each with different models and settings
- **Home Assistant Assist** — Registers as a native Conversation Agent for use in Assist pipelines
- **Device Control** — Optionally enable LLM-based control of Home Assistant entities via the Assist API

## Supported Backends

Any server implementing the OpenAI `/v1/chat/completions` streaming API:

| Backend | Notes |
|---------|-------|
| [llama-swap](https://github.com/mostlygeek/llama-swap) | Model hot-swapping proxy for llama.cpp |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | C++ inference server with OpenAI-compatible API |
| [Ollama](https://ollama.ai/) | Run LLMs locally with OpenAI compatibility mode |
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput inference engine |
| [LiteLLM](https://github.com/BerriAI/litellm) | Proxy for 100+ LLM APIs with unified interface |
| [LocalAI](https://localai.io/) | Drop-in OpenAI replacement |
| [text-generation-webui](https://github.com/oobabooga/text-generation-webui) | With OpenAI extension enabled |
| [OpenAI](https://platform.openai.com/) | Official OpenAI API |
| Any OpenAI-compatible API | If it serves `/v1/chat/completions`, it works |

## Installation

### HACS (Recommended)

1. Open HACS in your Home Assistant instance
2. Click the three dots menu in the top right corner
3. Select **Custom repositories**
4. Add `https://github.com/allada-homelab/LLM-Home-Controller` as an **Integration**
5. Search for "LLM Home Controller" and install it
6. Restart Home Assistant

### Manual Installation

1. Download the `custom_components/llm_home_controller` directory from this repository
2. Copy it to your Home Assistant `config/custom_components/` directory
3. Restart Home Assistant

## Configuration

### Step 1: Add the Integration

1. Go to **Settings** > **Devices & Services** > **Add Integration**
2. Search for **LLM Home Controller**
3. Enter your API endpoint URL (e.g., `http://192.168.1.100:8080/v1`)
4. Optionally enter an API key if your endpoint requires authentication
5. The integration validates the connection by fetching available models

### Step 2: Configure a Conversation Agent

A default conversation agent is created automatically when you add the integration. To add more or modify settings:

1. Go to **Settings** > **Devices & Services** > **LLM Home Controller**
2. Click **Add Conversation Agent** to create a new agent, or click an existing agent to reconfigure
3. Configure:
   - **Model** — Select from available models (dropdown) or enter a model name manually
   - **System Prompt** — Customize the assistant's behavior and personality
   - **Control Home Assistant** — Enable to allow the LLM to control your smart home devices
   - **Temperature** — Controls randomness (0.0 = deterministic, 2.0 = very creative)
   - **Max Tokens** — Maximum response length
   - **Top P** — Nucleus sampling threshold

### Step 3: Use with Assist

1. Go to **Settings** > **Voice Assistants**
2. Create or edit a voice assistant pipeline
3. Select your LLM Home Controller agent as the **Conversation Agent**
4. Use Assist from the sidebar, a dashboard card, or voice devices

## Architecture

This integration uses the **config subentry** pattern:

- **Parent entry** — Holds the API connection (URL + key). One per endpoint.
- **Conversation subentries** — Each represents an independent conversation agent with its own model, prompt, and settings. Multiple agents can share the same API connection.

```
API Connection (http://192.168.1.100:8080/v1)
  +-- Agent: "General Assistant" (model: llama-3.1-8b, temp: 0.7)
  +-- Agent: "Code Helper" (model: deepseek-coder-v2, temp: 0.2)
  +-- Agent: "Smart Home" (model: llama-3.1-8b, temp: 0.3, HA control: enabled)
```

## Requirements

- Home Assistant 2025.1.0 or later
- Python 3.13+
- An accessible OpenAI API-compatible endpoint

## Development

```bash
# Clone the repository
git clone https://github.com/allada-homelab/LLM-Home-Controller.git
cd LLM-Home-Controller

# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check .
uv run ruff format --check .
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
