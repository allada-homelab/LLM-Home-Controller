"""Tests for the Anthropic Messages API provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import voluptuous as vol
from homeassistant.components.conversation import (
    AssistantContent,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.helpers import llm

from custom_components.llm_home_controller.providers.anthropic import AnthropicProvider


def _make_tool(name: str, description: str, parameters: vol.Schema | dict) -> llm.Tool:
    """Create a mock llm.Tool."""
    tool = MagicMock(spec=llm.Tool)
    tool.name = name
    tool.description = description
    tool.parameters = parameters if isinstance(parameters, vol.Schema) else vol.Schema(parameters)
    return tool


@pytest.fixture
def provider() -> AnthropicProvider:
    """Return an Anthropic provider instance."""
    return AnthropicProvider()


# --- format_tools ---


def test_format_tools_basic(provider: AnthropicProvider) -> None:
    """Test formatting tools uses input_schema (not parameters)."""
    tool = _make_tool(
        name="turn_on_light",
        description="Turn on a light",
        parameters=vol.Schema({vol.Required("entity_id"): str}),
    )
    result = provider.format_tools([tool])
    assert len(result) == 1
    assert result[0]["name"] == "turn_on_light"
    assert result[0]["description"] == "Turn on a light"
    schema = result[0]["input_schema"]
    assert schema["type"] == "object"
    assert "entity_id" in schema["properties"]
    assert schema["properties"]["entity_id"]["type"] == "string"


def test_format_tools_no_description(provider: AnthropicProvider) -> None:
    """Test formatting a tool with no description."""
    tool = _make_tool(name="test", description=None, parameters={})
    result = provider.format_tools([tool])
    assert result[0]["description"] == ""


# --- convert_content ---


def test_convert_content_system_separate(provider: AnthropicProvider) -> None:
    """Test that system content is extracted to 'system' key, not in messages."""
    content = [
        SystemContent(content="Be helpful"),
        UserContent(content="Hello"),
    ]
    converted = provider.convert_content(content)
    assert converted["system"] == "Be helpful"
    assert len(converted["messages"]) == 1
    assert converted["messages"][0]["role"] == "user"


def test_convert_content_alternating_roles(provider: AnthropicProvider) -> None:
    """Test that consecutive user messages are merged (Anthropic requires alternation)."""
    content = [
        UserContent(content="Hello"),
        UserContent(content="How are you?"),
    ]
    converted = provider.convert_content(content)
    messages = converted["messages"]
    # Both should be merged into one user message
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert len(messages[0]["content"]) == 2
    assert messages[0]["content"][0] == {"type": "text", "text": "Hello"}
    assert messages[0]["content"][1] == {"type": "text", "text": "How are you?"}


def test_convert_content_tool_results_in_user_message(provider: AnthropicProvider) -> None:
    """Test that tool results are placed inside user messages as tool_result blocks."""
    content = [
        UserContent(content="Turn on lights"),
        AssistantContent(
            agent_id="test",
            content=None,
            tool_calls=[
                llm.ToolInput(id="call_1", tool_name="light_on", tool_args={"id": "1"}),
            ],
        ),
        ToolResultContent(
            agent_id="test",
            tool_call_id="call_1",
            tool_name="light_on",
            tool_result={"success": True},
        ),
    ]
    converted = provider.convert_content(content)
    messages = converted["messages"]
    assert len(messages) == 3
    # Tool result should be in a user message
    tool_msg = messages[2]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == "call_1"


def test_convert_content_assistant_tool_use_blocks(provider: AnthropicProvider) -> None:
    """Test that tool calls are rendered as tool_use content blocks."""
    content = [
        AssistantContent(
            agent_id="test",
            content="Let me help.",
            tool_calls=[
                llm.ToolInput(id="call_1", tool_name="turn_on", tool_args={"entity_id": "light.1"}),
            ],
        ),
    ]
    converted = provider.convert_content(content)
    messages = converted["messages"]
    assert len(messages) == 1
    blocks = messages[0]["content"]
    assert blocks[0] == {"type": "text", "text": "Let me help."}
    assert blocks[1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "turn_on",
        "input": {"entity_id": "light.1"},
    }


def test_convert_content_assistant_thinking(provider: AnthropicProvider) -> None:
    """Test that thinking content is preserved in assistant blocks."""
    content = [
        AssistantContent(
            agent_id="test",
            content="I'll help.",
            thinking_content="Let me think about this...",
        ),
    ]
    converted = provider.convert_content(content)
    messages = converted["messages"]
    blocks = messages[0]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "Let me think about this..."}
    assert blocks[1] == {"type": "text", "text": "I'll help."}


def test_convert_content_full_conversation(provider: AnthropicProvider) -> None:
    """Test a complete multi-turn conversation with tools."""
    content = [
        SystemContent(content="Be helpful"),
        UserContent(content="Turn on the lights"),
        AssistantContent(
            agent_id="test",
            content=None,
            tool_calls=[
                llm.ToolInput(id="call_1", tool_name="turn_on", tool_args={"id": "1"}),
            ],
        ),
        ToolResultContent(
            agent_id="test",
            tool_call_id="call_1",
            tool_name="turn_on",
            tool_result={"success": True},
        ),
        AssistantContent(
            agent_id="test",
            content="Done! The light is on.",
        ),
    ]
    converted = provider.convert_content(content)
    assert converted["system"] == "Be helpful"
    messages = converted["messages"]
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"  # tool result
    assert messages[3]["role"] == "assistant"


# --- build_payload ---


def test_build_payload_basic(provider: AnthropicProvider) -> None:
    """Test basic payload construction."""
    payload = provider.build_payload(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )
    assert payload["model"] == "claude-3-5-sonnet"
    assert payload["max_tokens"] == 1024
    assert payload["stream"] is True
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert "system" not in payload
    assert "tools" not in payload


def test_build_payload_with_system(provider: AnthropicProvider) -> None:
    """Test that system prompt is included in payload."""
    payload = provider.build_payload(
        model="claude-3-5-sonnet",
        messages=[],
        system="Be helpful",
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )
    assert payload["system"] == "Be helpful"


def test_build_payload_with_tools(provider: AnthropicProvider) -> None:
    """Test that tools are included in payload."""
    tools = [{"name": "test", "description": "test", "input_schema": {}}]
    payload = provider.build_payload(
        model="claude-3-5-sonnet",
        messages=[],
        system=None,
        tools=tools,
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )
    assert payload["tools"] == tools


def test_build_payload_extended_thinking(provider: AnthropicProvider) -> None:
    """Test extended thinking payload — no temperature/top_p, adds thinking config."""
    payload = provider.build_payload(
        model="claude-3-5-sonnet",
        messages=[],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=16000,
        top_p=0.9,
        extra_options={"extended_thinking": True, "thinking_budget": 8000},
    )
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 8000}


# --- build_headers ---


def test_build_headers_with_key(provider: AnthropicProvider) -> None:
    """Test headers include x-api-key and anthropic-version."""
    headers = provider.build_headers("sk-ant-test-key")
    assert headers["x-api-key"] == "sk-ant-test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["Content-Type"] == "application/json"


def test_build_headers_without_key(provider: AnthropicProvider) -> None:
    """Test headers without API key."""
    headers = provider.build_headers(None)
    assert "x-api-key" not in headers
    assert "anthropic-version" in headers


# --- build_url ---


def test_build_url(provider: AnthropicProvider) -> None:
    """Test URL ends with /messages."""
    url = provider.build_url("https://api.anthropic.com/v1")
    assert url == "https://api.anthropic.com/v1/messages"


def test_build_url_strips_trailing_slash(provider: AnthropicProvider) -> None:
    """Test trailing slash is stripped."""
    url = provider.build_url("https://api.anthropic.com/v1/")
    assert url == "https://api.anthropic.com/v1/messages"


# --- transform_stream ---


class MockStreamResponse:
    """Mock aiohttp response that yields SSE lines (Anthropic format)."""

    def __init__(self, lines: list[str]) -> None:
        """Initialize with SSE lines."""
        self._lines = lines

    @property
    def content(self) -> AsyncIterator[bytes]:
        """Return an async iterator of encoded lines."""
        return self._iter_lines()

    async def _iter_lines(self) -> AsyncIterator[bytes]:
        for line in self._lines:
            yield (line + "\n").encode("utf-8")


def _anthropic_event(event_type: str, data: dict[str, Any]) -> list[str]:
    """Create Anthropic SSE event + data lines."""
    return [f"event: {event_type}", f"data: {json.dumps(data)}"]


@pytest.mark.asyncio
async def test_transform_stream_text(provider: AnthropicProvider) -> None:
    """Test transforming text_delta events."""
    lines = [
        *_anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
            },
        ),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world"},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *_anthropic_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
        ),
        *_anthropic_event("message_stop", {"type": "message_stop"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert deltas[0] == {"role": "assistant"}
    content_deltas = [d for d in deltas if "content" in d]
    assert content_deltas == [{"content": "Hello"}, {"content": " world"}]


@pytest.mark.asyncio
async def test_transform_stream_tool_use(provider: AnthropicProvider) -> None:
    """Test transforming tool_use block with input_json_delta."""
    lines = [
        *_anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
            },
        ),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "turn_on_light"},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"entity'},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '_id": "light.1"}'},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *_anthropic_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
            },
        ),
        *_anthropic_event("message_stop", {"type": "message_stop"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    tool_delta = next(d for d in deltas if "tool_calls" in d)
    assert len(tool_delta["tool_calls"]) == 1
    tc = tool_delta["tool_calls"][0]
    assert tc.id == "toolu_1"
    assert tc.tool_name == "turn_on_light"
    assert tc.tool_args == {"entity_id": "light.1"}


@pytest.mark.asyncio
async def test_transform_stream_thinking(provider: AnthropicProvider) -> None:
    """Test transforming thinking_delta blocks."""
    lines = [
        *_anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
            },
        ),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Here's my answer."},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 1}),
        *_anthropic_event("message_stop", {"type": "message_stop"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    thinking_deltas = [d for d in deltas if "thinking_content" in d]
    assert thinking_deltas == [{"thinking_content": "Let me think..."}]
    content_deltas = [d for d in deltas if "content" in d]
    assert content_deltas == [{"content": "Here's my answer."}]


@pytest.mark.asyncio
async def test_transform_stream_mixed_text_and_tools(provider: AnthropicProvider) -> None:
    """Test text content followed by a tool call in the same message."""
    lines = [
        *_anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
            },
        ),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "I'll turn that on."},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "light_on"},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"id": "1"}'},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 1}),
        *_anthropic_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
            },
        ),
        *_anthropic_event("message_stop", {"type": "message_stop"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert {"content": "I'll turn that on."} in deltas
    tool_delta = next(d for d in deltas if "tool_calls" in d)
    assert tool_delta["tool_calls"][0].tool_name == "light_on"


@pytest.mark.asyncio
async def test_transform_stream_usage_tracking(provider: AnthropicProvider) -> None:
    """Test that input and output usage are surfaced via native."""
    lines = [
        *_anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 25},
                },
            },
        ),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *_anthropic_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 12},
            },
        ),
        *_anthropic_event("message_stop", {"type": "message_stop"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    native_deltas = [d for d in deltas if "native" in d]
    usage_deltas = [d for d in native_deltas if "usage" in d.get("native", {})]
    assert len(usage_deltas) == 2
    assert usage_deltas[0]["native"]["usage"]["input_tokens"] == 25
    assert usage_deltas[1]["native"]["usage"]["output_tokens"] == 12


@pytest.mark.asyncio
async def test_transform_stream_finish_reason_max_tokens(provider: AnthropicProvider) -> None:
    """Test that max_tokens stop reason is surfaced via native."""
    lines = [
        *_anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
            },
        ),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Truncated..."},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *_anthropic_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens"},
            },
        ),
        *_anthropic_event("message_stop", {"type": "message_stop"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    native_deltas = [d for d in deltas if "native" in d]
    finish_delta = next(d for d in native_deltas if "finish_reason" in d.get("native", {}))
    assert finish_delta["native"]["finish_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_transform_stream_ping_ignored(provider: AnthropicProvider) -> None:
    """Test that ping events are silently ignored."""
    lines = [
        *_anthropic_event("ping", {"type": "ping"}),
        *_anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
            },
        ),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "OK"},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *_anthropic_event("message_stop", {"type": "message_stop"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert {"content": "OK"} in deltas


@pytest.mark.asyncio
async def test_transform_stream_signature_delta_ignored(provider: AnthropicProvider) -> None:
    """Test that signature_delta events are silently ignored."""
    lines = [
        *_anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
            },
        ),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "abc123"},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *_anthropic_event("message_stop", {"type": "message_stop"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    content_deltas = [d for d in deltas if "content" in d]
    assert content_deltas == [{"content": "Hello"}]


@pytest.mark.asyncio
async def test_transform_stream_malformed_json_skipped(provider: AnthropicProvider) -> None:
    """Test that malformed JSON is skipped gracefully."""
    lines = [
        "event: message_start",
        "data: {invalid json}",
        *_anthropic_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_1", "type": "message", "role": "assistant", "content": []},
            },
        ),
        *_anthropic_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        *_anthropic_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "recovered"},
            },
        ),
        *_anthropic_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        *_anthropic_event("message_stop", {"type": "message_stop"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert {"content": "recovered"} in deltas


# --- get_models ---


@pytest.mark.asyncio
async def test_get_models_success(provider: AnthropicProvider) -> None:
    """Test fetching models successfully."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"data": [{"id": "claude-3-sonnet"}, {"id": "claude-3-haiku"}]})
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.get = MagicMock(return_value=mock_response)

    models = await provider.get_models(session, "https://api.anthropic.com/v1", "sk-ant-key")
    assert models == ["claude-3-haiku", "claude-3-sonnet"]


@pytest.mark.asyncio
async def test_get_models_fallback_empty(provider: AnthropicProvider) -> None:
    """Test that get_models returns [] when endpoint is unavailable."""
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.get = MagicMock(return_value=mock_response)

    models = await provider.get_models(session, "https://api.anthropic.com/v1", "sk-ant-key")
    assert models == []


@pytest.mark.asyncio
async def test_get_models_fallback_on_error(provider: AnthropicProvider) -> None:
    """Test that get_models returns [] on connection error."""
    import aiohttp

    session = AsyncMock()
    session.get = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))

    models = await provider.get_models(session, "https://api.anthropic.com/v1", "sk-ant-key")
    assert models == []


# --- Feature 2: Vision attachment error handling ---


def test_convert_content_image_attachment(provider: AnthropicProvider) -> None:
    """Test user content with an image attachment."""
    from pathlib import Path

    att = MagicMock()
    att.mime_type = "image/jpeg"
    att.path = MagicMock(spec=Path)
    att.path.read_bytes.return_value = b"fake-jpeg"

    content = [
        UserContent(content="What's this?", attachments=[att]),
    ]
    converted = provider.convert_content(content)
    blocks = converted["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "What's this?"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["media_type"] == "image/jpeg"
    assert blocks[1]["source"]["type"] == "base64"


def test_convert_content_pdf_attachment(provider: AnthropicProvider) -> None:
    """Test user content with a PDF attachment."""
    from pathlib import Path

    att = MagicMock()
    att.mime_type = "application/pdf"
    att.path = MagicMock(spec=Path)
    att.path.read_bytes.return_value = b"fake-pdf"

    content = [
        UserContent(content="Summarize this", attachments=[att]),
    ]
    converted = provider.convert_content(content)
    blocks = converted["messages"][0]["content"]
    assert blocks[1]["type"] == "document"
    assert blocks[1]["source"]["media_type"] == "application/pdf"


def test_convert_content_missing_attachment_skipped(provider: AnthropicProvider) -> None:
    """Test that a missing attachment file is skipped gracefully."""
    from pathlib import Path

    att = MagicMock()
    att.mime_type = "image/png"
    att.path = MagicMock(spec=Path)
    att.path.read_bytes.side_effect = FileNotFoundError("No such file")

    content = [
        UserContent(content="Look at this", attachments=[att]),
    ]
    converted = provider.convert_content(content)
    blocks = converted["messages"][0]["content"]
    # Only the text block should be present, image skipped
    assert len(blocks) == 1
    assert blocks[0] == {"type": "text", "text": "Look at this"}


# --- Feature 4: Stop sequences ---


def test_build_payload_with_stop_sequences(provider: AnthropicProvider) -> None:
    """Test that stop sequences are included in Anthropic payload."""
    payload = provider.build_payload(
        model="claude-3-5-sonnet",
        messages=[],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        extra_options={"stop_sequences": ["STOP", "END"]},
    )
    assert payload["stop_sequences"] == ["STOP", "END"]


def test_build_payload_stop_sequences_not_set(provider: AnthropicProvider) -> None:
    """Test that stop_sequences is not included when not configured."""
    payload = provider.build_payload(
        model="claude-3-5-sonnet",
        messages=[],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )
    assert "stop_sequences" not in payload


def test_build_payload_json_schema_injected_into_system(provider: AnthropicProvider) -> None:
    """Test JSON schema is injected into system prompt for Anthropic."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    payload = provider.build_payload(
        model="claude-3-5-sonnet",
        messages=[],
        system="Be helpful",
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        extra_options={"response_format": "json_schema", "json_schema": schema},
    )
    assert "json_schema" not in payload  # Not a native Anthropic field
    assert "json" in payload["system"].lower()
    assert '"name"' in payload["system"]


def test_build_payload_json_schema_creates_system_if_none(provider: AnthropicProvider) -> None:
    """Test JSON schema creates system prompt if none existed."""
    schema = {"type": "object"}
    payload = provider.build_payload(
        model="claude-3-5-sonnet",
        messages=[],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        extra_options={"response_format": "json_schema", "json_schema": schema},
    )
    assert "system" in payload
    assert "json" in payload["system"].lower()
