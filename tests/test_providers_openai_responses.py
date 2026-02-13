"""Tests for the OpenAI Responses API provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
import voluptuous as vol
from homeassistant.components.conversation import (
    AssistantContent,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.helpers import llm

from custom_components.llm_home_controller.providers.openai_responses import (
    OpenAIResponsesProvider,
)


def _make_tool(name: str, description: str, parameters: vol.Schema | dict) -> llm.Tool:
    """Create a mock llm.Tool."""
    tool = MagicMock(spec=llm.Tool)
    tool.name = name
    tool.description = description
    tool.parameters = parameters if isinstance(parameters, vol.Schema) else vol.Schema(parameters)
    return tool


@pytest.fixture
def provider() -> OpenAIResponsesProvider:
    """Return a Responses API provider instance."""
    return OpenAIResponsesProvider()


# --- format_tools ---


def test_format_tools_basic(provider: OpenAIResponsesProvider) -> None:
    """Test flat tool format (no function wrapper)."""
    tool = _make_tool(
        name="turn_on_light",
        description="Turn on a light",
        parameters=vol.Schema({vol.Required("entity_id"): str}),
    )
    result = provider.format_tools([tool])
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["name"] == "turn_on_light"
    assert result[0]["description"] == "Turn on a light"
    assert "parameters" in result[0]
    # No wrapping "function" key like Chat Completions
    assert "function" not in result[0]


def test_format_tools_no_description(provider: OpenAIResponsesProvider) -> None:
    """Test tool with None description gets empty string."""
    tool = _make_tool("test", None, vol.Schema({}))
    result = provider.format_tools([tool])
    assert result[0]["description"] == ""


# --- convert_content ---


def test_convert_content_system_extracted(provider: OpenAIResponsesProvider) -> None:
    """Test system content is extracted to 'system' key."""
    content = [
        SystemContent(content="You are helpful"),
        UserContent(content="Hello"),
    ]
    result = provider.convert_content(content)
    assert result["system"] == "You are helpful"
    assert len(result["messages"]) == 1
    assert result["messages"][0] == {"role": "user", "content": "Hello"}


def test_convert_content_user_message(provider: OpenAIResponsesProvider) -> None:
    """Test simple user message."""
    content = [UserContent(content="Hi there")]
    result = provider.convert_content(content)
    assert result["messages"] == [{"role": "user", "content": "Hi there"}]
    assert result["system"] is None


def test_convert_content_assistant_text(provider: OpenAIResponsesProvider) -> None:
    """Test assistant text becomes role: assistant message."""
    content = [
        AssistantContent(
            agent_id="test",
            content="Hello!",
        )
    ]
    result = provider.convert_content(content)
    assert result["messages"] == [{"role": "assistant", "content": "Hello!"}]


def test_convert_content_assistant_tool_calls(provider: OpenAIResponsesProvider) -> None:
    """Test assistant tool calls become function_call items."""
    tc = llm.ToolInput(id="call_1", tool_name="turn_on", tool_args={"name": "light"})
    content = [
        AssistantContent(
            agent_id="test",
            content=None,
            tool_calls=[tc],
        )
    ]
    result = provider.convert_content(content)
    assert len(result["messages"]) == 1
    assert result["messages"][0] == {
        "type": "function_call",
        "id": "call_1",
        "name": "turn_on",
        "arguments": '{"name": "light"}',
    }


def test_convert_content_tool_results(provider: OpenAIResponsesProvider) -> None:
    """Test tool results become function_call_output items."""
    content = [
        ToolResultContent(
            agent_id="test",
            tool_call_id="call_1",
            tool_name="turn_on",
            tool_result={"success": True},
        )
    ]
    result = provider.convert_content(content)
    assert result["messages"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": '{"success": true}',
        }
    ]


def test_convert_content_assistant_text_and_tool_calls(provider: OpenAIResponsesProvider) -> None:
    """Test assistant with both text and tool calls produces separate items."""
    tc = llm.ToolInput(id="call_1", tool_name="get_state", tool_args={})
    content = [
        AssistantContent(
            agent_id="test",
            content="Let me check.",
            tool_calls=[tc],
        )
    ]
    result = provider.convert_content(content)
    assert len(result["messages"]) == 2
    assert result["messages"][0] == {"role": "assistant", "content": "Let me check."}
    assert result["messages"][1] == {
        "type": "function_call",
        "id": "call_1",
        "name": "get_state",
        "arguments": "{}",
    }


def test_convert_content_full_conversation(provider: OpenAIResponsesProvider) -> None:
    """Test full multi-turn conversation."""
    tc = llm.ToolInput(id="call_1", tool_name="turn_on", tool_args={"name": "light"})
    content = [
        SystemContent(content="You are helpful"),
        UserContent(content="Turn on the light"),
        AssistantContent(agent_id="test", content=None, tool_calls=[tc]),
        ToolResultContent(
            agent_id="test",
            tool_call_id="call_1",
            tool_name="turn_on",
            tool_result={"success": True},
        ),
        AssistantContent(agent_id="test", content="Done!"),
    ]
    result = provider.convert_content(content)
    assert result["system"] == "You are helpful"
    assert len(result["messages"]) == 4
    assert result["messages"][0] == {"role": "user", "content": "Turn on the light"}
    assert result["messages"][1]["type"] == "function_call"
    assert result["messages"][2]["type"] == "function_call_output"
    assert result["messages"][3] == {"role": "assistant", "content": "Done!"}


# --- build_payload ---


def test_build_payload_basic(provider: OpenAIResponsesProvider) -> None:
    """Test payload uses 'input', 'max_output_tokens', no 'instructions' when system is None."""
    payload = provider.build_payload(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )
    assert payload["model"] == "gpt-4o"
    assert payload["input"] == [{"role": "user", "content": "Hi"}]
    assert payload["max_output_tokens"] == 1024
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["stream"] is True
    assert "instructions" not in payload
    assert "messages" not in payload
    assert "max_tokens" not in payload


def test_build_payload_with_system(provider: OpenAIResponsesProvider) -> None:
    """Test payload includes 'instructions' when system is provided."""
    payload = provider.build_payload(
        model="gpt-4o",
        messages=[],
        system="You are helpful",
        tools=[],
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
    )
    assert payload["instructions"] == "You are helpful"


def test_build_payload_with_tools(provider: OpenAIResponsesProvider) -> None:
    """Test payload includes tools."""
    tools = [{"type": "function", "name": "test", "parameters": {}}]
    payload = provider.build_payload(
        model="gpt-4o",
        messages=[],
        system=None,
        tools=tools,
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
    )
    assert payload["tools"] == tools


# --- build_headers ---


def test_build_headers_with_key(provider: OpenAIResponsesProvider) -> None:
    """Test headers include Authorization when API key provided."""
    headers = provider.build_headers("sk-test")
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["Content-Type"] == "application/json"


def test_build_headers_without_key(provider: OpenAIResponsesProvider) -> None:
    """Test headers omit Authorization when no API key."""
    headers = provider.build_headers(None)
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


# --- build_url ---


def test_build_url(provider: OpenAIResponsesProvider) -> None:
    """Test URL ends with /responses."""
    assert provider.build_url("http://api.example.com/v1") == "http://api.example.com/v1/responses"


def test_build_url_strips_trailing_slash(provider: OpenAIResponsesProvider) -> None:
    """Test trailing slash is stripped."""
    assert provider.build_url("http://api.example.com/v1/") == "http://api.example.com/v1/responses"


# --- transform_stream ---


class MockStreamResponse:
    """Mock aiohttp response that yields SSE lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    @property
    def content(self) -> AsyncIterator[bytes]:
        return _async_iter(self._lines)


async def _async_iter(items: list[bytes]) -> AsyncIterator[bytes]:
    for item in items:
        yield item


def _responses_event(event_type: str, data: dict[str, Any]) -> list[bytes]:
    """Create SSE event: + data: lines for a Responses API event."""
    return [
        f"event: {event_type}\n".encode(),
        f"data: {json.dumps(data)}\n".encode(),
        b"\n",
    ]


@pytest.mark.asyncio
async def test_transform_stream_text(provider: OpenAIResponsesProvider) -> None:
    """Test text streaming."""
    lines = [
        *_responses_event("response.created", {"type": "response.created", "response": {}}),
        *_responses_event(
            "response.output_item.added",
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "role": "assistant"}},
        ),
        *_responses_event(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hello"},
        ),
        *_responses_event(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": " world"},
        ),
        *_responses_event(
            "response.completed",
            {"type": "response.completed", "response": {"usage": {"input_tokens": 10, "output_tokens": 5}}},
        ),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert deltas[0] == {"role": "assistant"}
    content_deltas = [d for d in deltas if "content" in d]
    assert content_deltas == [{"content": "Hello"}, {"content": " world"}]


@pytest.mark.asyncio
async def test_transform_stream_tool_call(provider: OpenAIResponsesProvider) -> None:
    """Test function call streaming."""
    lines = [
        *_responses_event("response.created", {"type": "response.created", "response": {}}),
        *_responses_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "id": "call_1", "name": "turn_on"},
            },
        ),
        *_responses_event(
            "response.function_call_arguments.delta",
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '{"entity'},
        ),
        *_responses_event(
            "response.function_call_arguments.delta",
            {"type": "response.function_call_arguments.delta", "output_index": 0, "delta": '_id": "light.1"}'},
        ),
        *_responses_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "turn_on",
                    "arguments": '{"entity_id": "light.1"}',
                },
            },
        ),
        *_responses_event(
            "response.completed",
            {"type": "response.completed", "response": {}},
        ),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    tool_delta = next(d for d in deltas if "tool_calls" in d)
    assert len(tool_delta["tool_calls"]) == 1
    tc = tool_delta["tool_calls"][0]
    assert tc.id == "call_1"
    assert tc.tool_name == "turn_on"
    assert tc.tool_args == {"entity_id": "light.1"}


@pytest.mark.asyncio
async def test_transform_stream_text_then_tool(provider: OpenAIResponsesProvider) -> None:
    """Test mixed text output followed by a function call."""
    lines = [
        *_responses_event("response.created", {"type": "response.created", "response": {}}),
        *_responses_event(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Checking..."},
        ),
        *_responses_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {"type": "function_call", "id": "call_2", "name": "get_state"},
            },
        ),
        *_responses_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "get_state",
                    "arguments": "{}",
                },
            },
        ),
        *_responses_event(
            "response.completed",
            {"type": "response.completed", "response": {}},
        ),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert {"content": "Checking..."} in deltas
    tool_delta = next(d for d in deltas if "tool_calls" in d)
    assert tool_delta["tool_calls"][0].tool_name == "get_state"


@pytest.mark.asyncio
async def test_transform_stream_multiple_tool_calls(provider: OpenAIResponsesProvider) -> None:
    """Test two function calls in one response."""
    lines = [
        *_responses_event("response.created", {"type": "response.created", "response": {}}),
        *_responses_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "function_call", "id": "c1", "name": "turn_on"},
            },
        ),
        *_responses_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "turn_on",
                    "arguments": '{"name": "light1"}',
                },
            },
        ),
        *_responses_event(
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {"type": "function_call", "id": "c2", "name": "turn_on"},
            },
        ),
        *_responses_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "turn_on",
                    "arguments": '{"name": "light2"}',
                },
            },
        ),
        *_responses_event("response.completed", {"type": "response.completed", "response": {}}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    tool_deltas = [d for d in deltas if "tool_calls" in d]
    assert len(tool_deltas) == 2
    assert tool_deltas[0]["tool_calls"][0].tool_args == {"name": "light1"}
    assert tool_deltas[1]["tool_calls"][0].tool_args == {"name": "light2"}


@pytest.mark.asyncio
async def test_transform_stream_usage(provider: OpenAIResponsesProvider) -> None:
    """Test usage info from response.completed."""
    lines = [
        *_responses_event("response.created", {"type": "response.created", "response": {}}),
        *_responses_event(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "Hi"},
        ),
        *_responses_event(
            "response.completed",
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 50, "output_tokens": 10}},
            },
        ),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    usage_delta = next(d for d in deltas if "native" in d)
    assert usage_delta["native"]["usage"]["input_tokens"] == 50
    assert usage_delta["native"]["usage"]["output_tokens"] == 10


@pytest.mark.asyncio
async def test_transform_stream_malformed_json(provider: OpenAIResponsesProvider) -> None:
    """Test malformed JSON is skipped."""
    lines = [
        b"event: response.created\n",
        b"data: {not json}\n",
        b"\n",
        *_responses_event(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "OK"},
        ),
        *_responses_event("response.completed", {"type": "response.completed", "response": {}}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert {"content": "OK"} in deltas


@pytest.mark.asyncio
async def test_transform_stream_incomplete(provider: OpenAIResponsesProvider) -> None:
    """Test response.incomplete yields finish_reason."""
    lines = [
        *_responses_event("response.created", {"type": "response.created", "response": {}}),
        *_responses_event("response.incomplete", {"type": "response.incomplete"}),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert {"native": {"finish_reason": "incomplete"}} in deltas


# --- get_models ---


@pytest.mark.asyncio
async def test_get_models_success(provider: OpenAIResponsesProvider) -> None:
    """Test fetching models returns sorted IDs."""
    from unittest.mock import AsyncMock

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"data": [{"id": "gpt-4o"}, {"id": "gpt-3.5"}]})
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.get = MagicMock(return_value=mock_resp)

    models = await provider.get_models(session, "http://api.example.com/v1", "sk-test")
    assert models == ["gpt-3.5", "gpt-4o"]


# --- Vision / attachment tests ---


def test_convert_content_user_with_image(provider: OpenAIResponsesProvider) -> None:
    """Test user message with image attachment."""
    from pathlib import Path

    att = MagicMock()
    att.mime_type = "image/png"
    att.path = MagicMock(spec=Path)
    att.path.read_bytes.return_value = b"fake-image-data"

    content = [
        UserContent(content="What's this?", attachments=[att]),
    ]
    result = provider.convert_content(content)
    msg = result["messages"][0]
    assert msg["role"] == "user"
    assert msg["content"][0] == {"type": "input_text", "text": "What's this?"}
    assert msg["content"][1]["type"] == "input_image"
    assert msg["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_convert_content_attachment_missing_file(provider: OpenAIResponsesProvider) -> None:
    """Test that missing attachment file is skipped gracefully."""
    from pathlib import Path

    att = MagicMock()
    att.mime_type = "image/png"
    att.path = MagicMock(spec=Path)
    att.path.read_bytes.side_effect = FileNotFoundError("No such file")

    content = [
        UserContent(content="See this?", attachments=[att]),
    ]
    result = provider.convert_content(content)
    msg = result["messages"][0]
    # Should still have the text part but no image part
    assert len(msg["content"]) == 1
    assert msg["content"][0]["type"] == "input_text"


# --- Thinking / reasoning support tests ---


def test_convert_content_thinking(provider: OpenAIResponsesProvider) -> None:
    """Test that thinking_content generates a reasoning item."""
    content = [
        AssistantContent(
            agent_id="test",
            content="The answer is 42.",
            thinking_content="Let me reason about this...",
        )
    ]
    result = provider.convert_content(content)
    items = result["messages"]
    assert len(items) == 2
    assert items[0] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Let me reason about this..."}],
    }
    assert items[1] == {"role": "assistant", "content": "The answer is 42."}


def test_build_payload_with_stop_sequences(provider: OpenAIResponsesProvider) -> None:
    """Test that stop sequences are included in payload."""
    payload = provider.build_payload(
        model="gpt-4o",
        messages=[],
        system=None,
        tools=[],
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
        extra_options={"stop_sequences": ["STOP", "END"]},
    )
    assert payload["stop"] == ["STOP", "END"]


def test_build_payload_with_response_format(provider: OpenAIResponsesProvider) -> None:
    """Test that response format is included in payload."""
    payload = provider.build_payload(
        model="gpt-4o",
        messages=[],
        system=None,
        tools=[],
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
        extra_options={"response_format": "json_object"},
    )
    assert payload["text"] == {"format": {"type": "json_object"}}


def test_build_payload_with_json_schema(provider: OpenAIResponsesProvider) -> None:
    """Test that JSON schema response format includes the schema."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    payload = provider.build_payload(
        model="gpt-4o",
        messages=[],
        system=None,
        tools=[],
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
        extra_options={"response_format": "json_schema", "json_schema": schema},
    )
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["schema"] == schema


def test_build_payload_with_extended_thinking(provider: OpenAIResponsesProvider) -> None:
    """Test that extended thinking adds reasoning config."""
    payload = provider.build_payload(
        model="gpt-4o",
        messages=[],
        system=None,
        tools=[],
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
        extra_options={"extended_thinking": True},
    )
    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}


@pytest.mark.asyncio
async def test_transform_stream_reasoning_delta(provider: OpenAIResponsesProvider) -> None:
    """Test reasoning summary text delta is mapped to thinking_content."""
    lines = [
        *_responses_event(
            "response.created",
            {"type": "response.created", "response": {}},
        ),
        *_responses_event(
            "response.reasoning_summary_text.delta",
            {"type": "response.reasoning_summary_text.delta", "delta": "Thinking..."},
        ),
        *_responses_event(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "Answer"},
        ),
        *_responses_event(
            "response.completed",
            {"type": "response.completed", "response": {}},
        ),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    thinking_deltas = [d for d in deltas if "thinking_content" in d]
    assert thinking_deltas == [{"thinking_content": "Thinking..."}]
    content_deltas = [d for d in deltas if "content" in d]
    assert content_deltas == [{"content": "Answer"}]
