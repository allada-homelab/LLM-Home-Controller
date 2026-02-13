"""Tests for LLM Home Controller entity module (OpenAI provider)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import voluptuous as vol
from homeassistant.components.conversation import (
    AssistantContent,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm

from custom_components.llm_home_controller.const import (
    CONF_FALLBACK_MODEL,
    CONF_MAX_CONTEXT_TOKENS,
    CONF_MAX_RETRIES,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_RESPONSE_FORMAT,
    CONF_SEED,
    CONF_STOP_SEQUENCES,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from custom_components.llm_home_controller.entity import (
    LLMHomeControllerBaseLLMEntity,
    _estimate_tokens,
    _prune_messages,
)
from custom_components.llm_home_controller.providers.openai import OpenAIProvider


def _make_tool(name: str, description: str, parameters: vol.Schema | dict) -> llm.Tool:
    """Create a mock llm.Tool."""
    tool = MagicMock(spec=llm.Tool)
    tool.name = name
    tool.description = description
    tool.parameters = parameters if isinstance(parameters, vol.Schema) else vol.Schema(parameters)
    return tool


@pytest.fixture
def provider() -> OpenAIProvider:
    """Return an OpenAI provider instance."""
    return OpenAIProvider()


def test_format_tool_basic(provider: OpenAIProvider) -> None:
    """Test formatting a basic tool."""
    tool = _make_tool(
        name="turn_on_light",
        description="Turn on a light",
        parameters=vol.Schema({vol.Required("entity_id"): str}),
    )
    result = provider.format_tools([tool])

    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "turn_on_light"
    assert result[0]["function"]["description"] == "Turn on a light"
    params = result[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert "entity_id" in params["properties"]
    assert params["properties"]["entity_id"]["type"] == "string"


def test_format_tool_no_description(provider: OpenAIProvider) -> None:
    """Test formatting a tool with no description."""
    tool = _make_tool(name="test", description=None, parameters={})
    result = provider.format_tools([tool])
    assert result[0]["function"]["description"] == ""


def test_convert_content_system(provider: OpenAIProvider) -> None:
    """Test converting system content."""
    content = [SystemContent(content="You are a helpful assistant.")]
    converted = provider.convert_content(content)
    assert converted["system"] is None
    assert converted["messages"] == [{"role": "system", "content": "You are a helpful assistant."}]


def test_convert_content_user(provider: OpenAIProvider) -> None:
    """Test converting user content."""
    content = [UserContent(content="Hello")]
    converted = provider.convert_content(content)
    assert converted["messages"] == [{"role": "user", "content": "Hello"}]


def test_convert_content_user_with_image(provider: OpenAIProvider) -> None:
    """Test converting user content with image attachment."""
    from pathlib import Path

    att = MagicMock()
    att.mime_type = "image/png"
    att.path = MagicMock(spec=Path)
    att.path.read_bytes.return_value = b"fake-image"

    content = [UserContent(content="What's this?", attachments=[att])]
    converted = provider.convert_content(content)
    msg = converted["messages"][0]
    assert msg["role"] == "user"
    assert msg["content"][0] == {"type": "text", "text": "What's this?"}
    assert msg["content"][1]["type"] == "image_url"
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_convert_content_missing_attachment_skipped(provider: OpenAIProvider) -> None:
    """Test that missing attachment file is skipped gracefully."""
    from pathlib import Path

    att = MagicMock()
    att.mime_type = "image/png"
    att.path = MagicMock(spec=Path)
    att.path.read_bytes.side_effect = FileNotFoundError("No such file")

    content = [UserContent(content="Look", attachments=[att])]
    converted = provider.convert_content(content)
    msg = converted["messages"][0]
    # Only text part, no image
    assert len(msg["content"]) == 1
    assert msg["content"][0]["type"] == "text"


def test_convert_content_assistant_text(provider: OpenAIProvider) -> None:
    """Test converting assistant text content."""
    content = [
        AssistantContent(
            agent_id="test",
            content="Hello there!",
        )
    ]
    converted = provider.convert_content(content)
    assert converted["messages"] == [{"role": "assistant", "content": "Hello there!"}]


def test_convert_content_assistant_tool_calls(provider: OpenAIProvider) -> None:
    """Test converting assistant content with tool calls."""
    content = [
        AssistantContent(
            agent_id="test",
            content=None,
            tool_calls=[
                llm.ToolInput(
                    id="call_123",
                    tool_name="turn_on_light",
                    tool_args={"entity_id": "light.living_room"},
                ),
            ],
        )
    ]
    converted = provider.convert_content(content)
    messages = converted["messages"]
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "assistant"
    assert "content" not in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_123"
    assert tc["function"]["name"] == "turn_on_light"
    assert json.loads(tc["function"]["arguments"]) == {"entity_id": "light.living_room"}


def test_convert_content_tool_result(provider: OpenAIProvider) -> None:
    """Test converting tool result content."""
    content = [
        ToolResultContent(
            agent_id="test",
            tool_call_id="call_123",
            tool_name="turn_on_light",
            tool_result={"success": True},
        )
    ]
    converted = provider.convert_content(content)
    assert converted["messages"] == [
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"success": true}',
        }
    ]


def test_convert_content_full_conversation(provider: OpenAIProvider) -> None:
    """Test converting a full conversation with all content types."""
    content = [
        SystemContent(content="Be helpful"),
        UserContent(content="Turn on the lights"),
        AssistantContent(
            agent_id="test",
            content=None,
            tool_calls=[
                llm.ToolInput(
                    id="call_1",
                    tool_name="turn_on",
                    tool_args={"entity_id": "light.living"},
                ),
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
            content="Done! The living room light is now on.",
        ),
    ]
    converted = provider.convert_content(content)
    messages = converted["messages"]
    assert len(messages) == 5
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["id"] == "call_1"
    assert messages[3]["role"] == "tool"
    assert messages[4]["role"] == "assistant"
    assert messages[4]["content"] == "Done! The living room light is now on."


class MockStreamResponse:
    """Mock aiohttp response that yields SSE lines."""

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

    def release(self) -> None:
        """Mock release."""

    def close(self) -> None:
        """Mock close."""


def _sse(data: dict[str, Any] | str) -> str:
    """Create an SSE data line."""
    if isinstance(data, str):
        return f"data: {data}"
    return f"data: {json.dumps(data)}"


@pytest.mark.asyncio
async def test_transform_stream_text_response(provider: OpenAIProvider) -> None:
    """Test transforming a simple text response."""
    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": " world"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert deltas[0] == {"role": "assistant"}
    assert deltas[1] == {"content": "Hello"}
    assert deltas[2] == {"content": " world"}


@pytest.mark.asyncio
async def test_transform_stream_tool_call(provider: OpenAIProvider) -> None:
    """Test transforming a response with a tool call."""
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "turn_on_light", "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"entity_id"'},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": ': "light.living"}'},
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        _sse("[DONE]"),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    # Should have: role, tool_calls
    assert deltas[0] == {"role": "assistant"}
    tool_delta = next(d for d in deltas if "tool_calls" in d)
    assert len(tool_delta["tool_calls"]) == 1
    tc = tool_delta["tool_calls"][0]
    assert tc.id == "call_abc"
    assert tc.tool_name == "turn_on_light"
    assert tc.tool_args == {"entity_id": "light.living"}


@pytest.mark.asyncio
async def test_transform_stream_parallel_tool_calls(provider: OpenAIProvider) -> None:
    """Test transforming a response with parallel tool calls."""
    lines = [
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "light_on", "arguments": ""},
                                },
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "light_off", "arguments": ""},
                                },
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"id": "a"}'}}]},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 1, "function": {"arguments": '{"id": "b"}'}}]},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        _sse("[DONE]"),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    tool_delta = next(d for d in deltas if "tool_calls" in d)
    assert len(tool_delta["tool_calls"]) == 2
    names = {tc.tool_name for tc in tool_delta["tool_calls"]}
    assert names == {"light_on", "light_off"}


@pytest.mark.asyncio
async def test_transform_stream_empty_content_ignored(provider: OpenAIProvider) -> None:
    """Test that empty content deltas are not yielded."""
    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": ""}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    content_deltas = [d for d in deltas if "content" in d]
    assert len(content_deltas) == 1
    assert content_deltas[0] == {"content": "Hi"}


@pytest.mark.asyncio
async def test_transform_stream_malformed_json_skipped(provider: OpenAIProvider) -> None:
    """Test that malformed JSON chunks are skipped."""
    lines = [
        _sse({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}),
        "data: {invalid json}",
        _sse({"choices": [{"delta": {"content": "recovered"}, "finish_reason": None}]}),
        _sse("[DONE]"),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert {"content": "recovered"} in deltas


@pytest.mark.asyncio
async def test_transform_stream_usage_tracked(provider: OpenAIProvider) -> None:
    """Test that usage data is surfaced via native field."""
    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]}),
        _sse(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ),
        _sse("[DONE]"),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    # Usage is now surfaced via native field
    native_deltas = [d for d in deltas if "native" in d]
    assert len(native_deltas) >= 1
    usage_delta = next(d for d in native_deltas if "usage" in d.get("native", {}))
    assert usage_delta["native"]["usage"]["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_transform_stream_sse_comments_ignored(provider: OpenAIProvider) -> None:
    """Test that SSE comments and blank lines are ignored."""
    lines = [
        ": this is a comment",
        "",
        _sse({"choices": [{"delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]}),
        _sse("[DONE]"),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    assert {"content": "OK"} in deltas


# --- build_payload / build_headers / build_url ---


def test_build_payload_basic(provider: OpenAIProvider) -> None:
    """Test basic payload construction."""
    payload = provider.build_payload(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hi"}],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )
    assert payload["model"] == "gpt-4"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 1024
    assert payload["top_p"] == 0.9
    assert payload["stream"] is True
    assert "tools" not in payload


def test_build_payload_with_tools(provider: OpenAIProvider) -> None:
    """Test that tools are included in payload."""
    tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    payload = provider.build_payload(
        model="gpt-4",
        messages=[],
        system=None,
        tools=tools,
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
    )
    assert payload["tools"] == tools


def test_build_payload_with_seed(provider: OpenAIProvider) -> None:
    """Test that seed is included in payload."""
    payload = provider.build_payload(
        model="gpt-4",
        messages=[],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        extra_options={"seed": 42},
    )
    assert payload["seed"] == 42


def test_build_payload_with_stop_sequences(provider: OpenAIProvider) -> None:
    """Test that stop sequences are included in payload."""
    payload = provider.build_payload(
        model="gpt-4",
        messages=[],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        extra_options={"stop_sequences": ["STOP", "END"]},
    )
    assert payload["stop"] == ["STOP", "END"]


def test_build_payload_with_response_format(provider: OpenAIProvider) -> None:
    """Test that response format is included in payload."""
    payload = provider.build_payload(
        model="gpt-4",
        messages=[],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        extra_options={"response_format": "json_object"},
    )
    assert payload["response_format"] == {"type": "json_object"}


def test_build_payload_with_json_schema(provider: OpenAIProvider) -> None:
    """Test that JSON schema response format includes the schema."""
    schema = {"type": "object", "properties": {"result": {"type": "string"}}}
    payload = provider.build_payload(
        model="gpt-4",
        messages=[],
        system=None,
        tools=[],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.9,
        extra_options={"response_format": "json_schema", "json_schema": schema},
    )
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == schema
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_build_headers_with_key(provider: OpenAIProvider) -> None:
    """Test headers include Authorization bearer token."""
    headers = provider.build_headers("sk-test-key")
    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["Content-Type"] == "application/json"


def test_build_headers_without_key(provider: OpenAIProvider) -> None:
    """Test headers without API key."""
    headers = provider.build_headers(None)
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


def test_build_url(provider: OpenAIProvider) -> None:
    """Test URL ends with /chat/completions."""
    url = provider.build_url("http://localhost:8080/v1")
    assert url == "http://localhost:8080/v1/chat/completions"


def test_build_url_strips_trailing_slash(provider: OpenAIProvider) -> None:
    """Test trailing slash is stripped."""
    url = provider.build_url("http://localhost:8080/v1/")
    assert url == "http://localhost:8080/v1/chat/completions"


@pytest.mark.asyncio
async def test_transform_stream_thinking_content(provider: OpenAIProvider) -> None:
    """Test DeepSeek-style reasoning_content is mapped to thinking_content."""
    lines = [
        _sse(
            {"choices": [{"delta": {"role": "assistant", "reasoning_content": "Thinking..."}, "finish_reason": None}]}
        ),
        _sse({"choices": [{"delta": {"content": "Answer"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    thinking_deltas = [d for d in deltas if "thinking_content" in d]
    assert thinking_deltas == [{"thinking_content": "Thinking..."}]
    content_deltas = [d for d in deltas if "content" in d]
    assert content_deltas == [{"content": "Answer"}]


@pytest.mark.asyncio
async def test_transform_stream_finish_reason_surfaced(provider: OpenAIProvider) -> None:
    """Test non-standard finish reasons (max_tokens, content_filter) are surfaced via native."""
    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Cut off"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "length"}]}),
        _sse("[DONE]"),
    ]
    response = MockStreamResponse(lines)

    deltas = []
    async for delta in provider.transform_stream(response):
        deltas.append(delta)

    native_deltas = [d for d in deltas if "native" in d]
    finish_delta = next(d for d in native_deltas if "finish_reason" in d.get("native", {}))
    assert finish_delta["native"]["finish_reason"] == "length"


# --- Integration tests for _async_handle_chat_log ---


async def _noop_add_delta_stream(entity_id: str, stream: Any) -> AsyncIterator:
    """Mock async_add_delta_content_stream that consumes the stream."""
    async for _ in stream:
        pass
    return
    yield  # makes this an async generator


def _make_entity(
    subentry_data: dict[str, Any] | None = None,
    custom_headers_raw: str | None = None,
) -> tuple[LLMHomeControllerBaseLLMEntity, MagicMock]:
    """Create a test entity with a mock session."""
    sub = MagicMock()
    sub.subentry_id = "test-sub"
    sub.title = "Test Agent"
    sub.data = subentry_data or {
        CONF_MODEL: "test-model",
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_TOP_P: DEFAULT_TOP_P,
    }

    session = MagicMock()
    entity = LLMHomeControllerBaseLLMEntity(
        entry=sub,
        subentry=sub,
        session=session,
        api_url="http://localhost:8080/v1",
        api_key="test-key",
        provider=OpenAIProvider(),
        custom_headers_raw=custom_headers_raw,
    )
    entity.entity_id = "conversation.test_agent"
    return entity, session


def _make_chat_log(*, unresponded: bool = False) -> MagicMock:
    """Create a mock chat log."""
    chat_log = MagicMock()
    chat_log.content = [
        SystemContent(content="Be helpful"),
        UserContent(content="Hello"),
    ]
    chat_log.llm_api = None
    chat_log.unresponded_tool_results = {"tool1"} if unresponded else set()
    chat_log.async_add_delta_content_stream = _noop_add_delta_stream
    return chat_log


def _mock_sse_response(
    lines: list[str],
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> MockStreamResponse:
    """Create a MockStreamResponse with a status code."""
    resp = MockStreamResponse(lines)
    resp.status = status
    resp.headers = headers or {}
    resp.close = MagicMock()
    resp.text = AsyncMock(return_value="Error")
    return resp


@pytest.mark.asyncio
async def test_handle_chat_log_simple_text() -> None:
    """Test _async_handle_chat_log with a simple text response."""
    entity, session = _make_entity()
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi!"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    session.post.assert_called_once()
    call_kwargs = session.post.call_args
    assert "json" in call_kwargs.kwargs
    assert call_kwargs.kwargs["json"]["model"] == "test-model"


@pytest.mark.asyncio
async def test_handle_chat_log_api_401() -> None:
    """Test that 401 raises HomeAssistantError."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 0,
        }
    )
    chat_log = _make_chat_log()

    resp = MagicMock()
    resp.status = 401
    session.post = AsyncMock(return_value=resp)

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="Authentication failed"),
    ):
        await entity._async_handle_chat_log(chat_log)


@pytest.mark.asyncio
async def test_handle_chat_log_api_500() -> None:
    """Test that 500 raises HomeAssistantError with response body."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 0,
        }
    )
    chat_log = _make_chat_log()

    resp = MagicMock()
    resp.status = 500
    resp.headers = {}
    resp.text = AsyncMock(return_value="Internal Server Error")
    session.post = AsyncMock(return_value=resp)

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="HTTP 500"),
    ):
        await entity._async_handle_chat_log(chat_log)


@pytest.mark.asyncio
async def test_handle_chat_log_api_timeout() -> None:
    """Test that timeout raises HomeAssistantError."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 0,
        }
    )
    chat_log = _make_chat_log()

    session.post = AsyncMock(side_effect=TimeoutError("Timed out"))

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="Timeout"),
    ):
        await entity._async_handle_chat_log(chat_log)


@pytest.mark.asyncio
async def test_handle_chat_log_connection_error() -> None:
    """Test that connection error raises HomeAssistantError."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 0,
        }
    )
    chat_log = _make_chat_log()

    session.post = AsyncMock(side_effect=aiohttp.ClientError("Connection refused"))

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="Error communicating"),
    ):
        await entity._async_handle_chat_log(chat_log)


@pytest.mark.asyncio
async def test_handle_chat_log_max_iterations() -> None:
    """Test that the loop stops after MAX_TOOL_ITERATIONS."""
    entity, session = _make_entity()
    chat_log = _make_chat_log(unresponded=True)

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "x"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    # Each iteration returns a valid response, but unresponded_tool_results stays truthy
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    from custom_components.llm_home_controller.const import MAX_TOOL_ITERATIONS

    assert session.post.call_count == MAX_TOOL_ITERATIONS


@pytest.mark.asyncio
async def test_handle_chat_log_tracing() -> None:
    """Test that trace events are emitted."""
    entity, session = _make_entity()
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch(
        "custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"
    ) as mock_trace:
        await entity._async_handle_chat_log(chat_log)

    # Should have at least 2 trace calls: initial request details + completion summary
    assert mock_trace.call_count == 2


# --- Context pruning tests ---


def test_estimate_tokens() -> None:
    """Test token estimation (~4 chars per token)."""
    assert _estimate_tokens("") == 1  # len 0 // 4 + 1 = 1
    assert _estimate_tokens("abcd") == 2  # 4 // 4 + 1 = 2
    assert _estimate_tokens("a" * 100) == 26  # 100 // 4 + 1 = 26


def test_prune_messages_empty() -> None:
    """Test pruning with empty messages."""
    result = _prune_messages([], None, 1000)
    assert result == []


def test_prune_messages_under_budget() -> None:
    """Test messages that fit within budget are returned unchanged."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = _prune_messages(messages, None, 100000)
    assert result == messages


def test_prune_messages_over_budget_removes_oldest() -> None:
    """Test that oldest messages are removed when over budget."""
    messages = [
        {"role": "user", "content": "A" * 400},  # ~100 tokens
        {"role": "assistant", "content": "B" * 400},  # ~100 tokens
        {"role": "user", "content": "C" * 400},  # ~100 tokens
    ]
    # Budget of 250 tokens should drop the first message
    result = _prune_messages(messages, None, 250)
    assert len(result) == 2
    assert result[0]["content"] == "B" * 400
    assert result[1]["content"] == "C" * 400


def test_prune_messages_always_keeps_last() -> None:
    """Test that the last message is always kept even if over budget."""
    messages = [
        {"role": "user", "content": "A" * 4000},  # ~1000 tokens
    ]
    result = _prune_messages(messages, None, 10)
    assert len(result) == 1
    assert result[0] == messages[0]


def test_prune_messages_accounts_for_system() -> None:
    """Test that system prompt tokens reduce the available budget."""
    messages = [
        {"role": "user", "content": "A" * 200},  # ~50 tokens
        {"role": "assistant", "content": "B" * 200},  # ~50 tokens
    ]
    # Large system prompt eats into the budget
    system = "S" * 800  # ~200 tokens
    result = _prune_messages(messages, system, 250)
    # Budget = 250 - 200 = 50 tokens, so first message should be dropped
    assert len(result) == 1
    assert result[0]["content"] == "B" * 200


def test_prune_messages_zero_budget_keeps_last() -> None:
    """Test that when system prompt exceeds budget, only last message kept."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    system = "S" * 40000  # Way bigger than budget
    result = _prune_messages(messages, system, 100)
    assert len(result) == 1
    assert result[0]["content"] == "Hi"


@pytest.mark.asyncio
async def test_handle_chat_log_with_context_pruning() -> None:
    """Test that context pruning is applied when max_context_tokens is set."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_CONTEXT_TOKENS: 50,  # Very small budget
        }
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi!"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    # Verify the payload was sent (pruning shouldn't break anything)
    session.post.assert_called_once()


@pytest.mark.asyncio
async def test_handle_chat_log_no_pruning_when_zero() -> None:
    """Test that pruning is skipped when max_context_tokens is 0 (default)."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_CONTEXT_TOKENS: 0,
        }
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi!"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with (
        patch("custom_components.llm_home_controller.entity._prune_messages", wraps=_prune_messages) as mock_prune,
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
    ):
        await entity._async_handle_chat_log(chat_log)

    # _prune_messages should NOT be called when max_context_tokens is 0
    mock_prune.assert_not_called()


# --- Feature 3: Seed parameter ---


@pytest.mark.asyncio
async def test_handle_chat_log_seed_in_payload() -> None:
    """Test that seed is passed through to the provider payload."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_SEED: 42,
        }
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi!"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    payload = session.post.call_args.kwargs["json"]
    assert payload["seed"] == 42


@pytest.mark.asyncio
async def test_handle_chat_log_seed_not_set() -> None:
    """Test that seed is not included when not configured."""
    entity, session = _make_entity()
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi!"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    payload = session.post.call_args.kwargs["json"]
    assert "seed" not in payload


# --- Feature 4: Stop sequences ---


@pytest.mark.asyncio
async def test_handle_chat_log_stop_sequences() -> None:
    """Test that stop sequences are parsed and passed through."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_STOP_SEQUENCES: "STOP, END, HALT",
        }
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi!"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    payload = session.post.call_args.kwargs["json"]
    assert payload["stop"] == ["STOP", "END", "HALT"]


@pytest.mark.asyncio
async def test_handle_chat_log_stop_sequences_empty_ignored() -> None:
    """Test that empty stop sequences are ignored."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_STOP_SEQUENCES: "  ,  ,  ",
        }
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi!"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    payload = session.post.call_args.kwargs["json"]
    assert "stop" not in payload


# --- Feature 5: Response format ---


@pytest.mark.asyncio
async def test_handle_chat_log_response_format_json() -> None:
    """Test that response format json_object is passed through."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_RESPONSE_FORMAT: "json_object",
        }
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "{}"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    payload = session.post.call_args.kwargs["json"]
    assert payload["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_handle_chat_log_response_format_text_ignored() -> None:
    """Test that 'text' response format is not included (it's the default)."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_RESPONSE_FORMAT: "text",
        }
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    payload = session.post.call_args.kwargs["json"]
    assert "response_format" not in payload


# --- Feature 7: Custom HTTP headers ---


@pytest.mark.asyncio
async def test_handle_chat_log_custom_headers_merged() -> None:
    """Test that custom HTTP headers are merged into requests."""
    entity, session = _make_entity(
        custom_headers_raw='{"X-Custom": "value", "X-Another": "test"}',
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    headers = session.post.call_args.kwargs["headers"]
    assert headers["X-Custom"] == "value"
    assert headers["X-Another"] == "test"
    # Provider headers should still be present
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_handle_chat_log_custom_headers_empty() -> None:
    """Test that empty custom headers don't break anything."""
    entity, session = _make_entity(custom_headers_raw=None)
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    headers = session.post.call_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_handle_chat_log_custom_headers_invalid_json() -> None:
    """Test that invalid JSON in custom headers is ignored."""
    entity, session = _make_entity(custom_headers_raw="{not valid json}")
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    # Should still work, just without custom headers
    headers = session.post.call_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"
    assert "X-Custom" not in headers


@pytest.mark.asyncio
async def test_handle_chat_log_custom_headers_override() -> None:
    """Test that custom headers can override provider headers."""
    entity, session = _make_entity(
        custom_headers_raw='{"Authorization": "Token custom-token"}',
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    headers = session.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Token custom-token"


# --- Feature 8: Retry with backoff ---


@pytest.mark.asyncio
async def test_retry_on_429() -> None:
    """Test that 429 is retried."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 2,
        }
    )
    chat_log = _make_chat_log()

    resp_429 = MagicMock()
    resp_429.status = 429
    resp_429.headers = {}
    resp_429.text = AsyncMock(return_value="Rate limited")

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    resp_ok = _mock_sse_response(lines)

    session.post = AsyncMock(side_effect=[resp_429, resp_ok])

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        patch("custom_components.llm_home_controller.entity.asyncio.sleep", new_callable=AsyncMock),
    ):
        await entity._async_handle_chat_log(chat_log)

    assert session.post.call_count == 2


@pytest.mark.asyncio
async def test_retry_on_500() -> None:
    """Test that 500 is retried."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 1,
        }
    )
    chat_log = _make_chat_log()

    resp_500 = MagicMock()
    resp_500.status = 500
    resp_500.headers = {}
    resp_500.text = AsyncMock(return_value="Server Error")

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    resp_ok = _mock_sse_response(lines)

    session.post = AsyncMock(side_effect=[resp_500, resp_ok])

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        patch("custom_components.llm_home_controller.entity.asyncio.sleep", new_callable=AsyncMock),
    ):
        await entity._async_handle_chat_log(chat_log)

    assert session.post.call_count == 2


@pytest.mark.asyncio
async def test_no_retry_on_401() -> None:
    """Test that 401 is NOT retried (auth error)."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 3,
        }
    )
    chat_log = _make_chat_log()

    resp_401 = MagicMock()
    resp_401.status = 401
    session.post = AsyncMock(return_value=resp_401)

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="Authentication failed"),
    ):
        await entity._async_handle_chat_log(chat_log)

    # Should NOT retry on auth errors
    assert session.post.call_count == 1


@pytest.mark.asyncio
async def test_no_retry_on_400() -> None:
    """Test that 400 is NOT retried (client error)."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 3,
        }
    )
    chat_log = _make_chat_log()

    resp_400 = MagicMock()
    resp_400.status = 400
    resp_400.text = AsyncMock(return_value="Bad Request")
    session.post = AsyncMock(return_value=resp_400)

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="HTTP 400"),
    ):
        await entity._async_handle_chat_log(chat_log)

    assert session.post.call_count == 1


@pytest.mark.asyncio
async def test_retry_on_timeout() -> None:
    """Test that timeout errors are retried."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 1,
        }
    )
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    resp_ok = _mock_sse_response(lines)

    session.post = AsyncMock(side_effect=[TimeoutError("Timeout"), resp_ok])

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        patch("custom_components.llm_home_controller.entity.asyncio.sleep", new_callable=AsyncMock),
    ):
        await entity._async_handle_chat_log(chat_log)

    assert session.post.call_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted_raises() -> None:
    """Test that exhausted retries raise the error."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 1,
        }
    )
    chat_log = _make_chat_log()

    resp_500 = MagicMock()
    resp_500.status = 500
    resp_500.headers = {}
    resp_500.text = AsyncMock(return_value="Server Error")

    session.post = AsyncMock(return_value=resp_500)

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        patch("custom_components.llm_home_controller.entity.asyncio.sleep", new_callable=AsyncMock),
        pytest.raises(HomeAssistantError, match="HTTP 500"),
    ):
        await entity._async_handle_chat_log(chat_log)

    # Initial + 1 retry = 2 attempts
    assert session.post.call_count == 2


@pytest.mark.asyncio
async def test_retry_after_header_respected() -> None:
    """Test that Retry-After header delay is used."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 1,
        }
    )
    chat_log = _make_chat_log()

    resp_429 = MagicMock()
    resp_429.status = 429
    resp_429.headers = {"Retry-After": "5"}
    resp_429.text = AsyncMock(return_value="Rate limited")

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    resp_ok = _mock_sse_response(lines)

    session.post = AsyncMock(side_effect=[resp_429, resp_ok])

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        patch("custom_components.llm_home_controller.entity.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        await entity._async_handle_chat_log(chat_log)

    # Retry-After: 5 should result in sleep(5.0)
    mock_sleep.assert_called_once_with(5.0)


@pytest.mark.asyncio
async def test_retries_disabled() -> None:
    """Test that max_retries=0 disables retries."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "test-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 0,
        }
    )
    chat_log = _make_chat_log()

    resp_500 = MagicMock()
    resp_500.status = 500
    resp_500.headers = {}
    resp_500.text = AsyncMock(return_value="Server Error")
    session.post = AsyncMock(return_value=resp_500)

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="HTTP 500"),
    ):
        await entity._async_handle_chat_log(chat_log)

    # Only 1 attempt, no retries
    assert session.post.call_count == 1


# --- Feature 9: Model fallback ---


@pytest.mark.asyncio
async def test_fallback_on_primary_failure() -> None:
    """Test fallback model is tried when primary fails."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "primary-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 0,
            CONF_FALLBACK_MODEL: "fallback-model",
        }
    )
    chat_log = _make_chat_log()

    resp_500 = MagicMock()
    resp_500.status = 500
    resp_500.headers = {}
    resp_500.text = AsyncMock(return_value="Server Error")

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "OK"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    resp_ok = _mock_sse_response(lines)

    # First call (primary) fails, second call (fallback) succeeds
    session.post = AsyncMock(side_effect=[resp_500, resp_ok])

    with patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"):
        await entity._async_handle_chat_log(chat_log)

    assert session.post.call_count == 2
    # Second call should use fallback model
    second_call_payload = session.post.call_args_list[1].kwargs["json"]
    assert second_call_payload["model"] == "fallback-model"


@pytest.mark.asyncio
async def test_no_fallback_on_auth_error() -> None:
    """Test fallback is NOT tried on authentication errors."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "primary-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 0,
            CONF_FALLBACK_MODEL: "fallback-model",
        }
    )
    chat_log = _make_chat_log()

    resp_401 = MagicMock()
    resp_401.status = 401
    session.post = AsyncMock(return_value=resp_401)

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="Authentication failed"),
    ):
        await entity._async_handle_chat_log(chat_log)

    # Should NOT try fallback
    assert session.post.call_count == 1


@pytest.mark.asyncio
async def test_fallback_not_configured() -> None:
    """Test that without fallback model, error is raised directly."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "primary-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 0,
        }
    )
    chat_log = _make_chat_log()

    resp_500 = MagicMock()
    resp_500.status = 500
    resp_500.headers = {}
    resp_500.text = AsyncMock(return_value="Server Error")
    session.post = AsyncMock(return_value=resp_500)

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="HTTP 500"),
    ):
        await entity._async_handle_chat_log(chat_log)

    assert session.post.call_count == 1


@pytest.mark.asyncio
async def test_fallback_also_fails() -> None:
    """Test that when fallback also fails, primary error is raised."""
    entity, session = _make_entity(
        subentry_data={
            CONF_MODEL: "primary-model",
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TOP_P: DEFAULT_TOP_P,
            CONF_MAX_RETRIES: 0,
            CONF_FALLBACK_MODEL: "fallback-model",
        }
    )
    chat_log = _make_chat_log()

    resp_500 = MagicMock()
    resp_500.status = 500
    resp_500.headers = {}
    resp_500.text = AsyncMock(return_value="Server Error")

    resp_500b = MagicMock()
    resp_500b.status = 500
    resp_500b.headers = {}
    resp_500b.text = AsyncMock(return_value="Fallback also failed")

    session.post = AsyncMock(side_effect=[resp_500, resp_500b])

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        pytest.raises(HomeAssistantError, match="Server Error"),
    ):
        await entity._async_handle_chat_log(chat_log)

    assert session.post.call_count == 2


# --- Feature 10: Usage signal ---


@pytest.mark.asyncio
async def test_usage_signal_fired() -> None:
    """Test that usage data from stream fires a dispatcher signal."""
    entity, session = _make_entity()
    entity.hass = MagicMock(spec=HomeAssistant)
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse({"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        patch("custom_components.llm_home_controller.entity.async_dispatcher_send") as mock_send,
    ):
        await entity._async_handle_chat_log(chat_log)

    mock_send.assert_called_once_with(
        entity.hass,
        f"llm_home_controller_usage_{entity.subentry.subentry_id}",
        50,
        20,
    )


@pytest.mark.asyncio
async def test_no_usage_signal_when_no_usage_data() -> None:
    """Test that no signal is fired when stream has no usage data."""
    entity, session = _make_entity()
    entity.hass = MagicMock(spec=HomeAssistant)
    chat_log = _make_chat_log()

    lines = [
        _sse({"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        _sse("[DONE]"),
    ]
    session.post = AsyncMock(return_value=_mock_sse_response(lines))

    with (
        patch("custom_components.llm_home_controller.entity.conversation.async_conversation_trace_append"),
        patch("custom_components.llm_home_controller.entity.async_dispatcher_send") as mock_send,
    ):
        await entity._async_handle_chat_log(chat_log)

    mock_send.assert_not_called()


# --- Feature: Custom tools ---


def test_parse_custom_tools_valid() -> None:
    """Test parsing valid custom tool definitions."""
    from custom_components.llm_home_controller.entity import _parse_custom_tools

    hass = MagicMock(spec=HomeAssistant)
    raw = json.dumps(
        [
            {
                "name": "play_music",
                "description": "Play music",
                "service": "media_player.play_media",
                "parameters": {
                    "type": "object",
                    "properties": {"song": {"type": "string"}},
                    "required": ["song"],
                },
                "service_data": {"media_content_type": "music"},
            }
        ]
    )
    tools = _parse_custom_tools(raw, hass)
    assert len(tools) == 1
    assert tools[0].name == "play_music"
    assert tools[0].description == "Play music"


def test_parse_custom_tools_invalid_json() -> None:
    """Test that invalid JSON returns empty list."""
    from custom_components.llm_home_controller.entity import _parse_custom_tools

    hass = MagicMock(spec=HomeAssistant)
    tools = _parse_custom_tools("{not valid json}", hass)
    assert tools == []


def test_parse_custom_tools_missing_required_fields() -> None:
    """Test that tools without name or service are skipped."""
    from custom_components.llm_home_controller.entity import _parse_custom_tools

    hass = MagicMock(spec=HomeAssistant)
    raw = json.dumps(
        [
            {"name": "no_service", "description": "Missing service"},
            {"service": "test.test", "description": "Missing name"},
            {"name": "valid", "service": "test.test", "description": "OK"},
        ]
    )
    tools = _parse_custom_tools(raw, hass)
    assert len(tools) == 1
    assert tools[0].name == "valid"


def test_parse_custom_tools_not_a_list() -> None:
    """Test that non-array JSON returns empty list."""
    from custom_components.llm_home_controller.entity import _parse_custom_tools

    hass = MagicMock(spec=HomeAssistant)
    tools = _parse_custom_tools('{"name": "test"}', hass)
    assert tools == []


@pytest.mark.asyncio
async def test_custom_service_tool_call() -> None:
    """Test that CustomServiceTool executes the HA service call."""
    from custom_components.llm_home_controller.entity import CustomServiceTool

    hass = MagicMock(spec=HomeAssistant)
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock(return_value={"result": "ok"})

    tool = CustomServiceTool(
        hass=hass,
        name="test_tool",
        description="Test",
        parameters_schema=None,
        service="test_domain.test_service",
        service_data_template={"key": "default_value"},
    )

    tool_input = MagicMock()
    tool_input.tool_args = {"extra_key": "extra_value"}

    result = await tool.async_call(hass, tool_input, MagicMock())

    assert result["success"] is True
    hass.services.async_call.assert_called_once_with(
        "test_domain",
        "test_service",
        {"key": "default_value", "extra_key": "extra_value"},
        blocking=True,
        return_response=True,
    )


@pytest.mark.asyncio
async def test_custom_service_tool_call_error() -> None:
    """Test that CustomServiceTool handles service call errors."""
    from custom_components.llm_home_controller.entity import CustomServiceTool

    hass = MagicMock(spec=HomeAssistant)
    hass.services = MagicMock()
    hass.services.async_call = AsyncMock(side_effect=Exception("Service failed"))

    tool = CustomServiceTool(
        hass=hass,
        name="test_tool",
        description="Test",
        parameters_schema=None,
        service="test_domain.test_service",
    )

    tool_input = MagicMock()
    tool_input.tool_args = {}

    result = await tool.async_call(hass, tool_input, MagicMock())
    assert "error" in result
    assert "Service failed" in result["error"]
