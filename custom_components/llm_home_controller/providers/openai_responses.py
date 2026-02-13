"""OpenAI Responses API provider."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
from homeassistant.components.conversation import (
    AssistantContent,
    AssistantContentDeltaDict,
    Content,
    ToolResultContent,
    UserContent,
)
from homeassistant.helpers import llm
from voluptuous_openapi import convert as vol_to_openapi

_LOGGER = logging.getLogger(__name__)


class OpenAIResponsesProvider:
    """Provider for OpenAI /v1/responses API."""

    def format_tools(self, tools: list[llm.Tool], custom_serializer=None) -> list[dict[str, Any]]:
        """Convert HA tools to Responses API tool format.

        Responses API uses flat internally-tagged format:
        {"type": "function", "name": ..., "parameters": ...}
        """
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": vol_to_openapi(
                    tool.parameters,
                    custom_serializer=custom_serializer,
                ),
            }
            for tool in tools
        ]

    def convert_content(
        self,
        chat_content: list[Content],
    ) -> dict[str, Any]:
        """Convert ChatLog content to Responses API input format.

        Returns {"messages": input_items, "system": instructions_text}.
        The "messages" key holds Responses API input items; build_payload
        maps it to the "input" parameter.
        """
        input_items: list[dict[str, Any]] = []
        system_text: str | None = None

        for content in chat_content:
            if content.role == "system":
                system_text = content.content
                continue

            if isinstance(content, ToolResultContent):
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": content.tool_call_id,
                        "output": json.dumps(content.tool_result),
                    }
                )
                continue

            if content.role == "user":
                if isinstance(content, UserContent) and content.attachments:
                    parts: list[dict[str, Any]] = [{"type": "input_text", "text": content.content}]
                    for att in content.attachments:
                        try:
                            data = base64.b64encode(att.path.read_bytes()).decode("utf-8")
                        except (FileNotFoundError, OSError):
                            _LOGGER.warning("Failed to read attachment: %s", att.path)
                            continue
                        parts.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{att.mime_type};base64,{data}",
                            }
                        )
                    input_items.append({"role": "user", "content": parts})
                else:
                    input_items.append({"role": "user", "content": content.content})

            elif isinstance(content, AssistantContent):
                if content.thinking_content:
                    input_items.append(
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": content.thinking_content}],
                        }
                    )
                if content.content:
                    input_items.append({"role": "assistant", "content": content.content})
                if content.tool_calls:
                    for tc in content.tool_calls:
                        input_items.append(
                            {
                                "type": "function_call",
                                "id": tc.id,
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.tool_args),
                            }
                        )

        return {"messages": input_items, "system": system_text}

    def build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        extra_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the Responses API request payload."""
        payload: dict[str, Any] = {
            "model": model,
            "input": messages,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }
        if system:
            payload["instructions"] = system
        if tools:
            payload["tools"] = tools
        if extra_options:
            if stops := extra_options.get("stop_sequences"):
                payload["stop"] = stops
            if fmt := extra_options.get("response_format"):
                if fmt == "json_schema" and (schema := extra_options.get("json_schema")):
                    payload["text"] = {"format": {"type": "json_schema", "name": "custom_output", "schema": schema}}
                else:
                    payload["text"] = {"format": {"type": fmt}}
            if extra_options.get("extended_thinking"):
                payload["reasoning"] = {"effort": "high", "summary": "auto"}
        return payload

    def build_headers(self, api_key: str | None) -> dict[str, str]:
        """Build HTTP headers for OpenAI Responses API."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def build_url(self, api_url: str) -> str:
        """Build the responses endpoint URL."""
        return f"{api_url.rstrip('/')}/responses"

    async def transform_stream(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncGenerator[AssistantContentDeltaDict]:
        """Transform Responses API SSE stream into HA delta dicts.

        Key event types:
        - response.created → emit role
        - response.output_text.delta → text content
        - response.function_call_arguments.delta → buffer tool args
        - response.output_item.done → finalize tool calls
        - response.completed → usage info
        """
        first_chunk = True
        function_arg_buffers: dict[int, str] = {}

        async for line in response.content:
            text = line.decode("utf-8").rstrip("\n\r")

            if text.startswith("event: "):
                continue

            if not text.startswith("data: "):
                continue

            data_str = text[6:]
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                _LOGGER.warning("Failed to parse Responses API SSE chunk: %s", data_str)
                continue

            chunk_type = chunk.get("type", "")

            if chunk_type == "response.created":
                if first_chunk:
                    yield {"role": "assistant"}
                    first_chunk = False
                continue

            if chunk_type == "response.output_item.added":
                item = chunk.get("item", {})
                output_index = chunk.get("output_index", 0)
                if item.get("type") == "function_call":
                    function_arg_buffers[output_index] = ""
                if first_chunk:
                    yield {"role": "assistant"}
                    first_chunk = False
                continue

            if chunk_type == "response.reasoning_summary_text.delta":
                if first_chunk:
                    yield {"role": "assistant"}
                    first_chunk = False
                if delta := chunk.get("delta", ""):
                    yield {"thinking_content": delta}
                continue

            if chunk_type == "response.output_text.delta":
                if first_chunk:
                    yield {"role": "assistant"}
                    first_chunk = False
                if delta := chunk.get("delta", ""):
                    yield {"content": delta}
                continue

            if chunk_type == "response.function_call_arguments.delta":
                output_index = chunk.get("output_index", 0)
                if partial := chunk.get("delta", ""):
                    function_arg_buffers[output_index] = function_arg_buffers.get(output_index, "") + partial
                continue

            if chunk_type == "response.output_item.done":
                item = chunk.get("item", {})
                output_index = chunk.get("output_index", 0)
                if item.get("type") == "function_call":
                    call_id = item.get("call_id", item.get("id", ""))
                    name = item.get("name", "")
                    arguments_str = item.get("arguments", function_arg_buffers.pop(output_index, "{}"))
                    try:
                        tool_args = json.loads(arguments_str) if arguments_str else {}
                    except json.JSONDecodeError:
                        _LOGGER.warning(
                            "Failed to parse function call arguments: %s",
                            arguments_str,
                        )
                        tool_args = {}
                    function_arg_buffers.pop(output_index, None)
                    yield {
                        "tool_calls": [
                            llm.ToolInput(
                                id=call_id,
                                tool_name=name,
                                tool_args=tool_args,
                            )
                        ]
                    }
                continue

            if chunk_type == "response.completed":
                resp_data = chunk.get("response", {})
                if usage := resp_data.get("usage"):
                    yield {"native": {"usage": usage}}
                break

            if chunk_type == "response.incomplete":
                yield {"native": {"finish_reason": "incomplete"}}
                break

    async def get_models(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        api_key: str | None,
    ) -> list[str]:
        """Fetch available models from OpenAI /v1/models endpoint."""
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{api_url.rstrip('/')}/models"
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return sorted(model["id"] for model in data.get("data", []))
