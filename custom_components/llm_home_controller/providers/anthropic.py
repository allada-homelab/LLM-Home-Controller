"""Anthropic Messages API provider."""

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

from custom_components.llm_home_controller.const import ANTHROPIC_API_VERSION

_LOGGER = logging.getLogger(__name__)


class AnthropicProvider:
    """Provider for Anthropic /v1/messages API."""

    def format_tools(self, tools: list[llm.Tool], custom_serializer=None) -> list[dict[str, Any]]:
        """Convert HA tools to Anthropic tool format."""
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": vol_to_openapi(
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
        """Convert ChatLog content to Anthropic messages format.

        Anthropic requires:
        1. System message is SEPARATE (returned as "system" key)
        2. Messages must alternate user/assistant roles
        3. Tool results go inside user messages as tool_result blocks
        4. Tool calls are content blocks inside assistant messages
        """
        system_text: str | None = None
        messages: list[dict[str, Any]] = []

        for content in chat_content:
            if content.role == "system":
                system_text = content.content
                continue

            if isinstance(content, ToolResultContent):
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": content.tool_call_id,
                    "content": json.dumps(content.tool_result),
                }
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"].append(tool_result_block)
                else:
                    messages.append({"role": "user", "content": [tool_result_block]})
                continue

            if content.role == "user":
                user_blocks: list[dict[str, Any]] = [{"type": "text", "text": content.content}]
                if isinstance(content, UserContent) and content.attachments:
                    for att in content.attachments:
                        block = self._attachment_to_block(att)
                        if block is not None:
                            user_blocks.append(block)
                if messages and messages[-1]["role"] == "user":
                    messages[-1]["content"].extend(user_blocks)
                else:
                    messages.append({"role": "user", "content": user_blocks})

            elif isinstance(content, AssistantContent):
                blocks: list[dict[str, Any]] = []
                if content.thinking_content:
                    blocks.append({"type": "thinking", "thinking": content.thinking_content})
                if content.content:
                    blocks.append({"type": "text", "text": content.content})
                if content.tool_calls:
                    for tc in content.tool_calls:
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.tool_name,
                                "input": tc.tool_args,
                            }
                        )
                if blocks:
                    if messages and messages[-1]["role"] == "assistant":
                        messages[-1]["content"].extend(blocks)
                    else:
                        messages.append({"role": "assistant", "content": blocks})

        return {"messages": messages, "system": system_text}

    @staticmethod
    def _attachment_to_block(att: Any) -> dict[str, Any] | None:
        """Convert HA Attachment to Anthropic content block."""
        try:
            data = base64.b64encode(att.path.read_bytes()).decode("utf-8")
        except (FileNotFoundError, OSError):
            _LOGGER.warning("Failed to read attachment: %s", att.path)
            return None
        if att.mime_type == "application/pdf":
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": att.mime_type,
                    "data": data,
                },
            }
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": att.mime_type,
                "data": data,
            },
        }

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
        """Build the Anthropic Messages API request payload."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        # Extended thinking support
        if extra_options and extra_options.get("extended_thinking"):
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": extra_options.get("thinking_budget", 10000),
            }
            # Anthropic disallows temperature/top_p with extended thinking
        else:
            payload["temperature"] = temperature
            payload["top_p"] = top_p

        if extra_options and (stops := extra_options.get("stop_sequences")):
            payload["stop_sequences"] = stops

        # Anthropic doesn't support response_format natively — inject into system prompt
        if extra_options and extra_options.get("response_format") == "json_schema":
            schema = extra_options.get("json_schema", {})
            schema_instruction = (
                "\n\nYou must respond with valid JSON matching this schema:\n"
                f"```json\n{json.dumps(schema, indent=2)}\n```"
            )
            if "system" in payload:
                payload["system"] += schema_instruction
            else:
                payload["system"] = schema_instruction.strip()

        return payload

    def build_headers(self, api_key: str | None) -> dict[str, str]:
        """Build HTTP headers for Anthropic API."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_API_VERSION,
        }
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def build_url(self, api_url: str) -> str:
        """Build the messages endpoint URL."""
        return f"{api_url.rstrip('/')}/messages"

    async def transform_stream(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncGenerator[AssistantContentDeltaDict]:
        """Transform Anthropic SSE stream into HA delta dicts.

        Anthropic SSE format uses event: + data: lines with event types:
        message_start, content_block_start, content_block_delta,
        content_block_stop, message_delta, message_stop, ping.
        """
        first_chunk = True
        current_tool_calls: dict[int, dict[str, Any]] = {}
        tool_input_buffers: dict[int, str] = {}

        async for line in response.content:
            text = line.decode("utf-8").rstrip("\n\r")

            # Parse event type line (ignored — we use the "type" field in data)
            if text.startswith("event: "):
                continue

            if not text.startswith("data: "):
                continue

            data_str = text[6:]
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                _LOGGER.warning("Failed to parse Anthropic SSE chunk: %s", data_str)
                continue

            chunk_type = chunk.get("type", "")

            if chunk_type == "message_start":
                if first_chunk:
                    yield {"role": "assistant"}
                    first_chunk = False
                message = chunk.get("message", {})
                if usage := message.get("usage"):
                    yield {"native": {"usage": usage}}
                continue

            if chunk_type == "content_block_start":
                block = chunk.get("content_block", {})
                index = chunk.get("index", 0)
                if block.get("type") == "tool_use":
                    current_tool_calls[index] = {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                    }
                    tool_input_buffers[index] = ""
                continue

            if chunk_type == "content_block_delta":
                delta = chunk.get("delta", {})
                delta_type = delta.get("type", "")

                if delta_type == "text_delta":
                    if text_val := delta.get("text"):
                        yield {"content": text_val}
                elif delta_type == "thinking_delta":
                    if thinking_val := delta.get("thinking"):
                        yield {"thinking_content": thinking_val}
                elif delta_type == "input_json_delta":
                    index = chunk.get("index", 0)
                    if partial := delta.get("partial_json", ""):
                        tool_input_buffers[index] = tool_input_buffers.get(index, "") + partial
                elif delta_type == "signature_delta":
                    pass  # Signature deltas are for verification, not content
                continue

            if chunk_type == "content_block_stop":
                index = chunk.get("index", 0)
                if index in current_tool_calls:
                    tc = current_tool_calls.pop(index)
                    input_json = tool_input_buffers.pop(index, "{}")
                    try:
                        tool_args = json.loads(input_json) if input_json else {}
                    except json.JSONDecodeError:
                        _LOGGER.warning("Failed to parse tool input JSON: %s", input_json)
                        tool_args = {}
                    yield {
                        "tool_calls": [
                            llm.ToolInput(
                                id=tc["id"],
                                tool_name=tc["name"],
                                tool_args=tool_args,
                            )
                        ]
                    }
                continue

            if chunk_type == "message_delta":
                delta = chunk.get("delta", {})
                stop_reason = delta.get("stop_reason")
                if stop_reason and stop_reason not in ("end_turn", "tool_use"):
                    yield {"native": {"finish_reason": stop_reason}}
                if usage := chunk.get("usage"):
                    yield {"native": {"usage": usage}}
                continue

            if chunk_type == "message_stop":
                break

            if chunk_type == "ping":
                continue

    async def get_models(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        api_key: str | None,
    ) -> list[str]:
        """Fetch models from Anthropic-compatible endpoint.

        Try /v1/models first (works for proxies). Fall back to empty list
        to allow manual model input in the config flow.
        """
        headers: dict[str, str] = {
            "anthropic-version": ANTHROPIC_API_VERSION,
        }
        if api_key:
            headers["x-api-key"] = api_key

        url = f"{api_url.rstrip('/')}/models"
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "data" in data:
                        return sorted(model["id"] for model in data["data"])
        except (aiohttp.ClientError, TimeoutError):
            pass

        return []
