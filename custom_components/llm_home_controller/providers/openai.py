"""OpenAI-compatible API provider."""

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


class OpenAIProvider:
    """Provider for OpenAI-compatible /v1/chat/completions API."""

    def format_tools(self, tools: list[llm.Tool], custom_serializer=None) -> list[dict[str, Any]]:
        """Convert HA tools to OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": vol_to_openapi(
                        tool.parameters,
                        custom_serializer=custom_serializer,
                    ),
                },
            }
            for tool in tools
        ]

    def convert_content(
        self,
        chat_content: list[Content],
    ) -> dict[str, Any]:
        """Convert ChatLog content to OpenAI messages format."""
        messages: list[dict[str, Any]] = []

        for content in chat_content:
            if isinstance(content, ToolResultContent):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": content.tool_call_id,
                        "content": json.dumps(content.tool_result),
                    }
                )
                continue

            if content.role == "system":
                messages.append({"role": "system", "content": content.content})
            elif content.role == "user":
                if isinstance(content, UserContent) and content.attachments:
                    parts: list[dict[str, Any]] = [{"type": "text", "text": content.content}]
                    for att in content.attachments:
                        try:
                            data = base64.b64encode(att.path.read_bytes()).decode("utf-8")
                        except (FileNotFoundError, OSError):
                            _LOGGER.warning("Failed to read attachment: %s", att.path)
                            continue
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{att.mime_type};base64,{data}"},
                            }
                        )
                    messages.append({"role": "user", "content": parts})
                else:
                    messages.append({"role": "user", "content": content.content})
            elif isinstance(content, AssistantContent):
                msg: dict[str, Any] = {"role": "assistant"}
                if content.content:
                    msg["content"] = content.content
                if content.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.tool_args),
                            },
                        }
                        for tc in content.tool_calls
                    ]
                messages.append(msg)

        return {"messages": messages, "system": None}

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
        """Build the OpenAI-compatible request payload."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if extra_options:
            if (seed := extra_options.get("seed")) is not None:
                payload["seed"] = seed
            if stops := extra_options.get("stop_sequences"):
                payload["stop"] = stops
            if fmt := extra_options.get("response_format"):
                if fmt == "json_schema" and (schema := extra_options.get("json_schema")):
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {"name": "custom_output", "strict": True, "schema": schema},
                    }
                else:
                    payload["response_format"] = {"type": fmt}
        return payload

    def build_headers(self, api_key: str | None) -> dict[str, str]:
        """Build HTTP headers for OpenAI-compatible API."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def build_url(self, api_url: str) -> str:
        """Build the chat completions endpoint URL."""
        return f"{api_url.rstrip('/')}/chat/completions"

    async def transform_stream(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncGenerator[AssistantContentDeltaDict]:
        """Transform OpenAI SSE stream into HA delta dicts."""
        current_tool_calls: dict[int, dict[str, Any]] = {}
        first_chunk = True

        async for line in response.content:
            text = line.decode("utf-8").strip()

            if not text or text.startswith(":"):
                continue

            if not text.startswith("data: "):
                continue

            data_str = text[6:]
            if data_str == "[DONE]":
                if current_tool_calls:
                    yield {
                        "tool_calls": [
                            llm.ToolInput(
                                id=tc["id"],
                                tool_name=tc["function"]["name"],
                                tool_args=json.loads(tc["function"]["arguments"]),
                            )
                            for tc in current_tool_calls.values()
                        ]
                    }
                    current_tool_calls.clear()
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                _LOGGER.warning("Failed to parse SSE chunk: %s", data_str)
                continue

            choices = chunk.get("choices", [])
            if not choices:
                if chunk.get("usage"):
                    yield {"native": {"usage": chunk["usage"]}}
                continue

            delta = choices[0].get("delta", {})
            finish_reason = choices[0].get("finish_reason")

            if first_chunk:
                yield {"role": "assistant"}
                first_chunk = False

            if delta.get("content"):
                yield {"content": delta["content"]}

            # Support reasoning_content for DeepSeek-style extended thinking
            if delta.get("reasoning_content"):
                yield {"thinking_content": delta["reasoning_content"]}

            if "tool_calls" in delta:
                for tc_delta in delta["tool_calls"]:
                    idx = tc_delta["index"]
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tc_delta.get("id", ""),
                            "function": {
                                "name": tc_delta.get("function", {}).get("name", ""),
                                "arguments": "",
                            },
                        }
                    tc = current_tool_calls[idx]
                    if tc_delta.get("id"):
                        tc["id"] = tc_delta["id"]
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        tc["function"]["name"] = fn["name"]
                    if "arguments" in fn:
                        tc["function"]["arguments"] += fn["arguments"]

            if finish_reason == "tool_calls" and current_tool_calls:
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=tc["id"],
                            tool_name=tc["function"]["name"],
                            tool_args=json.loads(tc["function"]["arguments"]),
                        )
                        for tc in current_tool_calls.values()
                    ]
                }
                current_tool_calls.clear()

            # Surface non-standard finish reasons via native
            if finish_reason and finish_reason not in ("stop", "tool_calls"):
                yield {"native": {"finish_reason": finish_reason}}

            if chunk.get("usage"):
                usage = chunk["usage"]
                yield {"native": {"usage": usage}}
                _LOGGER.debug(
                    "Token usage: prompt=%s, completion=%s, total=%s",
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                )

    async def get_models(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        api_key: str | None,
    ) -> list[str]:
        """Fetch available models from OpenAI-compatible /v1/models endpoint."""
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = f"{api_url.rstrip('/')}/models"
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return sorted(model["id"] for model in data.get("data", []))
