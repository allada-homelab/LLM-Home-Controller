"""LLM API providers for LLM Home Controller."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol

import aiohttp
from homeassistant.components.conversation import AssistantContentDeltaDict, Content
from homeassistant.helpers import llm


class LLMProvider(Protocol):
    """Protocol for LLM API providers."""

    def format_tools(self, tools: list[llm.Tool], custom_serializer=None) -> list[dict[str, Any]]:
        """Convert HA tools to provider-native format."""
        ...

    def convert_content(
        self,
        chat_content: list[Content],
    ) -> dict[str, Any]:
        """Convert ChatLog content to provider-native request structure.

        Returns a dict with:
        - "messages": list of provider-formatted messages
        - "system": str | None (system prompt, extracted separately for Anthropic)
        """
        ...

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
        """Build the full request payload."""
        ...

    def build_headers(self, api_key: str | None) -> dict[str, str]:
        """Build HTTP headers for the request."""
        ...

    def build_url(self, api_url: str) -> str:
        """Build the full endpoint URL from the base API URL."""
        ...

    async def transform_stream(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncGenerator[AssistantContentDeltaDict]:
        """Transform SSE response into HA delta dicts."""
        ...

    async def get_models(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        api_key: str | None,
    ) -> list[str]:
        """Fetch available model IDs from the API."""
        ...


def get_provider(api_type: str) -> LLMProvider:
    """Return the provider for the given API type."""
    from custom_components.llm_home_controller.const import (
        API_TYPE_ANTHROPIC,
        API_TYPE_OPENAI_RESPONSES,
    )

    if api_type == API_TYPE_ANTHROPIC:
        from .anthropic import AnthropicProvider

        return AnthropicProvider()

    if api_type == API_TYPE_OPENAI_RESPONSES:
        from .openai_responses import OpenAIResponsesProvider

        return OpenAIResponsesProvider()

    from .openai import OpenAIProvider

    return OpenAIProvider()
