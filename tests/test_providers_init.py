"""Tests for the provider factory."""

from __future__ import annotations

from custom_components.llm_home_controller.providers import get_provider
from custom_components.llm_home_controller.providers.anthropic import AnthropicProvider
from custom_components.llm_home_controller.providers.openai import OpenAIProvider
from custom_components.llm_home_controller.providers.openai_responses import OpenAIResponsesProvider


def test_get_provider_openai() -> None:
    """Test that 'openai' returns an OpenAIProvider."""
    provider = get_provider("openai")
    assert isinstance(provider, OpenAIProvider)


def test_get_provider_anthropic() -> None:
    """Test that 'anthropic' returns an AnthropicProvider."""
    provider = get_provider("anthropic")
    assert isinstance(provider, AnthropicProvider)


def test_get_provider_openai_responses() -> None:
    """Test that 'openai_responses' returns an OpenAIResponsesProvider."""
    provider = get_provider("openai_responses")
    assert isinstance(provider, OpenAIResponsesProvider)


def test_get_provider_default() -> None:
    """Test that an unknown type falls back to OpenAIProvider."""
    provider = get_provider("unknown_type")
    assert isinstance(provider, OpenAIProvider)
