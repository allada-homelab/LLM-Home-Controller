"""Base LLM entity for LLM Home Controller."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.entity import Entity
from voluptuous_openapi import convert

from .const import (
    CONF_CUSTOM_TOOLS,
    CONF_EXTENDED_THINKING,
    CONF_EXTRA_MODEL_PARAMS,
    CONF_FALLBACK_MODEL,
    CONF_FALLBACK_MODEL_2,
    CONF_JSON_SCHEMA,
    CONF_MAX_CONTEXT_TOKENS,
    CONF_MAX_RETRIES,
    CONF_MAX_TOKENS,
    CONF_MODEL,
    CONF_RESPONSE_FORMAT,
    CONF_SEED,
    CONF_STOP_SEQUENCES,
    CONF_TEMPERATURE,
    CONF_THINKING_BUDGET,
    CONF_TOP_P,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_BUDGET,
    DEFAULT_TOP_P,
    DOMAIN,
    MAX_TOOL_ITERATIONS,
)
from .providers import LLMProvider
from .sensor import SIGNAL_USAGE

# Type alias for JSON-like objects
type JsonObjectType = dict[str, Any]


class CustomServiceTool(llm.Tool):
    """User-defined tool that executes a Home Assistant service call."""

    def __init__(
        self,
        hass: HomeAssistant,
        name: str,
        description: str,
        parameters_schema: dict[str, Any] | None,
        service: str,
        service_data_template: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the custom tool."""
        self.name = name
        self.description = description or ""
        self.parameters = vol.Schema({}, extra=vol.ALLOW_EXTRA)
        if parameters_schema and "properties" in parameters_schema:
            schema_dict: dict[Any, Any] = {}
            required = set(parameters_schema.get("required", []))
            for prop_name, prop_def in parameters_schema["properties"].items():
                key = vol.Required(prop_name) if prop_name in required else vol.Optional(prop_name)
                prop_type = prop_def.get("type", "string")
                if prop_type == "number":
                    schema_dict[key] = vol.Coerce(float)
                elif prop_type == "integer":
                    schema_dict[key] = vol.Coerce(int)
                elif prop_type == "boolean":
                    schema_dict[key] = bool
                else:
                    schema_dict[key] = str
            self.parameters = vol.Schema(schema_dict)
        self._hass = hass
        self._service = service
        self._service_data_template = service_data_template or {}

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Execute the HA service call."""
        parts = self._service.split(".", 1)
        if len(parts) != 2:
            return {"error": f"Invalid service format: {self._service}"}
        domain, service_name = parts
        service_data = {**self._service_data_template, **tool_input.tool_args}
        try:
            result = await hass.services.async_call(
                domain,
                service_name,
                service_data,
                blocking=True,
                return_response=True,
            )
            return {"success": True, "result": result or {}}
        except Exception as err:
            _LOGGER.warning("Custom tool '%s' service call failed: %s", self.name, err)
            return {"error": str(err)}


def _parse_custom_tools(
    raw_json: str,
    hass: HomeAssistant,
) -> list[llm.Tool]:
    """Parse custom tool definitions from JSON config string."""
    try:
        tool_defs = json.loads(raw_json)
    except json.JSONDecodeError:
        _LOGGER.warning("Invalid custom tools JSON, ignoring")
        return []

    if not isinstance(tool_defs, list):
        _LOGGER.warning("Custom tools must be a JSON array, ignoring")
        return []

    tools: list[llm.Tool] = []
    for tool_def in tool_defs:
        if not isinstance(tool_def, dict):
            continue
        name = tool_def.get("name")
        service = tool_def.get("service")
        if not name or not service:
            _LOGGER.warning("Custom tool missing 'name' or 'service', skipping")
            continue
        tools.append(
            CustomServiceTool(
                hass=hass,
                name=name,
                description=tool_def.get("description", ""),
                parameters_schema=tool_def.get("parameters"),
                service=service,
                service_data_template=tool_def.get("service_data"),
            )
        )
    return tools


_LOGGER = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (~4 chars per token)."""
    return len(text) // 4 + 1


def _prune_messages(
    messages: list[dict[str, Any]],
    system: str | None,
    max_context_tokens: int,
) -> list[dict[str, Any]]:
    """Prune oldest messages to fit within the token budget.

    Keeps the system prompt (separate, not in messages list) and always
    preserves the most recent message. Removes from the front (oldest).
    """
    if not messages:
        return messages

    system_tokens = _estimate_tokens(system) if system else 0
    budget = max_context_tokens - system_tokens

    if budget <= 0:
        return messages[-1:]

    msg_tokens = [_estimate_tokens(json.dumps(m)) for m in messages]
    total = sum(msg_tokens)

    if total <= budget:
        return messages

    # Remove oldest messages until under budget, always keep the last one
    start = 0
    while start < len(messages) - 1 and total > budget:
        total -= msg_tokens[start]
        start += 1

    return messages[start:]


async def _usage_capturing_stream(
    stream: AsyncGenerator,
    usage_totals: list[int],
) -> AsyncGenerator:
    """Wrap a stream to capture usage data from native delta dicts."""
    async for delta in stream:
        if "native" in delta:
            usage = delta["native"].get("usage")
            if usage:
                usage_totals[0] += usage.get("prompt_tokens", 0) + usage.get("input_tokens", 0)
                usage_totals[1] += usage.get("completion_tokens", 0) + usage.get("output_tokens", 0)
        yield delta


class LLMHomeControllerBaseLLMEntity(Entity):
    """Base entity for LLM Home Controller."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self,
        entry: ConfigSubentry,
        subentry: ConfigSubentry,
        session: aiohttp.ClientSession,
        api_url: str,
        api_key: str | None,
        provider: LLMProvider,
        custom_headers_raw: str | None = None,
    ) -> None:
        """Initialize the base LLM entity."""
        self.entry_session = session
        self._api_url = api_url
        self._api_key = api_key
        self._provider = provider
        self._custom_headers_raw = custom_headers_raw
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="LLM Home Controller",
            model=subentry.data.get(CONF_MODEL, "unknown"),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    async def _async_post_with_retry(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        max_retries: int,
    ) -> aiohttp.ClientResponse:
        """Post to the LLM API with retry and exponential backoff.

        Retries on: HTTP 429, 500-599, TimeoutError, aiohttp.ClientError.
        No retry on: HTTP 400, 401, 404 (raises immediately).
        """
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.entry_session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=300),
                )

                if response.status == 401:
                    raise HomeAssistantError("Authentication failed with the LLM API")

                if response.status in (400, 404):
                    body = await response.text()
                    raise HomeAssistantError(f"LLM API error (HTTP {response.status}): {body[:500]}")

                if response.status == 429 or response.status >= 500:
                    body = await response.text()
                    last_error = HomeAssistantError(f"LLM API error (HTTP {response.status}): {body[:500]}")
                    if attempt < max_retries:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                delay = min(1.0 * 2**attempt, 30.0)
                        else:
                            delay = min(1.0 * 2**attempt, 30.0)
                        _LOGGER.warning(
                            "LLM API returned %s, retrying in %.1fs (attempt %d/%d)",
                            response.status,
                            delay,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise last_error

                if response.status >= 400:
                    body = await response.text()
                    raise HomeAssistantError(f"LLM API error (HTTP {response.status}): {body[:500]}")

                return response

            except TimeoutError as err:
                last_error = HomeAssistantError("Timeout waiting for LLM API response")
                last_error.__cause__ = err
                if attempt < max_retries:
                    delay = min(1.0 * 2**attempt, 30.0)
                    _LOGGER.warning(
                        "LLM API timeout, retrying in %.1fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue

            except aiohttp.ClientError as err:
                last_error = HomeAssistantError(f"Error communicating with LLM API: {err}")
                last_error.__cause__ = err
                if attempt < max_retries:
                    delay = min(1.0 * 2**attempt, 30.0)
                    _LOGGER.warning(
                        "LLM API connection error, retrying in %.1fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)
                    continue

        raise last_error  # type: ignore[misc]

    async def _async_post_with_fallback(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        max_retries: int,
        model_chain: list[str],
        start_idx: int,
    ) -> tuple[aiohttp.ClientResponse, int]:
        """Post, walking the model chain from start_idx until one succeeds.

        The model at start_idx (the currently committed model) gets the full
        retry budget; each subsequent model in the chain is probed with no
        retries so we fail over quickly. Returns the response and the index of
        the model that succeeded. Authentication failures never fail over.
        """
        first_error: HomeAssistantError | None = None

        for idx in range(start_idx, len(model_chain)):
            payload["model"] = model_chain[idx]
            retries = max_retries if idx == start_idx else 0
            try:
                response = await self._async_post_with_retry(url, payload, headers, retries)
            except HomeAssistantError as err:
                if "Authentication failed" in str(err):
                    raise
                if first_error is None:
                    first_error = err
                if idx + 1 < len(model_chain):
                    _LOGGER.warning(
                        "Model '%s' failed, trying fallback '%s': %s",
                        model_chain[idx],
                        model_chain[idx + 1],
                        err,
                    )
                continue
            return response, idx

        raise first_error  # type: ignore[misc]

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        *,
        structure: vol.Schema | None = None,
    ) -> None:
        """Generate an answer for the chat log.

        `structure` is an optional per-call output schema (e.g. an AI Task's
        `structure`). When provided it constrains the model to emit matching
        JSON and overrides the statically configured response format.
        """
        options = self.subentry.data
        model = options.get(CONF_MODEL, "")
        temperature = options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        max_tokens = options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = options.get(CONF_TOP_P, DEFAULT_TOP_P)
        max_retries = int(options.get(CONF_MAX_RETRIES, DEFAULT_MAX_RETRIES))
        # Primary first, then any configured fallbacks tried in order.
        model_chain = [model]
        model_chain += [m for m in (options.get(CONF_FALLBACK_MODEL), options.get(CONF_FALLBACK_MODEL_2)) if m]

        tools: list[dict[str, Any]] = []
        custom_serializer = None
        if chat_log.llm_api:
            custom_serializer = chat_log.llm_api.custom_serializer
            tools = self._provider.format_tools(
                chat_log.llm_api.tools,
                custom_serializer=custom_serializer,
            )

        # Parse and inject custom tool definitions. Custom tools are executed
        # through chat_log.llm_api.async_call_tool, so without an LLM API a
        # tool call would raise "No LLM API configured" mid-turn. Only advertise
        # them when an LLM API is present (matches entity/memory tool gating).
        if custom_tools_raw := options.get(CONF_CUSTOM_TOOLS):
            if chat_log.llm_api:
                custom_tools = _parse_custom_tools(custom_tools_raw, self.hass)
                if custom_tools:
                    tools.extend(self._provider.format_tools(custom_tools, custom_serializer=custom_serializer))
                    # Inject into llm_api so async_call_tool can find them
                    chat_log.llm_api.tools.extend(custom_tools)
            else:
                _LOGGER.warning(
                    "Custom tools are configured but no LLM API is selected; skipping them "
                    "(they cannot be executed without an LLM API)"
                )

        # Build extra options for provider-specific features
        extra_options: dict[str, Any] = {}
        if options.get(CONF_EXTENDED_THINKING):
            extra_options["extended_thinking"] = True
            extra_options["thinking_budget"] = options.get(CONF_THINKING_BUDGET, DEFAULT_THINKING_BUDGET)
        if (seed := options.get(CONF_SEED)) is not None:
            extra_options["seed"] = int(seed)
        if stop_raw := options.get(CONF_STOP_SEQUENCES):
            stop_list = [s.strip() for s in stop_raw.split(",") if s.strip()]
            if stop_list:
                extra_options["stop_sequences"] = stop_list
        if (response_fmt := options.get(CONF_RESPONSE_FORMAT)) and response_fmt != "text":
            extra_options["response_format"] = response_fmt
            if response_fmt == "json_schema" and (schema_raw := options.get(CONF_JSON_SCHEMA)):
                try:
                    extra_options["json_schema"] = json.loads(schema_raw)
                except json.JSONDecodeError:
                    _LOGGER.warning("Invalid JSON schema, ignoring")

        # A per-call output schema (e.g. AI Task `structure`) takes precedence
        # over the statically configured response format.
        if structure is not None:
            extra_options["response_format"] = "json_schema"
            extra_options["json_schema"] = convert(structure, custom_serializer=custom_serializer)

        headers = self._provider.build_headers(self._api_key)
        if self._custom_headers_raw:
            try:
                custom = json.loads(self._custom_headers_raw)
                if not isinstance(custom, dict):
                    _LOGGER.warning("Custom headers must be a JSON object, ignoring")
                else:
                    overridden = set(custom) & set(headers)
                    if overridden:
                        _LOGGER.debug(
                            "Custom headers overriding default keys: %s",
                            ", ".join(sorted(overridden)),
                        )
                    headers.update(custom)
            except (json.JSONDecodeError, TypeError):
                _LOGGER.warning("Invalid custom headers JSON, ignoring")
        url = self._provider.build_url(self._api_url)

        # Trace initial request details
        conversation.async_conversation_trace_append(
            conversation.ConversationTraceEventType.AGENT_DETAIL,
            {
                "tools": tools or None,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            },
        )

        max_context_tokens = options.get(CONF_MAX_CONTEXT_TOKENS, DEFAULT_MAX_CONTEXT_TOKENS)
        usage_totals = [0, 0]  # [input_tokens, output_tokens] — mutable for closure

        iteration = 0
        # Sticky across tool iterations: once we fail over to a fallback within
        # this message, keep using it instead of re-probing the dead model.
        current_model_idx = 0
        for iteration in range(MAX_TOOL_ITERATIONS):  # noqa: B007 — used after loop
            # Re-convert full chat log each iteration
            # (required for Anthropic's strict role alternation)
            converted = self._provider.convert_content(chat_log.content)
            messages = converted["messages"]
            system = converted["system"]

            if max_context_tokens > 0:
                messages = _prune_messages(messages, system, max_context_tokens)

            payload = self._provider.build_payload(
                model=model,
                messages=messages,
                system=system,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_options=extra_options or None,
            )

            # Apply extra model params — overrides payload keys like
            # temperature, top_p, top_k, repetition_penalty, etc.
            if extra_params_raw := options.get(CONF_EXTRA_MODEL_PARAMS):
                try:
                    extra_params = json.loads(extra_params_raw)
                    if isinstance(extra_params, dict):
                        payload.update(extra_params)
                    else:
                        _LOGGER.warning("Extra model params must be a JSON object, ignoring")
                except json.JSONDecodeError:
                    _LOGGER.warning("Invalid extra model params JSON, ignoring")

            response, current_model_idx = await self._async_post_with_fallback(
                url, payload, headers, max_retries, model_chain, current_model_idx
            )

            try:
                async for _content in chat_log.async_add_delta_content_stream(
                    self.entity_id,
                    _usage_capturing_stream(
                        self._provider.transform_stream(response),
                        usage_totals,
                    ),
                ):
                    pass
            finally:
                response.release()

            if not chat_log.unresponded_tool_results:
                break

        # Fire usage signal for sensor tracking
        if (usage_totals[0] or usage_totals[1]) and self.hass:
            async_dispatcher_send(
                self.hass,
                f"{SIGNAL_USAGE}_{self.subentry.subentry_id}",
                usage_totals[0],
                usage_totals[1],
            )

        # Trace completion summary
        conversation.async_conversation_trace_append(
            conversation.ConversationTraceEventType.AGENT_DETAIL,
            {"stats": {"iterations": iteration + 1, "model": model_chain[current_model_idx]}},
        )
