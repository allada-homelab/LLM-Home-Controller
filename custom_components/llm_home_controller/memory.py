"""Persistent memory store for LLM Home Controller agents."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.storage import Store

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1

# Type alias for JSON-like objects
type JsonObjectType = dict[str, Any]


class AgentMemoryStore:
    """Persistent memory store for a single agent (subentry).

    Stores user preferences and instructions as a list of entries
    in a JSON file under .storage/
    """

    def __init__(self, hass: HomeAssistant, subentry_id: str) -> None:
        """Initialize the memory store."""
        self._store: Store[dict[str, Any]] = Store(
            hass,
            STORAGE_VERSION,
            f"{DOMAIN}.memory.{subentry_id}",
        )
        self._memories: list[dict[str, str]] = []
        self._loaded = False

    async def async_load(self) -> None:
        """Load memories from disk."""
        data = await self._store.async_load()
        if data and "memories" in data:
            self._memories = data["memories"]
        self._loaded = True

    async def _async_save(self) -> None:
        """Save memories to disk."""
        await self._store.async_save({"memories": self._memories})

    @property
    def memories(self) -> list[dict[str, str]]:
        """Return all memories."""
        return list(self._memories)

    async def async_add(self, content: str) -> str:
        """Add a new memory. Returns the memory ID."""
        memory_id = uuid.uuid4().hex[:8]
        self._memories.append({"id": memory_id, "content": content})
        await self._async_save()
        return memory_id

    async def async_update(self, memory_id: str, content: str) -> bool:
        """Update an existing memory by ID. Returns True if found."""
        for mem in self._memories:
            if mem["id"] == memory_id:
                mem["content"] = content
                await self._async_save()
                return True
        return False

    async def async_remove(self, memory_id: str) -> bool:
        """Remove a memory by ID. Returns True if found."""
        before = len(self._memories)
        self._memories = [m for m in self._memories if m["id"] != memory_id]
        if len(self._memories) < before:
            await self._async_save()
            return True
        return False

    async def async_clear(self) -> int:
        """Remove all memories. Returns count removed."""
        count = len(self._memories)
        self._memories = []
        await self._async_save()
        return count

    def format_for_prompt(self) -> str | None:
        """Format memories as a prompt section. Returns None if empty."""
        if not self._memories:
            return None
        lines = [f"- {m['content']} (id: {m['id']})" for m in self._memories]
        return "# User Preferences & Instructions\n" + "\n".join(lines)


# --- LLM Tools for memory management ---


class SaveMemoryTool(llm.Tool):
    """Tool that saves a user preference or instruction to persistent memory."""

    name = "SaveMemory"
    description = (
        "Save a user preference or instruction to memory. "
        "Use when the user asks you to remember something. "
        "Returns the memory ID for future reference."
    )
    parameters = vol.Schema(
        {
            vol.Required("content"): str,
        }
    )

    def __init__(self, store: AgentMemoryStore) -> None:
        """Initialize."""
        self._store = store

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Save a memory."""
        content = tool_input.tool_args["content"]
        memory_id = await self._store.async_add(content)
        return {"success": True, "id": memory_id}


class UpdateMemoryTool(llm.Tool):
    """Tool that updates an existing memory by ID."""

    name = "UpdateMemory"
    description = "Update an existing memory entry. Use when the user wants to change a previously saved preference."
    parameters = vol.Schema(
        {
            vol.Required("id"): str,
            vol.Required("content"): str,
        }
    )

    def __init__(self, store: AgentMemoryStore) -> None:
        """Initialize."""
        self._store = store

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Update a memory."""
        found = await self._store.async_update(
            tool_input.tool_args["id"],
            tool_input.tool_args["content"],
        )
        if found:
            return {"success": True}
        return {"success": False, "error": "Memory not found"}


class RemoveMemoryTool(llm.Tool):
    """Tool that removes a memory by ID."""

    name = "RemoveMemory"
    description = "Remove a memory entry. Use when the user wants to forget a previously saved preference."
    parameters = vol.Schema(
        {
            vol.Required("id"): str,
        }
    )

    def __init__(self, store: AgentMemoryStore) -> None:
        """Initialize."""
        self._store = store

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Remove a memory."""
        found = await self._store.async_remove(tool_input.tool_args["id"])
        if found:
            return {"success": True}
        return {"success": False, "error": "Memory not found"}


def get_memory_tools(store: AgentMemoryStore) -> list[llm.Tool]:
    """Return the set of memory management tools for an agent."""
    return [
        SaveMemoryTool(store),
        UpdateMemoryTool(store),
        RemoveMemoryTool(store),
    ]
