"""Tests for persistent memory store and tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from custom_components.llm_home_controller.memory import (
    AgentMemoryStore,
    RemoveMemoryTool,
    SaveMemoryTool,
    UpdateMemoryTool,
    get_memory_tools,
)


@pytest.fixture
def mock_store():
    """Create a memory store with a mocked HA Store backend."""
    hass = MagicMock(spec=HomeAssistant)
    with patch("custom_components.llm_home_controller.memory.Store") as mock_store_cls:
        mock_ha_store = MagicMock()
        mock_ha_store.async_load = AsyncMock(return_value=None)
        mock_ha_store.async_save = AsyncMock()
        mock_store_cls.return_value = mock_ha_store
        store = AgentMemoryStore(hass, "test-subentry-id")
    return store, mock_ha_store


@pytest.mark.asyncio
async def test_load_empty(mock_store):
    """Test loading when no data exists."""
    store, _ = mock_store
    await store.async_load()
    assert store.memories == []


@pytest.mark.asyncio
async def test_load_existing_data(mock_store):
    """Test loading existing memories from storage."""
    store, mock_ha_store = mock_store
    mock_ha_store.async_load = AsyncMock(
        return_value={
            "memories": [
                {"id": "abc123", "content": "Exclude closet light from all lights"},
            ]
        }
    )
    await store.async_load()
    assert len(store.memories) == 1
    assert store.memories[0]["content"] == "Exclude closet light from all lights"


@pytest.mark.asyncio
async def test_add_memory(mock_store):
    """Test adding a memory."""
    store, mock_ha_store = mock_store
    await store.async_load()

    memory_id = await store.async_add("Turn off porch light at sunset")
    assert len(memory_id) == 8
    assert len(store.memories) == 1
    assert store.memories[0]["content"] == "Turn off porch light at sunset"
    mock_ha_store.async_save.assert_called()


@pytest.mark.asyncio
async def test_update_memory(mock_store):
    """Test updating an existing memory."""
    store, _ = mock_store
    await store.async_load()

    memory_id = await store.async_add("Old preference")
    result = await store.async_update(memory_id, "Updated preference")
    assert result is True
    assert store.memories[0]["content"] == "Updated preference"


@pytest.mark.asyncio
async def test_update_nonexistent(mock_store):
    """Test updating a memory that doesn't exist."""
    store, _ = mock_store
    await store.async_load()

    result = await store.async_update("nonexistent", "content")
    assert result is False


@pytest.mark.asyncio
async def test_remove_memory(mock_store):
    """Test removing a memory."""
    store, _ = mock_store
    await store.async_load()

    memory_id = await store.async_add("To be removed")
    result = await store.async_remove(memory_id)
    assert result is True
    assert len(store.memories) == 0


@pytest.mark.asyncio
async def test_remove_nonexistent(mock_store):
    """Test removing a memory that doesn't exist."""
    store, _ = mock_store
    await store.async_load()

    result = await store.async_remove("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_clear_all(mock_store):
    """Test clearing all memories."""
    store, _ = mock_store
    await store.async_load()

    await store.async_add("Memory 1")
    await store.async_add("Memory 2")
    count = await store.async_clear()
    assert count == 2
    assert len(store.memories) == 0


def test_format_for_prompt_empty(mock_store):
    """Test that empty memories returns None."""
    store, _ = mock_store
    assert store.format_for_prompt() is None


@pytest.mark.asyncio
async def test_format_for_prompt(mock_store):
    """Test formatting memories for prompt injection."""
    store, _ = mock_store
    await store.async_load()

    await store.async_add("Exclude closet light")
    await store.async_add("Living room default is 50% brightness")

    prompt = store.format_for_prompt()
    assert prompt is not None
    assert "# User Preferences & Instructions" in prompt
    assert "Exclude closet light" in prompt
    assert "Living room default is 50% brightness" in prompt


# --- Tool tests ---


@pytest.mark.asyncio
async def test_save_memory_tool(mock_store):
    """Test SaveMemoryTool calls the store."""
    store, _ = mock_store
    await store.async_load()
    tool = SaveMemoryTool(store)
    hass = MagicMock(spec=HomeAssistant)
    tool_input = MagicMock(spec=llm.ToolInput)
    tool_input.tool_args = {"content": "Test preference"}
    llm_context = MagicMock(spec=llm.LLMContext)

    result = await tool.async_call(hass, tool_input, llm_context)
    assert result["success"] is True
    assert "id" in result
    assert len(store.memories) == 1


@pytest.mark.asyncio
async def test_update_memory_tool(mock_store):
    """Test UpdateMemoryTool calls the store."""
    store, _ = mock_store
    await store.async_load()
    memory_id = await store.async_add("Old")
    tool = UpdateMemoryTool(store)
    hass = MagicMock(spec=HomeAssistant)
    tool_input = MagicMock(spec=llm.ToolInput)
    tool_input.tool_args = {"id": memory_id, "content": "New"}
    llm_context = MagicMock(spec=llm.LLMContext)

    result = await tool.async_call(hass, tool_input, llm_context)
    assert result["success"] is True
    assert store.memories[0]["content"] == "New"


@pytest.mark.asyncio
async def test_update_memory_tool_not_found(mock_store):
    """Test UpdateMemoryTool with nonexistent ID."""
    store, _ = mock_store
    await store.async_load()
    tool = UpdateMemoryTool(store)
    hass = MagicMock(spec=HomeAssistant)
    tool_input = MagicMock(spec=llm.ToolInput)
    tool_input.tool_args = {"id": "nope", "content": "New"}
    llm_context = MagicMock(spec=llm.LLMContext)

    result = await tool.async_call(hass, tool_input, llm_context)
    assert result["success"] is False


@pytest.mark.asyncio
async def test_remove_memory_tool(mock_store):
    """Test RemoveMemoryTool calls the store."""
    store, _ = mock_store
    await store.async_load()
    memory_id = await store.async_add("To remove")
    tool = RemoveMemoryTool(store)
    hass = MagicMock(spec=HomeAssistant)
    tool_input = MagicMock(spec=llm.ToolInput)
    tool_input.tool_args = {"id": memory_id}
    llm_context = MagicMock(spec=llm.LLMContext)

    result = await tool.async_call(hass, tool_input, llm_context)
    assert result["success"] is True
    assert len(store.memories) == 0


def test_get_memory_tools(mock_store):
    """Test get_memory_tools returns all three tools."""
    store, _ = mock_store
    tools = get_memory_tools(store)
    assert len(tools) == 3
    names = {t.name for t in tools}
    assert names == {"SaveMemory", "UpdateMemory", "RemoveMemory"}
