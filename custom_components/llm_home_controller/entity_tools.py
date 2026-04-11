"""Entity query tools for LLM Home Controller agents."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)
from homeassistant.helpers import (
    llm,
)

# Type alias for JSON-like objects
type JsonObjectType = dict[str, Any]


class GetEntityDetailsTool(llm.Tool):
    """Tool that searches Home Assistant entities and returns detailed JSON.

    Allows the agent to query entities by ID, name, domain, area, or state
    and get back full details including attributes and timestamps.
    """

    name = "GetEntityDetails"
    description = (
        "Search Home Assistant entities and return detailed information as JSON. "
        "Use this to look up specific entities or find entities matching criteria. "
        "Accepts optional filters that are combined with AND logic. "
        "At least one filter must be provided. "
        "Returns a JSON object with 'count' and 'entities' array. "
        "Each entity contains: entity_id, state, name, domain, area, "
        "attributes (dict with keys like brightness, color_temp, "
        "unit_of_measurement, device_class, etc.), last_changed (ISO timestamp), "
        "and last_updated (ISO timestamp)."
    )
    parameters = vol.Schema(
        {
            vol.Optional("entity_ids", description="Exact entity IDs to look up"): vol.All([str], vol.Length(min=1)),
            vol.Optional("names", description="Substring search on friendly name (case-insensitive)"): vol.All(
                [str], vol.Length(min=1)
            ),
            vol.Optional(
                "domains",
                description="Filter by domain (e.g. light, switch, sensor, climate)",
            ): vol.All([str], vol.Length(min=1)),
            vol.Optional("areas", description="Filter by area name (case-insensitive substring match)"): vol.All(
                [str], vol.Length(min=1)
            ),
            vol.Optional(
                "states",
                description="Filter by current state value (e.g. on, off, home, 23.5)",
            ): vol.All([str], vol.Length(min=1)),
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Search entities and return detailed info."""
        args = tool_input.tool_args
        entity_ids = args.get("entity_ids")
        names = args.get("names")
        domains = args.get("domains")
        areas = args.get("areas")
        states = args.get("states")

        if not any([entity_ids, names, domains, areas, states]):
            return {"error": "At least one filter parameter must be provided."}

        area_reg = ar.async_get(hass)
        ent_reg = er.async_get(hass)
        dev_reg = dr.async_get(hass)

        def _get_area_name(entity_id: str) -> str | None:
            """Resolve area name for an entity (entity area → device area)."""
            entry = ent_reg.async_get(entity_id)
            area_id = None
            if entry:
                area_id = entry.area_id
                if not area_id and entry.device_id:
                    device = dev_reg.async_get(entry.device_id)
                    if device:
                        area_id = device.area_id
            if area_id:
                area_entry = area_reg.async_get_area(area_id)
                if area_entry:
                    return area_entry.name
            return None

        results: list[dict[str, Any]] = []

        # If filtering by entity_ids only, skip iterating all states
        if entity_ids and not any([names, domains, areas, states]):
            candidate_states = [s for eid in entity_ids if (s := hass.states.get(eid)) is not None]
        elif domains and not any([entity_ids, names, areas, states]):
            # Optimize: async_all accepts domain filter
            candidate_states = []
            for domain in domains:
                candidate_states.extend(hass.states.async_all(domain))
        else:
            candidate_states = hass.states.async_all()

        for state in candidate_states:
            # Apply filters (AND logic — all provided filters must match)
            if entity_ids and state.entity_id not in entity_ids:
                continue
            if domains and state.domain not in domains:
                continue
            if states and state.state not in states:
                continue
            if names:
                name_lower = state.name.lower()
                if not any(n.lower() in name_lower for n in names):
                    continue

            area_name: str | None = None
            if areas:
                area_name = _get_area_name(state.entity_id)
                if not area_name or not any(a.lower() in area_name.lower() for a in areas):
                    continue
            else:
                area_name = _get_area_name(state.entity_id)

            results.append(
                {
                    "entity_id": state.entity_id,
                    "state": state.state,
                    "name": state.name,
                    "domain": state.domain,
                    "area": area_name or "",
                    "attributes": dict(state.attributes),
                    "last_changed": state.last_changed.isoformat(),
                    "last_updated": state.last_updated.isoformat(),
                }
            )

        return {"count": len(results), "entities": results}


def get_entity_tools() -> list[llm.Tool]:
    """Return the set of entity query tools."""
    return [GetEntityDetailsTool()]
