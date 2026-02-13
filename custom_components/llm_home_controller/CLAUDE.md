# LLM Home Controller — Integration

## Domain
`llm_home_controller`

## Config Subentry Pattern
- **Parent entry**: API URL + API key → stored in `entry.data`
- **Child subentries**: Per-agent config (model, system prompt, temperature, max_tokens, top_p, LLM API) → stored in `subentry.data`
- No separate OptionsFlow — all settings modified via subentry reconfigure flow (`async_step_reconfigure`)
- All settings accessed via `self.subentry.data` directly (no `| self.subentry.options` merge)

## Entity Class Hierarchy
```
LLMHomeControllerConversationEntity(
    conversation.ConversationEntity,          # Base (extends RestoreEntity)
    conversation.AbstractConversationAgent,   # Required for agent manager registration
    LLMHomeControllerBaseLLMEntity,           # Our base with _async_handle_chat_log()
)
```

## Key Method Chain
```
_async_handle_message(user_input, chat_log)
  → chat_log.async_provide_llm_data(llm_context, llm_api, prompt, extra_prompt)
  → self._async_handle_chat_log(chat_log)
  → conversation.async_get_result_from_chat_log(user_input, chat_log)
```

## Tool Calling Loop (in `_async_handle_chat_log`)
```python
for _iteration in range(MAX_TOOL_ITERATIONS):  # max 10
    # Build messages from chat_log.content
    # POST /v1/chat/completions with stream=True
    # Transform SSE → AssistantContentDeltaDict via _transform_stream()
    await chat_log.async_add_delta_content_stream(agent_id, stream)
    # chat_log auto-executes tools, adds ToolResultContent
    if not chat_log.unresponded_tool_results:
        break  # No more tools to process
```

## Streaming
- `_attr_supports_streaming = True`
- SSE response chunks → `_transform_stream()` → `AsyncGenerator[AssistantContentDeltaDict]`
- Delta dict fields: `role`, `content`, `tool_calls`, `native`
- `chat_log.async_add_delta_content_stream()` handles accumulation + tool execution

## Agent Registration
```python
async def async_added_to_hass(self):
    await super().async_added_to_hass()
    conversation.async_set_agent(self.hass, self.entry, self)  # entry, not entry_id

async def async_will_remove_from_hass(self):
    conversation.async_unset_agent(self.hass, self.entry)
    await super().async_will_remove_from_hass()
```

## Content Type Reference
- `SystemContent(role="system", content=str)`
- `UserContent(role="user", content=str, attachments=list[Attachment]|None)`
- `AssistantContent(role="assistant", agent_id=str, content=str|None, thinking_content=str|None, tool_calls=list[ToolInput]|None, native=Any)`
- `ToolResultContent(role="tool_result", agent_id=str, tool_call_id=str, tool_name=str, tool_result=JsonObjectType)`
