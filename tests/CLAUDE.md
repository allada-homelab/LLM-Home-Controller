# LLM Home Controller — Tests

## Framework
- pytest + pytest-homeassistant-custom-component + pytest-asyncio
- Run: `uv run pytest tests/ -v`
- Collect: `uv run pytest tests/ --collect-only`

## Conventions
- All tests async: `async def test_*(hass):`
- Test naming: `test_<module>_<scenario>` (e.g., `test_config_flow_connection_error`)
- Always test error paths alongside happy paths
- One test file per source module

## Mock Patterns

### Mock aiohttp
Use `aiohttp_client_mock` fixture or manual `AsyncMock` on `ClientSession`.

### Mock Config Entries
```python
from tests.common import MockConfigEntry

entry = MockConfigEntry(
    domain=DOMAIN,
    data={"api_url": "http://localhost:8080", "api_key": "test-key"},
    runtime_data=mock_session,
)
```

### Mock Subentries
Create `ConfigSubentry` with test data dicts for per-agent config.

### MockChatLog (from HA core pattern)
```python
@dataclass
class MockChatLog(conversation.ChatLog):
    _mock_tool_results: dict = field(default_factory=dict)

    def mock_tool_results(self, results: dict):
        self._mock_tool_results = results

    @property
    def llm_api(self):
        return self._llm_api

    @llm_api.setter
    def llm_api(self, value):
        self._llm_api = value
        if not value:
            return
        async def async_call_tool(tool_input):
            if tool_input.id not in self._mock_tool_results:
                raise ValueError(f"Tool {tool_input.id} not found")
            return self._mock_tool_results[tool_input.id]
        self._llm_api.async_call_tool = async_call_tool
```

### mock_chat_log fixture
```python
@pytest.fixture
async def mock_chat_log(hass):
    with (
        patch("homeassistant.components.conversation.chat_log.ChatLog", MockChatLog),
        chat_session.async_get_chat_session(hass, "mock-conversation-id") as session,
        conversation.async_get_chat_log(hass, session) as chat_log,
    ):
        yield chat_log
```
