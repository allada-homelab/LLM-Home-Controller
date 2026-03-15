"""Constants for the LLM Home Controller integration."""

DOMAIN = "llm_home_controller"

# Config entry data keys (parent entry — API connection)
CONF_API_URL = "api_url"
CONF_API_KEY = "api_key"
CONF_API_TYPE = "api_type"

# API type constants
API_TYPE_OPENAI = "openai"
API_TYPE_OPENAI_RESPONSES = "openai_responses"
API_TYPE_ANTHROPIC = "anthropic"
ANTHROPIC_API_VERSION = "2023-06-01"

# Subentry data keys (per-agent config)
CONF_MODEL = "model"
CONF_PROMPT = "prompt"
CONF_TEMPERATURE = "temperature"
CONF_MAX_TOKENS = "max_tokens"
CONF_TOP_P = "top_p"
CONF_MAX_CONTEXT_TOKENS = "max_context_tokens"
CONF_EXTENDED_THINKING = "extended_thinking"
CONF_THINKING_BUDGET = "thinking_budget"

# Parent entry options
CONF_CUSTOM_HEADERS = "custom_headers"

# Extra payload options (per-agent)
CONF_SEED = "seed"
CONF_STOP_SEQUENCES = "stop_sequences"
CONF_RESPONSE_FORMAT = "response_format"
CONF_JSON_SCHEMA = "json_schema"

# Retry / fallback
CONF_MAX_RETRIES = "max_retries"
DEFAULT_MAX_RETRIES = 3
CONF_FALLBACK_MODEL = "fallback_model"

# Voice mode
CONF_VOICE_MODE = "voice_mode"
VOICE_MODE_SUFFIX = (
    "\n\nThis conversation is happening via voice. "
    "Keep responses concise and under 2 sentences. "
    "Optimize for spoken output — avoid markdown, lists, or code blocks."
)

# Conversation memory
CONF_MEMORY_ENABLED = "memory_enabled"
CONF_MEMORY_MAX_MESSAGES = "memory_max_messages"
DEFAULT_MEMORY_MAX_MESSAGES = 20

# Custom tools
CONF_CUSTOM_TOOLS = "custom_tools"

# Entity context template (replaces HA's default YAML entity listing)
CONF_ENTITY_CONTEXT_TEMPLATE = "entity_context_template"

# Extra model parameters (raw JSON dict override)
CONF_EXTRA_MODEL_PARAMS = "extra_model_params"

# Defaults
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_PROMPT = "You are a helpful smart home assistant for Home Assistant."
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_CONTEXT_TOKENS = 0
DEFAULT_THINKING_BUDGET = 10000

# Limits
MAX_TOOL_ITERATIONS = 10
