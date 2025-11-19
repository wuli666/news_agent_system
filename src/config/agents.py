"""
Agent configuration mapping.
Maps each agent to its preferred LLM model.
"""

# Map each agent to its LLM type
AGENT_LLM_MAP = {
    "text_agent": "qwen",  # Use Qwen3 for text understanding
    "video_agent": "qwen_vl",  # Use Qwen3-VL for video understanding
    "supervisor": "qwen",  # Use Qwen3 for supervision
    "summary": "qwen",  # Use Qwen3 for summary generation
}

# Agent-specific configurations
AGENT_CONFIG = {
    "text_agent": {
        "name": "Text News Agent",
        "description": "Analyzes text-based news articles",
        "temperature": 0.7,
        "max_tokens": 2000,
    },
    "video_agent": {
        "name": "Video News Agent",
        "description": "Analyzes video news content",
        "temperature": 0.7,
        "max_tokens": 2000,
    },
    "supervisor": {
        "name": "Supervisor Agent",
        "description": "Evaluates quality and controls workflow",
        "temperature": 0.3,  # Lower temperature for more consistent decisions
        "max_tokens": 1000,
    },
    "summary": {
        "name": "Summary Agent",
        "description": "Generates final news reports",
        "temperature": 0.5,
        "max_tokens": 3000,
    },
}
