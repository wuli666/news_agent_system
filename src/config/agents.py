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
    "research": "qwen",  # Use Qwen3 for research/expansion
    "relationship_graph": "qwen",
    "text_reflect": "qwen",
    "text_sentiment": "qwen",
    "summary_timeline": "qwen",
    "summary_trend": "qwen",
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
    "research": {
        "name": "Research Agent",
        "description": "Expands hot news items with external search",
        "temperature": 0.3,
        "max_tokens": 1800,
    },
    "relationship_graph": {
        "name": "Relationship Graph Agent",
        "description": "Builds textual relationship/causal graph of hotspots",
        "temperature": 0.4,
        "max_tokens": 1200,
    },
    "text_reflect": {
        "name": "Text Reflect Agent",
        "description": "Reflects on supervisor questions and multi-view checks",
        "temperature": 0.5,
        "max_tokens": 1600,
    },
    "text_sentiment": {
        "name": "Text Sentiment Agent",
        "description": "Analyzes sentiment and risk for text news",
        "temperature": 0.4,
        "max_tokens": 1200,
    },
    "summary_timeline": {
        "name": "Summary Timeline Agent",
        "description": "Builds timelines for hotspots",
        "temperature": 0.4,
        "max_tokens": 1600,
    },
    "summary_trend": {
        "name": "Summary Trend Agent",
        "description": "Analyzes trends and next steps",
        "temperature": 0.4,
        "max_tokens": 1600,
    },
}
