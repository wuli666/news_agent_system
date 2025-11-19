"""
Graph state types for the news multi-agent system.
"""
from typing import TypedDict, List, Optional, Any
from datetime import datetime


class NewsItem(TypedDict):
    """Individual news item structure."""
    title: str
    content: str
    source: str
    timestamp: str
    category: str  # "text" or "video"
    url: Optional[str]


class State(TypedDict):
    """
    Main state for the news multi-agent system.
    This state is passed between all agents in the graph.
    """
    # Input parameters
    task: str  # Task description, e.g., "获取今日科技新闻"
    date: str  # Target date for news collection

    # Iteration control
    iteration: int  # Current iteration count
    max_iterations: int  # Maximum allowed iterations

    # Collected news data
    text_news: List[NewsItem]  # Text news collected by text agent
    video_news: List[NewsItem]  # Video news collected by video agent

    # Agent analysis results
    text_analysis: Optional[str]  # Text agent's analysis
    video_analysis: Optional[str]  # Video agent's analysis

    # Cached tools (loaded once by coordinator to avoid re-initialization)
    text_tools: Optional[List[Any]]  # Preloaded text news tools
    video_tools: Optional[List[Any]]  # Preloaded video news tools

    # Supervisor control
    supervisor_decision: str  # "continue" or "summarize"
    supervisor_feedback: str  # Feedback on current state
    quality_score: float  # Overall quality score (0-1)

    # Final output
    final_report: Optional[str]  # Final news report from summary agent

    # Metadata
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
