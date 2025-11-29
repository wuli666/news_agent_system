"""
Team subgraphs for the news collection system.
"""
from .collection_team import create_collection_team
from .analysis_team import create_analysis_team
from .summary_team import create_summary_team

__all__ = [
    "create_collection_team",
    "create_analysis_team",
    "create_summary_team",
]
