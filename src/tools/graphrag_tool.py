"""
GraphRAG Ingest Tool

Integrates with main news collection workflow to ingest news into knowledge graph.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
async def graphrag_ingest(
    news_items: List[Dict[str, Any]],
    top_k: int = 20,
    extract: bool = True
) -> Dict[str, Any]:
    """
    Ingest top news items into GraphRAG knowledge graph.

    This tool is called after news_collector to persist hot news
    into the long-term time-series knowledge graph.

    Args:
        news_items: List of news items to ingest
        top_k: Only ingest top K items by hot_score
        extract: Whether to use LLM extraction (vs Zep auto-extract)

    Returns:
        Result dict with indexed_count and graph_id
    """
    from src.graphrag.config import get_config

    config = get_config()

    # Check if GraphRAG is enabled
    if not config.ENABLED:
        logger.info("GraphRAG is disabled, skipping ingest")
        return {
            "success": True,
            "indexed_count": 0,
            "message": "GraphRAG disabled"
        }

    try:
        # Sort by hot_score and take top K
        sorted_items = sorted(
            news_items,
            key=lambda x: x.get("hot_score", 0),
            reverse=True
        )[:top_k]

        if not sorted_items:
            return {
                "success": True,
                "indexed_count": 0,
                "message": "No items to ingest"
            }

        # Prepare simplified data for graph (title/source/url/time only)
        graph_items = []
        for item in sorted_items:
            graph_items.append({
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "timestamp": item.get("timestamp", ""),
                "hot_score": item.get("hot_score", 0),
            })

        # Import and call ingest function
        from src.graphrag.team import ingest_news_to_graph

        result = await ingest_news_to_graph(
            news_items=graph_items,
            extract_first=extract
        )

        logger.info(f"GraphRAG ingested {result.indexed_count} items to graph {result.graph_id}")

        return {
            "success": True,
            "indexed_count": result.indexed_count,
            "graph_id": result.graph_id,
            "message": f"Successfully ingested {result.indexed_count} news items"
        }

    except Exception as e:
        logger.error(f"GraphRAG ingest failed: {e}")
        return {
            "success": False,
            "indexed_count": 0,
            "error": str(e)
        }


@tool
def graphrag_query(query: str, mode: str = "quick") -> Dict[str, Any]:
    """
    Query the GraphRAG knowledge graph.

    Args:
        query: User question
        mode: Search mode (quick/panorama/insight)

    Returns:
        Answer dict with content and sources
    """
    from src.graphrag.config import get_config
    import asyncio

    config = get_config()

    if not config.ENABLED:
        return {
            "success": False,
            "error": "GraphRAG is disabled"
        }

    try:
        from src.graphrag.team import query_graph

        # Run async query
        result = asyncio.run(query_graph(query, search_mode=mode))

        return {
            "success": True,
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources[:5],
            "insights": result.insights
        }

    except Exception as e:
        logger.error(f"GraphRAG query failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
