"""
GraphRAG LangGraph Team

Defines the GraphRAG agent team as a LangGraph subgraph that can be
integrated with the main news_agent_system workflow.
"""

import logging
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END
from langgraph.types import Command

from .agents.extractor import ExtractorAgent, ExtractionResult
from .agents.indexer import IndexerAgent, IndexResult
from .agents.retriever import RetrieverAgent, RetrievalResult
from .agents.reasoner import ReasonerAgent, ReasoningResult
from .services.zep_client import ZepGraphClient
from .config import get_config

logger = logging.getLogger(__name__)


# ============================================================
# State Definition
# ============================================================

class GraphRAGState(TypedDict):
    """State for GraphRAG team workflow."""
    # Input
    task: str  # "ingest" or "query"
    query: Optional[str]
    news_items: List[Dict[str, Any]]

    # Processing
    graph_id: Optional[str]
    extractions: Annotated[List[ExtractionResult], operator.add]
    retrieval: Optional[RetrievalResult]

    # Output
    index_result: Optional[IndexResult]
    answer: Optional[ReasoningResult]
    error: Optional[str]

    # Control
    last_agent: str


class IngestState(TypedDict):
    """State for ingest-only workflow."""
    news_items: List[Dict[str, Any]]
    graph_id: Optional[str]
    extractions: List[ExtractionResult]
    index_result: Optional[IndexResult]
    error: Optional[str]


class QueryState(TypedDict):
    """State for query-only workflow."""
    query: str
    graph_id: Optional[str]
    search_mode: str  # "quick", "panorama", "insight"
    retrieval: Optional[RetrievalResult]
    answer: Optional[ReasoningResult]
    error: Optional[str]


# ============================================================
# Agent Nodes
# ============================================================

async def extractor_node(state: IngestState) -> Dict[str, Any]:
    """Extract entities and relations from news items."""
    logger.info("=== Extractor Agent ===")

    news_items = state.get("news_items", [])
    if not news_items:
        return {"extractions": [], "error": "No news items to extract"}

    try:
        agent = ExtractorAgent()
        results = await agent.extract_from_news_items(news_items)

        logger.info(f"Extracted from {len(news_items)} items, got {len(results)} results")

        return {"extractions": results}

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {"extractions": [], "error": str(e)}


async def indexer_node(state: IngestState) -> Dict[str, Any]:
    """Index extracted data or raw news items into Zep."""
    logger.info("=== Indexer Agent ===")

    graph_id = state.get("graph_id")
    extractions = state.get("extractions", [])
    news_items = state.get("news_items", [])

    try:
        agent = IndexerAgent()

        # If we have extractions, index them
        if extractions:
            result = await agent.index_extractions(extractions, graph_id)
        else:
            # Otherwise index raw news items (Zep will extract)
            result = await agent.index_news_items(news_items, graph_id)

        logger.info(f"Indexed {result.indexed_count} items to graph {result.graph_id}")

        return {
            "index_result": result,
            "graph_id": result.graph_id
        }

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        return {"error": str(e)}


async def retriever_node(state: QueryState) -> Dict[str, Any]:
    """Search the knowledge graph."""
    logger.info("=== Retriever Agent ===")

    query = state.get("query", "")
    graph_id = state.get("graph_id")
    mode = state.get("search_mode", "quick")

    if not query:
        return {"error": "No query provided"}

    try:
        agent = RetrieverAgent()
        result = await agent.search(query, mode, graph_id)

        logger.info(f"Retrieved {len(result.results)} results for: {query[:50]}...")

        return {"retrieval": result}

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {"error": str(e)}


async def reasoner_node(state: QueryState) -> Dict[str, Any]:
    """Synthesize answer from retrieval results."""
    logger.info("=== Reasoner Agent ===")

    query = state.get("query", "")
    retrieval = state.get("retrieval")

    if not retrieval:
        return {"error": "No retrieval results to reason over"}

    try:
        agent = ReasonerAgent()
        result = await agent.reason(query, retrieval)

        logger.info(f"Generated answer with confidence {result.confidence:.2f}")

        return {"answer": result}

    except Exception as e:
        logger.error(f"Reasoning failed: {e}")
        return {"error": str(e)}


# ============================================================
# Workflow Definitions
# ============================================================

def create_ingest_workflow() -> StateGraph:
    """
    Create the ingest workflow subgraph.

    Flow: extractor -> indexer -> END
    """
    workflow = StateGraph(IngestState)

    # Add nodes
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("indexer", indexer_node)

    # Set entry point
    workflow.set_entry_point("extractor")

    # Add edges
    workflow.add_edge("extractor", "indexer")
    workflow.add_edge("indexer", END)

    return workflow.compile()


def create_query_workflow() -> StateGraph:
    """
    Create the query workflow subgraph.

    Flow: retriever -> reasoner -> END
    """
    workflow = StateGraph(QueryState)

    # Add nodes
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reasoner", reasoner_node)

    # Set entry point
    workflow.set_entry_point("retriever")

    # Add edges
    workflow.add_edge("retriever", "reasoner")
    workflow.add_edge("reasoner", END)

    return workflow.compile()


def create_graphrag_team() -> StateGraph:
    """
    Create the full GraphRAG team workflow.

    Routes based on task:
    - "ingest": extractor -> indexer -> END
    - "query": retriever -> reasoner -> END
    """
    workflow = StateGraph(GraphRAGState)

    # Add nodes
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("indexer", indexer_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("reasoner", reasoner_node)

    # Router function
    def route_task(state: GraphRAGState) -> str:
        task = state.get("task", "query")
        if task == "ingest":
            return "extractor"
        else:
            return "retriever"

    # Set conditional entry
    workflow.set_conditional_entry_point(
        route_task,
        {"extractor": "extractor", "retriever": "retriever"}
    )

    # Add edges for ingest path
    workflow.add_edge("extractor", "indexer")
    workflow.add_edge("indexer", END)

    # Add edges for query path
    workflow.add_edge("retriever", "reasoner")
    workflow.add_edge("reasoner", END)

    return workflow.compile()


# ============================================================
# Integration Helpers
# ============================================================

async def ingest_news_to_graph(
    news_items: List[Dict[str, Any]],
    graph_id: Optional[str] = None,
    extract_first: bool = True
) -> IndexResult:
    """
    Helper function to ingest news items into the graph.

    Args:
        news_items: List of news item dictionaries
        graph_id: Optional target graph ID
        extract_first: Whether to use LLM extraction first

    Returns:
        IndexResult
    """
    if extract_first:
        workflow = create_ingest_workflow()
        result = await workflow.ainvoke({
            "news_items": news_items,
            "graph_id": graph_id,
            "extractions": [],
        })
        return result.get("index_result")
    else:
        # Direct indexing without extraction
        agent = IndexerAgent()
        return await agent.index_news_items(news_items, graph_id)


async def query_graph(
    query: str,
    graph_id: Optional[str] = None,
    search_mode: str = "quick"
) -> ReasoningResult:
    """
    Helper function to query the graph.

    Args:
        query: User question
        graph_id: Target graph ID
        search_mode: Search mode ("quick", "panorama", "insight")

    Returns:
        ReasoningResult
    """
    workflow = create_query_workflow()
    result = await workflow.ainvoke({
        "query": query,
        "graph_id": graph_id,
        "search_mode": search_mode,
    })
    return result.get("answer")


# ============================================================
# Main Workflow Integration Hook
# ============================================================

async def graphrag_team_node(state: Dict[str, Any]) -> Command:
    """
    Integration node for main workflow.

    Can be added to the main news_agent_system workflow as:
    workflow.add_node("graphrag_team", graphrag_team_node)

    Expects state to have:
    - text_news: List of collected news items
    - graphrag_query: Optional query to answer

    Returns Command with:
    - graphrag_indexed: Number of items indexed
    - graphrag_answer: Answer to query (if provided)
    """
    config = get_config()

    if not config.ENABLED:
        logger.info("GraphRAG is disabled, skipping")
        return Command(update={"last_agent": "graphrag_team"}, goto=END)

    news_items = state.get("text_news", [])
    query = state.get("graphrag_query")

    updates = {"last_agent": "graphrag_team"}

    # Ingest news if available and auto_ingest is enabled
    if news_items and config.AUTO_INGEST:
        try:
            result = await ingest_news_to_graph(news_items)
            updates["graphrag_indexed"] = result.indexed_count if result else 0
            logger.info(f"GraphRAG indexed {updates['graphrag_indexed']} items")
        except Exception as e:
            logger.error(f"GraphRAG ingest failed: {e}")
            updates["graphrag_indexed"] = 0

    # Answer query if provided
    if query:
        try:
            answer = await query_graph(query)
            updates["graphrag_answer"] = answer.answer if answer else ""
            logger.info(f"GraphRAG answered query: {query[:50]}...")
        except Exception as e:
            logger.error(f"GraphRAG query failed: {e}")
            updates["graphrag_answer"] = f"Error: {e}"

    return Command(update=updates, goto=END)
