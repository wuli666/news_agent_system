"""
Summary Team Subgraph - handles final report generation.
"""
import logging
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from src.graph.types import SummaryTeamState

logger = logging.getLogger(__name__)


def create_summary_team():
    """
    Create the summary team subgraph.

    Architecture (sequential):
        coordinator -> timeline -> trend -> chart -> writer -> END

    Unlike collection/analysis teams, summary runs sequentially
    as each step builds on the previous.
    """
    from src.agents.nodes import (
        timeline_agent_node as _timeline_agent,
        trend_agent_node as _trend_agent,
        summary_chart_agent_node as _chart_agent,
        summary_writer_agent_node as _writer_agent
    )

    # Wrapper functions to adapt node names for subgraph
    async def timeline(state: SummaryTeamState):
        """Wrapper for timeline_agent with subgraph-aware routing."""
        result = await _timeline_agent(state)
        if hasattr(result, 'goto') and result.goto == "trend_agent":
            return Command(update=result.update, goto="trend")
        return result

    async def trend(state: SummaryTeamState):
        """Wrapper for trend_agent with subgraph-aware routing."""
        result = await _trend_agent(state)
        if hasattr(result, 'goto') and result.goto == "summary_chart_agent":
            return Command(update=result.update, goto="chart")
        return result

    async def chart(state: SummaryTeamState):
        """Wrapper for chart_agent with subgraph-aware routing."""
        result = await _chart_agent(state)
        if hasattr(result, 'goto') and result.goto == "summary_writer_agent":
            return Command(update=result.update, goto="writer")
        return result

    async def writer(state: SummaryTeamState):
        """Wrapper for writer_agent - last node in chain."""
        result = await _writer_agent(state)
        # Writer is the last node, should goto END
        # summary_writer_agent_node 返回的是普通 dict，需要按实际类型取 update
        update_payload = result if isinstance(result, dict) else getattr(result, "update", {})
        return Command(update=update_payload, goto=END)

    def coordinator(state: SummaryTeamState):
        """
        Coordinator: initiates summary generation.
        """
        logger.info("=== Summary Team Subgraph: Coordinator starting ===")

        # Directly start with timeline
        return Command(goto="timeline")


    # Create subgraph
    graph = StateGraph(SummaryTeamState)

    # Add nodes
    graph.add_node("coordinator", coordinator)
    graph.add_node("timeline", timeline)
    graph.add_node("trend", trend)
    graph.add_node("chart", chart)
    graph.add_node("writer", writer)

    # Set entry point
    graph.set_entry_point("coordinator")

    # Sequential flow
    # Note: Each agent uses Command(goto=...) to specify next node
    # We define the edges here for visualization, but routing is controlled by Command

    graph.add_edge("timeline", "trend")
    graph.add_edge("trend", "chart")
    graph.add_edge("chart", "writer")
    graph.add_edge("writer", END)

    logger.info("Summary team subgraph created successfully")

    return graph.compile()
