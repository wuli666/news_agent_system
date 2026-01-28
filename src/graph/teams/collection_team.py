"""
Collection Team Subgraph - handles text and video news collection in parallel.
"""
import logging
from typing import Any
from langgraph.graph import StateGraph, END
from langgraph.types import Send, Command
from src.graph.types import CollectionTeamState

logger = logging.getLogger(__name__)


def create_collection_team():
    """Create the collection team subgraph with parallel execution."""
    from src.agents.nodes import text_collector_agent as _text_collector, video_collector_agent as _video_collector

    async def text_collector(state: CollectionTeamState):
        """Wrapper for text_collector_agent with subgraph-aware routing."""
        agent_state = {
            "task": state.get("task", ""),
            "date": state.get("date", ""),
            "assigned_news": state.get("assigned_news", []),
            "text_news": state.get("text_news", []),
            "video_batch": state.get("video_batch", []),
            "iteration": 0,
        }

        result = await _text_collector(agent_state)
        updates = result.update if hasattr(result, 'update') else {}
        goto = result.goto if hasattr(result, 'goto') else "video_collector_agent"

        if goto == "video_collector_agent":
            goto = "video_collector"

        return Command(update=updates, goto=goto)

    async def video_collector(state: CollectionTeamState):
        """Wrapper for video_collector_agent with subgraph-aware routing."""
        agent_state = {
            "task": state.get("task", ""),
            "date": state.get("date", ""),
            "assigned_news": state.get("assigned_news", []),
            "video_news": state.get("video_news", []),
            "iteration": 0,
        }

        result = await _video_collector(agent_state)
        updates = result.update if hasattr(result, 'update') else {}
        goto = result.goto if hasattr(result, 'goto') else "collection_team_merger"

        if goto == "collection_team_merger":
            goto = "merger"

        return Command(update=updates, goto=goto)

    def coordinator(state: CollectionTeamState):
        """Coordinator: splits batch and prepares data for parallel execution."""
        input_batch = state.get("input_batch", [])

        if not input_batch:
            return {
                "text_batch": [],
                "video_batch": [],
            }

        text_batch = [item for item in input_batch if item.get("category") == "text"]
        video_batch = [item for item in input_batch if item.get("category") == "video"]

        logger.info(f"Batch split: {len(text_batch)} text, {len(video_batch)} video")

        return {
            "text_batch": text_batch,
            "video_batch": video_batch,
        }

    def route_to_collectors(state: CollectionTeamState):
        """Conditional edge function: routes to collectors in parallel using Send API."""
        text_batch = state.get("text_batch", [])
        video_batch = state.get("video_batch", [])

        sends = []
        if text_batch:
            text_state = {**state, "assigned_news": text_batch}
            sends.append(Send("text_collector", text_state))
        if video_batch:
            video_state = {**state, "assigned_news": video_batch}
            sends.append(Send("video_collector", video_state))

        if not sends:
            return "merger"

        return sends


    def merger(state: CollectionTeamState):
        """Merger: combines results from parallel collectors."""
        text_count = len(state.get("text_news", []))
        video_count = len(state.get("video_news", []))

        logger.info(f"Collection team completed: {text_count} text, {video_count} video")

        return Command(goto=END)


    # Create subgraph
    graph = StateGraph(CollectionTeamState)

    # Add nodes
    graph.add_node("coordinator", coordinator)
    graph.add_node("text_collector", text_collector)
    graph.add_node("video_collector", video_collector)
    graph.add_node("merger", merger)

    # Set entry point
    graph.set_entry_point("coordinator")

    graph.add_conditional_edges(
        "coordinator",
        route_to_collectors,
        {"merger": "merger"}
    )

    # Collectors route to merger
    graph.add_edge("text_collector", "merger")
    graph.add_edge("video_collector", "merger")
    graph.add_edge("merger", END)

    return graph.compile()
