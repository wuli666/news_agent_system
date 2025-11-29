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
    """
    Create the collection team subgraph.

    Architecture:
        coordinator -> [text_collector ∥ video_collector] -> merger -> END

    Uses Send API for parallel execution within subgraph.
    """
    from src.agents.nodes import text_collector_agent as _text_collector, video_collector_agent as _video_collector

    async def text_collector(state: CollectionTeamState):
        """Wrapper for text_collector_agent with subgraph-aware routing."""
        # Adapt CollectionTeamState to State format for agent
        agent_state = {
            "task": state.get("task", ""),
            "date": state.get("date", ""),
            "assigned_news": state.get("assigned_news", []),
            "text_news": state.get("text_news", []),
            "video_batch": state.get("video_batch", []),
            "iteration": 0,  # Not used in collector
        }

        result = await _text_collector(agent_state)

        # Extract updates and adapt goto
        updates = result.update if hasattr(result, 'update') else {}
        goto = result.goto if hasattr(result, 'goto') else "video_collector_agent"

        # Update the goto to match subgraph node names
        if goto == "video_collector_agent":
            goto = "video_collector"

        return Command(update=updates, goto=goto)

    async def video_collector(state: CollectionTeamState):
        """Wrapper for video_collector_agent with subgraph-aware routing."""
        # Adapt CollectionTeamState to State format for agent
        agent_state = {
            "task": state.get("task", ""),
            "date": state.get("date", ""),
            "assigned_news": state.get("assigned_news", []),
            "video_news": state.get("video_news", []),
            "iteration": 0,  # Not used in collector
        }

        result = await _video_collector(agent_state)

        # Extract updates and adapt goto
        updates = result.update if hasattr(result, 'update') else {}
        goto = result.goto if hasattr(result, 'goto') else "collection_team_merger"

        # Update the goto to match subgraph node names
        if goto == "collection_team_merger":
            goto = "merger"

        return Command(update=updates, goto=goto)

    def coordinator(state: CollectionTeamState):
        """
        Coordinator: splits batch and prepares data for parallel execution.
        """
        logger.info("=== Collection Team Subgraph: Coordinator starting ===")

        input_batch = state.get("input_batch", [])

        if not input_batch:
            logger.warning("No input batch provided to collection team")
            # Store empty batches, routing handled by conditional edge
            return {
                "text_batch": [],
                "video_batch": [],
            }

        # Split batch by category
        text_batch = [item for item in input_batch if item.get("category") == "text"]
        video_batch = [item for item in input_batch if item.get("category") == "video"]

        logger.info(f"Batch split: {len(text_batch)} text, {len(video_batch)} video")

        # Store batches (routing handled by conditional edge)
        return {
            "text_batch": text_batch,
            "video_batch": video_batch,
        }

    def route_to_collectors(state: CollectionTeamState):
        """
        Conditional edge function: routes to collectors in parallel using Send API.
        """
        text_batch = state.get("text_batch", [])
        video_batch = state.get("video_batch", [])

        sends = []
        if text_batch:
            # Send state with assigned_news set to text_batch
            text_state = {**state, "assigned_news": text_batch}
            sends.append(Send("text_collector", text_state))
        if video_batch:
            # Send state with assigned_news set to video_batch
            video_state = {**state, "assigned_news": video_batch}
            sends.append(Send("video_collector", video_state))

        if not sends:
            # No data to process, go directly to merger
            logger.info("No batches to process, skipping collectors")
            return "merger"

        logger.info(f"🔀 Parallel routing: dispatching to {len(sends)} collectors")
        return sends


    def merger(state: CollectionTeamState):
        """
        Merger: combines results from parallel collectors.
        """
        logger.info("=== Collection Team Subgraph: Merging results ===")

        text_count = len(state.get("text_news", []))
        video_count = len(state.get("video_news", []))

        logger.info(f"Collection team completed: {text_count} text, {video_count} video")

        # Return to main workflow
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

    # Use conditional edges for parallel routing
    graph.add_conditional_edges(
        "coordinator",
        route_to_collectors,
        # Explicit path mapping (Send API will handle dynamic routing)
        {
            "merger": "merger",  # When no batches to process
            # text_collector and video_collector paths handled by Send
        }
    )

    # Collectors route to merger
    graph.add_edge("text_collector", "merger")
    graph.add_edge("video_collector", "merger")

    # Merger completes subgraph
    graph.add_edge("merger", END)

    logger.info("Collection team subgraph created successfully with parallel execution")

    return graph.compile()
