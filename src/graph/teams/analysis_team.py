"""
Analysis Team Subgraph - handles news analysis with intelligent loop control.
"""
import logging
from langgraph.graph import StateGraph, END
from langgraph.types import Send, Command
from src.graph.types import AnalysisTeamState
from src.config.settings import settings

logger = logging.getLogger(__name__)


def create_analysis_team():
    """
    Create the analysis team subgraph with intelligent inner loop.

    Architecture:
        coordinator -> [research ∥ sentiment ∥ relationship] -> reflect -> merger
                       ↑______________________________________________|
                                    (loop if gaps found)

    Inner loop logic:
    - If skip_sentiment_relationship=True: research -> reflect (optimized path)
    - Otherwise: research ∥ sentiment ∥ relationship -> reflect (full analysis)
    - Merger checks information_gaps and cycles back if needed
    """
    from src.agents.nodes import (
        research_agent_node as _research_agent,
        sentiment_agent_node as _sentiment_agent,
        relationship_agent_node as _relationship_agent,
        reflect_agent_node as _reflect_agent
    )

    # Wrapper functions to adapt node names for subgraph
    async def research(state: AnalysisTeamState):
        """Wrapper for research_agent - parallel execution version."""
        result = await _research_agent(state)
        # Extract only the state updates, ignore goto (handled by aggregator)
        if hasattr(result, 'update'):
            return result.update
        return result

    async def sentiment(state: AnalysisTeamState):
        """Wrapper for sentiment_agent - parallel execution version."""
        result = await _sentiment_agent(state)
        # Extract only the state updates, ignore goto (handled by aggregator)
        if hasattr(result, 'update'):
            return result.update
        return result

    async def relationship(state: AnalysisTeamState):
        """Wrapper for relationship_agent - parallel execution version."""
        result = await _relationship_agent(state)
        # Extract only the state updates, ignore goto (handled by aggregator)
        if hasattr(result, 'update'):
            return result.update
        return result

    async def reflect(state: AnalysisTeamState):
        """Wrapper for reflect_agent with subgraph-aware routing."""
        result = await _reflect_agent(state)
        if hasattr(result, 'goto') and result.goto == "analysis_team_merger":
            return Command(update=result.update, goto="merger")
        return result

    def coordinator(state: AnalysisTeamState):
        """
        Coordinator: prepares data for parallel execution.
        Routing is handled by conditional edge function.
        """
        logger.info("=== Analysis Team Subgraph: Coordinator starting ===")

        text_news = state.get("text_news", [])

        if not text_news:
            logger.warning("No text news for analysis team")
            # Return empty state, routing handled by conditional edge
            return {}

        # Just return empty dict, routing is handled by route_to_analyzers
        return {}

    def route_to_analyzers(state: AnalysisTeamState):
        """
        Conditional edge function: routes to analyzers in parallel using Send API.
        """
        text_news = state.get("text_news", [])

        if not text_news:
            # No news to analyze, go directly to END
            logger.info("No text news to analyze, ending subgraph")
            return END

        skip_sentiment_relationship = state.get("skip_sentiment_relationship", False)

        if skip_sentiment_relationship:
            # Optimized path: only research
            logger.info("⚡ Optimized path: research only")
            return [Send("research", state)]
        else:
            # Full analysis path: parallel execution
            logger.info("🔀 Full analysis: research ∥ sentiment ∥ relationship (parallel)")
            return [
                Send("research", state),
                Send("sentiment", state),
                Send("relationship", state),
            ]

    def aggregator(state: AnalysisTeamState):
        """
        Aggregator: collects results from parallel analyzers and routes to reflect.
        """
        logger.info("=== Analysis Team: Aggregating parallel results ===")
        research_count = len(state.get("research_notes", []))
        logger.info(f"Collected: {research_count} research notes")
        return Command(goto="reflect")


    def merger(state: AnalysisTeamState):
        """
        Merger: controls inner loop based on information gaps.

        Inner loop decision:
        - If gaps exist and cycle < 3 -> loop back to coordinator
        - Otherwise -> complete subgraph (return to main workflow)
        """
        logger.info("=== Analysis Team Subgraph: Merging and loop control ===")

        information_gaps = state.get("information_gaps", [])
        cycle_count = state.get("cycle_count", 0)
        max_cycles = settings.ANALYSIS_MAX_CYCLES

        logger.info(f"Cycle status: count={cycle_count}/{max_cycles}, gaps={len(information_gaps)}")

        # Inner loop decision
        if information_gaps and cycle_count < max_cycles:
            new_cycle_count = cycle_count + 1

            # Intelligent routing strategy:
            # - Cycle 1 (new_cycle_count == 1): Full analysis (research + sentiment + relationship)
            #   First pass needs complete analysis to establish baseline
            # - Cycle 2+ (new_cycle_count >= 2): Research only
            #   Subsequent passes only fill information gaps, sentiment/relationship won't change much

            if new_cycle_count == 1:
                # First cycle: always full analysis
                logger.info(f"🔄 Inner loop round {new_cycle_count}: full analysis (first cycle, {len(information_gaps)} gaps)")
                return Command(update={
                    "cycle_count": new_cycle_count,
                    "skip_sentiment_relationship": False,
                    "loop_complete": False,
                    "analysis_gap_pending": True,
                }, goto="coordinator")
            else:
                # Subsequent cycles: research only (optimize for efficiency)
                gap_types = [gap.get("missing_type", "") for gap in information_gaps]
                logger.info(f"⚡ Inner loop round {new_cycle_count}: research only (filling gaps, types: {set(gap_types)})")
                return Command(update={
                    "cycle_count": new_cycle_count,
                    "skip_sentiment_relationship": True,
                    "loop_complete": False,
                    "analysis_gap_pending": True,
                }, goto="coordinator")
        else:
            # Loop complete
            if information_gaps and cycle_count >= max_cycles:
                logger.info(f"⚠️ Max cycles reached ({max_cycles}), {len(information_gaps)} gaps remain")
            else:
                logger.info("✅ Inner loop complete: no gaps or all filled")

            # Reset cycle count and complete subgraph
            return Command(update={
                "cycle_count": cycle_count,
                "skip_sentiment_relationship": False,
                "analysis_gap_pending": bool(information_gaps),
                "loop_complete": True,
            }, goto=END)


    # Create subgraph
    graph = StateGraph(AnalysisTeamState)

    # Add nodes
    graph.add_node("coordinator", coordinator)
    graph.add_node("research", research)
    graph.add_node("sentiment", sentiment)
    graph.add_node("relationship", relationship)
    graph.add_node("aggregator", aggregator)  # New: aggregator for parallel results
    graph.add_node("reflect", reflect)
    graph.add_node("merger", merger)

    # Set entry point
    graph.set_entry_point("coordinator")

    # Use conditional edges for parallel routing from coordinator
    graph.add_conditional_edges(
        "coordinator",
        route_to_analyzers,
        # Explicit path mapping (Send API will handle dynamic routing)
        {
            END: END,  # When no news to analyze
            # research, sentiment, relationship paths handled by Send
        }
    )

    # All analyzers route to aggregator (collects parallel results)
    graph.add_edge("research", "aggregator")
    graph.add_edge("sentiment", "aggregator")
    graph.add_edge("relationship", "aggregator")

    # Aggregator routes to reflect
    graph.add_edge("aggregator", "reflect")

    # Reflect routes to merger
    graph.add_edge("reflect", "merger")

    # Conditional routing from merger
    def route_merger(state: AnalysisTeamState):
        """Route from merger: loop back or complete."""
        # Respect explicit completion flag to avoid looping when max cycles hit
        max_cycles = settings.ANALYSIS_MAX_CYCLES

        if state.get("loop_complete"):
            return END

        information_gaps = state.get("information_gaps") or []
        cycle_count = state.get("cycle_count", 0)

        if information_gaps and cycle_count < max_cycles:
            return "coordinator"
        return END

    graph.add_conditional_edges(
        "merger",
        route_merger,
        {
            "coordinator": "coordinator",
            END: END,
        }
    )

    # NOTE: Parallel execution is now handled by Send API via conditional edges
    # All analyzers complete before reflect is invoked

    logger.info("Analysis team subgraph created successfully with parallel execution")
    logger.info("Architecture: coordinator -> [research ∥ sentiment ∥ relationship] -> aggregator -> reflect -> merger")

    return graph.compile()
