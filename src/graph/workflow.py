"""
Subgraph-based LangGraph workflow for news collection system.

Key improvements over flat architecture:
- Team logic encapsulated in subgraphs for modularity
- State isolation: each team has its own state type
- Inner loops contained within subgraphs (analysis team)
- Parallel execution via Send API within subgraphs
- Cleaner main workflow with only 5 nodes
"""
import logging
from langgraph.graph import StateGraph, END
from src.graph.types import State, CollectionTeamState, AnalysisTeamState, SummaryTeamState

# Import core nodes
from src.agents.nodes import (
    coordinator_node,
    news_collector_node,
    main_supervisor_node,
)

# Import team subgraphs
from src.graph.teams import (
    create_collection_team,
    create_analysis_team,
    create_summary_team,
)

logger = logging.getLogger(__name__)


def route_supervisor(state: State) -> str:
    """Main supervisor routing function."""
    decision = state.get("supervisor_decision", "collect")

    if decision == "collect":
        return "news_collector"
    elif decision == "collection_team":
        return "collection_team"
    elif decision == "analysis_team":
        return "analysis_team"
    elif decision == "summarize":
        return "summary_team"
    else:
        # Default: back to supervisor
        return "main_supervisor"


def prepare_collection_team_input(state: State) -> CollectionTeamState:
    """Prepare input for collection team subgraph."""
    return {
        "input_batch": state.get("latest_news_batch", []),
        "task": state.get("task", ""),
        "date": state.get("date", ""),
        "text_batch": [],
        "video_batch": [],
        "assigned_news": [],  # Will be set by coordinator
        "text_news": state.get("text_news", []),
        "video_news": state.get("video_news", []),
        "text_analysis": state.get("text_analysis"),
        "video_analysis": state.get("video_analysis"),
    }


def prepare_analysis_team_input(state: State) -> AnalysisTeamState:
    """Prepare input for analysis team subgraph."""
    return {
        "text_news": state.get("text_news", []),
        "date": state.get("date", ""),
        "iteration": state.get("iteration", 0),
        "cycle_count": state.get("text_team_cycle_count", 0),
        "information_gaps": state.get("information_gaps", []),
        "skip_sentiment_relationship": bool(state.get("skip_sentiment_relationship", False)),
        "analysis_gap_pending": bool(state.get("analysis_gap_pending", False)),
        "gap_fill_mode": bool(state.get("gap_fill_mode", False)),
        "research_notes": state.get("research_notes", []),
        "news_images": state.get("news_images", {}),
        "text_sentiment": state.get("text_sentiment"),
        "text_relationship_graph": state.get("text_relationship_graph"),
        "text_reflection": state.get("text_reflection"),
    }


def prepare_summary_team_input(state: State) -> SummaryTeamState:
    """Prepare input for summary team subgraph."""
    return {
        "task": state.get("task", ""),
        "date": state.get("date", ""),
        "text_news": state.get("text_news", []),
        "video_news": state.get("video_news", []),
        "research_notes": state.get("research_notes", []),
        "text_sentiment": state.get("text_sentiment"),
        "text_relationship_graph": state.get("text_relationship_graph"),
        "trending_keywords": state.get("trending_keywords"),
        "timeline_analysis": state.get("timeline_analysis"),
        "trend_analysis": state.get("trend_analysis"),
        "timeline_chart_path": state.get("timeline_chart_path"),
        "final_report": state.get("final_report"),
        "news_images": state.get("news_images", {}),
        "daily_papers": state.get("daily_papers", []),
    }


def merge_collection_team_output(main_state: State, team_state: CollectionTeamState) -> State:
    """Merge collection team output back to main state."""
    return {
        **main_state,
        "text_news": team_state.get("text_news", []),
        "video_news": team_state.get("video_news", []),
        "text_analysis": team_state.get("text_analysis"),
        "video_analysis": team_state.get("video_analysis"),
        "last_agent": "collection_team",
    }


def merge_analysis_team_output(main_state: State, team_state: AnalysisTeamState) -> State:
    """Merge analysis team output back to main state."""
    return {
        **main_state,
        "research_notes": team_state.get("research_notes", []),
        "news_images": team_state.get("news_images", {}),
        "text_sentiment": team_state.get("text_sentiment"),
        "text_relationship_graph": team_state.get("text_relationship_graph"),
        "text_reflection": team_state.get("text_reflection"),
        "information_gaps": team_state.get("information_gaps", []),
        "text_team_cycle_count": team_state.get("cycle_count", main_state.get("text_team_cycle_count", 0)),
        "analysis_gap_pending": bool(team_state.get("analysis_gap_pending", False) or (team_state.get("information_gaps"))),
        "gap_fill_mode": bool(team_state.get("gap_fill_mode", main_state.get("gap_fill_mode", False))),
        "skip_sentiment_relationship": bool(team_state.get("skip_sentiment_relationship", main_state.get("skip_sentiment_relationship", False))),
        "last_agent": "analysis_team",
    }


def merge_summary_team_output(main_state: State, team_state: SummaryTeamState) -> State:
    """Merge summary team output back to main state."""
    return {
        **main_state,
        "timeline_analysis": team_state.get("timeline_analysis"),
        "trend_analysis": team_state.get("trend_analysis"),
        "timeline_chart_path": team_state.get("timeline_chart_path"),
        "final_report": team_state.get("final_report"),
        "completed_at": team_state.get("completed_at", main_state.get("completed_at")),
        "last_agent": team_state.get("last_agent", main_state.get("last_agent", "summary_team")),
    }


def create_workflow():
    """Create the subgraph-based LangGraph workflow."""
    workflow = StateGraph(State)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("main_supervisor", main_supervisor_node)
    workflow.add_node("news_collector", news_collector_node)

    collection_team_graph = create_collection_team()
    analysis_team_graph = create_analysis_team()
    summary_team_graph = create_summary_team()
    async def collection_team_node(state: State):
        """Wrapper for collection team subgraph."""
        logger.info("=== Invoking Collection Team Subgraph ===")
        team_input = prepare_collection_team_input(state)
        team_output = await collection_team_graph.ainvoke(team_input)
        return merge_collection_team_output(state, team_output)

    async def analysis_team_node(state: State):
        """Wrapper for analysis team subgraph."""
        logger.info("=== Invoking Analysis Team Subgraph ===")
        team_input = prepare_analysis_team_input(state)
        team_output = await analysis_team_graph.ainvoke(team_input)
        return merge_analysis_team_output(state, team_output)

    async def summary_team_node(state: State):
        """Wrapper for summary team subgraph."""
        logger.info("=== Invoking Summary Team Subgraph ===")
        team_input = prepare_summary_team_input(state)
        team_output = await summary_team_graph.ainvoke(team_input)
        return merge_summary_team_output(state, team_output)

    workflow.add_node("collection_team", collection_team_node)
    workflow.add_node("analysis_team", analysis_team_node)
    workflow.add_node("summary_team", summary_team_node)

    workflow.set_entry_point("coordinator")

    workflow.add_edge("coordinator", "main_supervisor")
    workflow.add_edge("news_collector", "main_supervisor")
    workflow.add_conditional_edges(
        "main_supervisor",
        route_supervisor,
        {
            "news_collector": "news_collector",
            "collection_team": "collection_team",
            "analysis_team": "analysis_team",
            "summary_team": "summary_team",
            "main_supervisor": "main_supervisor",
        }
    )

    workflow.add_edge("collection_team", "main_supervisor")
    workflow.add_edge("analysis_team", "main_supervisor")
    workflow.add_edge("summary_team", END)

    return workflow.compile()


if __name__ == "__main__":
    # Test workflow creation
    workflow = create_workflow()
    print("✅ Subgraph-based workflow structure:")
    print("   Main: coordinator -> supervisor -> [teams] -> END")
    print("   Collection Team: coordinator -> [text ∥ video] -> merger")
    print("   Analysis Team: coordinator -> [research ∥ sentiment ∥ relationship] -> reflect -> merger (with loop)")
    print("   Summary Team: coordinator -> timeline -> trend -> chart -> writer")
