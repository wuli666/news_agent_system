"""
LangGraph workflow definition for news collection system.
"""
import logging
from langgraph.graph import StateGraph, END
from src.graph.types import State
from src.agents.nodes import (
    coordinator_node,
    text_agent_node,
    video_agent_node,
    supervisor_node,
    summary_agent_node,
)

logger = logging.getLogger(__name__)


def should_continue(state: State) -> str:
    """
    Routing function to decide next step after supervisor.
    
    Args:
        state: Current state
        
    Returns:
        Next node name
    """
    decision = state.get("supervisor_decision", "continue")
    
    if decision == "summarize":
        return "summary"
    else:
        # Decide which agent to run next
        text_count = len(state.get("text_news", []))
        video_count = len(state.get("video_news", []))
        
        # Alternate between text and video agents, prioritizing the one with fewer items
        if text_count <= video_count:
            return "text_agent"
        else:
            return "video_agent"


def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for news collection.
    
    Workflow structure:
        coordinator -> supervisor -> [text_agent / video_agent] -> supervisor -> ... -> summary -> END
    
    Returns:
        Compiled workflow
    """
    # Create state graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("text_agent", text_agent_node)
    workflow.add_node("video_agent", video_agent_node)
    workflow.add_node("summary", summary_agent_node)
    
    # Set entry point
    workflow.set_entry_point("coordinator")
    
    # Add edges
    # coordinator -> supervisor
    workflow.add_edge("coordinator", "supervisor")
    
    # supervisor -> text_agent / video_agent / summary (conditional)
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "text_agent": "text_agent",
            "video_agent": "video_agent",
            "summary": "summary",
        }
    )
    
    # text_agent -> supervisor
    workflow.add_edge("text_agent", "supervisor")
    
    # video_agent -> supervisor
    workflow.add_edge("video_agent", "supervisor")
    
    # summary -> END
    workflow.add_edge("summary", END)
    
    # Compile workflow
    compiled_workflow = workflow.compile()
    
    logger.info("Workflow created and compiled successfully")
    
    return compiled_workflow


if __name__ == "__main__":
    # Test workflow creation
    workflow = create_workflow()
    print("Workflow structure:")
    print("coordinator -> supervisor -> [text_agent / video_agent] -> supervisor -> ... -> summary -> END")
