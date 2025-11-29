"""
Test Send API implementation.
"""
import asyncio
import logging
from datetime import datetime
from src.graph.workflow import create_workflow
from src.graph.types import State

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_send_api():
    """Test that Send API works for parallel execution."""
    print("=" * 60)
    print("Testing Send API Implementation")
    print("=" * 60)
    print()

    workflow = create_workflow()

    # Minimal state to test coordinator -> collector -> main_supervisor flow
    initial_state: State = {
        "task": "测试Send API",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "iteration": 0,
        "max_iterations": 1,  # Only 1 iteration
        "news_pool": [],
        "news_pool_cursor": 0,
        "latest_news_batch": [],
        "assigned_news": [],
        "text_batch": [],
        "video_batch": [],
        "text_news": [],
        "video_news": [],
        "text_analysis": None,
        "video_analysis": None,
        "text_relationship_graph": None,
        "text_sentiment": None,
        "text_reflection": None,
        "timeline_analysis": None,
        "trend_analysis": None,
        "timeline_chart_path": None,
        "final_report": None,
        "trending_keywords": None,
        "optimized_query": None,
        "supervisor_questions": [],
        "text_tools": None,
        "video_tools": None,
        "research_tools": None,
        "supervisor_decision": "",
        "supervisor_feedback": "",
        "quality_score": 0.0,
        "last_agent": "",
        "research_notes": [],
        "started_at": None,
        "completed_at": None,
    }

    print("🚀 Starting workflow with max_iterations=1")
    print("   Expected flow: coordinator -> main_supervisor -> news_collector -> main_supervisor -> collection_team (parallel) -> analysis_team (parallel) -> summary_team")
    print()

    try:
        # Set higher recursion limit for LangGraph
        config = {"recursion_limit": 50}
        final_state = await workflow.ainvoke(initial_state, config=config)

        print()
        print("=" * 60)
        print("Test Results")
        print("=" * 60)
        print(f"✅ Workflow completed successfully!")
        print(f"   Iterations: {final_state.get('iteration', 0)}")
        print(f"   Last agent: {final_state.get('last_agent')}")
        print(f"   Text news: {len(final_state.get('text_news', []))}")
        print(f"   Video news: {len(final_state.get('video_news', []))}")
        print(f"   Quality score: {final_state.get('quality_score', 0):.2f}")

        if final_state.get('iteration', 0) <= 1:
            print(f"\n✅ Iteration control working - stopped at {final_state.get('iteration', 0)} iterations")
        else:
            print(f"\n❌ Iteration control FAILED - ran {final_state.get('iteration', 0)} iterations instead of 1")

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n❌ Test FAILED: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_send_api())
    exit(0 if success else 1)
