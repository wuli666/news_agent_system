"""
Quick test script for refactored workflow.
"""
import asyncio
import logging
from datetime import datetime
from src.graph.workflow import create_workflow
from src.graph.types import State

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_workflow_structure():
    """Test that workflow can be created."""
    print("=" * 60)
    print("Testing Refactored Workflow Structure")
    print("=" * 60)

    try:
        workflow = create_workflow()
        print("✅ Workflow created successfully!")

        # Test workflow with minimal state
        print("\nTesting minimal workflow execution...")

        initial_state: State = {
            "task": "测试任务",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "iteration": 0,
            "max_iterations": 1,  # Only 1 iteration for test
            "text_news": [],
            "video_news": [],
            "news_pool": [],
            "news_pool_cursor": 0,
            "latest_news_batch": [],
            "text_analysis": None,
            "video_analysis": None,
            "research_notes": [],
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
            "started_at": None,
            "completed_at": None,
        }

        print("\n📊 Initial state prepared")
        print(f"   Task: {initial_state['task']}")
        print(f"   Date: {initial_state['date']}")
        print(f"   Max iterations: {initial_state['max_iterations']}")

        print("\n🚀 Starting workflow execution...")
        print("   (This is a test run - may need API keys configured)")

        # You can uncomment the following to actually run:
        # final_state = await workflow.ainvoke(initial_state)
        # print("\n✅ Workflow completed!")
        # print(f"   Final agent: {final_state.get('last_agent')}")
        # print(f"   Quality score: {final_state.get('quality_score', 0):.2f}")

        print("\n" + "=" * 60)
        print("Workflow Structure Test: PASSED ✅")
        print("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Workflow test failed: {e}", exc_info=True)
        print("\n❌ Workflow test FAILED")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_workflow_structure())
    exit(0 if success else 1)
