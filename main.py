"""
Multi-Agent News Collection System - Main Entry Point
"""
import asyncio
import logging
import sys
from datetime import datetime
from src.graph.workflow import create_workflow  # Using refactored workflow
from src.config.settings import settings
from src.graph.types import State

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_news_collection(
    task: str = "获取今日最热新闻",
    date: str = None,
    max_iterations: int = None
):
    """
    Run the news collection system.
    
    Args:
        task: Task description
        date: Target date (defaults to today)
        max_iterations: Maximum iterations (defaults to config)
    """
    print("=" * 60)
    print("Multi-Agent News Collection System")
    print("=" * 60)
    print()
    
    # Display configuration
    settings.display()
    
    # Create workflow
    logger.info("Initializing workflow...")
    workflow = create_workflow()
    logger.info("Workflow initialized successfully")
    print()
    
    # Prepare initial state
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    initial_state: State = {
        "task": task,
        "date": date,
        "iteration": 0,
        "max_iterations": max_iterations or settings.MAX_ITERATIONS,
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
        "information_gaps": [],
        "text_team_cycle_count": 0,
        "analysis_gap_pending": None,
        "gap_fill_mode": False,
        "skip_sentiment_relationship": False,
        "daily_papers": [],
        "text_tools": None,  # Will be preloaded by coordinator
        "video_tools": None,  # Will be preloaded by coordinator
        "research_tools": None,  # Will be preloaded by coordinator
        "supervisor_decision": "",
        "supervisor_feedback": "",
        "quality_score": 0.0,
        "last_agent": "",
        "research_notes": [],
        "news_images": {},
        "started_at": None,
        "completed_at": None,
    }
    
    print(f"Task: {task}")
    print(f"Date: {date}")
    print(f"Max Iterations: {initial_state['max_iterations']}")
    print()
    print("-" * 60)
    print()
    
    try:
        # Run workflow with higher recursion limit
        # LangGraph 的 recursion_limit 按节点调用计数，分析内层循环一轮就有
        # research/sentiment/relationship/reflect 等 4+ 节点，叠加 3 轮内层
        # 和多次外层迭代，需给更高上限避免误报。
        config = {
            "recursion_limit": max(100, initial_state["max_iterations"] * 30)
        }
        final_state = await workflow.ainvoke(initial_state, config=config)
        
        # Display results
        print()
        print("=" * 60)
        print("Execution Completed!")
        print("=" * 60)
        print()
        
        if final_state.get("final_report"):
            print(final_state["final_report"])
        
        # Display statistics
        print()
        print("=" * 60)
        print("Statistics")
        print("=" * 60)
        print(f"Total Iterations: {final_state.get('iteration', 0)}")
        print(f"Text News Collected: {len(final_state.get('text_news', []))}")
        print(f"Video News Collected: {len(final_state.get('video_news', []))}")
        print(f"Final Quality Score: {final_state.get('quality_score', 0.0):.2f}")
        
        if final_state.get("started_at") and final_state.get("completed_at"):
            duration = final_state["completed_at"] - final_state["started_at"]
            print(f"Duration: {duration.total_seconds():.2f} seconds")
        
        return final_state
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent News Collection System"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="获取今日最热新闻",
        help="Task description"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations"
    )
    
    args = parser.parse_args()
    
    # Run system
    asyncio.run(run_news_collection(
        task=args.task,
        date=args.date,
        max_iterations=args.max_iterations
    ))


if __name__ == "__main__":
    main()
