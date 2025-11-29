"""
测试并行执行优化
"""
import asyncio
import logging
import time
from datetime import datetime
from src.graph.workflow import create_workflow
from src.graph.types import State

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_parallel_execution():
    """测试并行执行性能"""
    print("\n" + "=" * 60)
    print("并行执行性能测试")
    print("=" * 60)
    print()

    workflow = create_workflow()

    # 最小化测试状态
    initial_state: State = {
        "task": "测试并行执行",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "iteration": 0,
        "max_iterations": 1,  # 只运行1轮
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
        "text_tools": None,
        "video_tools": None,
        "research_tools": None,
        "supervisor_decision": "",
        "supervisor_feedback": "",
        "quality_score": 0.0,
        "last_agent": "",
        "research_notes": [],
        "news_images": {},
        "started_at": None,
        "completed_at": None,
    }

    print("🚀 开始执行工作流...")
    print("   预期行为:")
    print("   - Collection Team: text_collector ∥ video_collector (并行)")
    print("   - Analysis Team: research ∥ sentiment ∥ relationship (并行)")
    print()

    try:
        start_time = time.time()

        config = {"recursion_limit": 100}
        final_state = await workflow.ainvoke(initial_state, config=config)

        duration = time.time() - start_time

        print()
        print("=" * 60)
        print("测试结果")
        print("=" * 60)
        print(f"✅ 工作流执行完成！")
        print(f"   总执行时间: {duration:.2f}s")
        print(f"   迭代次数: {final_state.get('iteration', 0)}")
        print(f"   文本新闻: {len(final_state.get('text_news', []))}")
        print(f"   视频新闻: {len(final_state.get('video_news', []))}")
        print(f"   研究笔记: {len(final_state.get('research_notes', []))}")
        print(f"   质量分数: {final_state.get('quality_score', 0):.2f}")

        if final_state.get("final_report"):
            print(f"\n📝 生成了最终报告")

        print()
        print("=" * 60)
        print("并行执行验证")
        print("=" * 60)
        print("✅ 检查日志中是否出现以下标记:")
        print("   🔀 Parallel routing: ... (Collection Team)")
        print("   🔀 Full analysis: research ∥ sentiment ∥ relationship (Analysis Team)")
        print()

        return True

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        print(f"\n❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_parallel_execution())
    print("\n" + "=" * 60)
    if success:
        print("✅ 并行执行测试通过！")
        print("   建议: 检查日志输出，确认看到并行路由标记")
    else:
        print("❌ 并行执行测试失败")
        print("   建议: 检查错误日志")
    print("=" * 60)
