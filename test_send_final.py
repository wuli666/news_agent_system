"""
✅ 最终方案：条件边 + Send API + Annotated 处理并发更新
"""
import asyncio
import logging
import operator
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.types import Send

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubgraphState(TypedDict):
    """子图状态 - 使用 Annotated 处理并发更新"""
    input_data: List[str]
    # 关键：使用 Annotated[list, operator.add] 支持并发追加
    results: Annotated[List[str], operator.add]


async def worker_a(state: SubgraphState):
    """工作节点 A"""
    logger.info("🔵 Worker A started")
    await asyncio.sleep(0.5)
    # 返回新的列表项（会被 add 到现有列表）
    return {"results": ["A completed"]}


async def worker_b(state: SubgraphState):
    """工作节点 B"""
    logger.info("🟢 Worker B started")
    await asyncio.sleep(0.5)
    # 返回新的列表项（会被 add 到现有列表）
    return {"results": ["B completed"]}


def coordinator(state: SubgraphState):
    """Coordinator: 准备数据"""
    logger.info("📍 Coordinator: Preparing data...")
    return {}


def route_to_workers(state: SubgraphState):
    """
    条件边函数：返回 Send 列表实现并行
    ✅ 这在条件边中是允许的！
    """
    logger.info("🔀 Conditional edge: Sending to workers in parallel...")
    return [
        Send("worker_a", state),
        Send("worker_b", state),
    ]


def merger(state: SubgraphState):
    """合并结果"""
    results = state.get("results", [])
    logger.info(f"✅ Merger: Collected {len(results)} results: {results}")
    return {}


def create_test_subgraph():
    """创建测试子图（正确方案）"""
    graph = StateGraph(SubgraphState)

    graph.add_node("coordinator", coordinator)
    graph.add_node("worker_a", worker_a)
    graph.add_node("worker_b", worker_b)
    graph.add_node("merger", merger)

    graph.set_entry_point("coordinator")

    # ✅ 使用条件边从 coordinator 并行分发到 workers
    graph.add_conditional_edges(
        "coordinator",
        route_to_workers,
    )

    # Workers 完成后到 merger
    graph.add_edge("worker_a", "merger")
    graph.add_edge("worker_b", "merger")

    # Merger 结束
    graph.add_edge("merger", END)

    return graph.compile()


async def test_send_api():
    """测试函数"""
    print("\n" + "=" * 60)
    print("✅ 最终方案测试：条件边 + Send + Annotated")
    print("=" * 60)

    subgraph = create_test_subgraph()

    initial_state: SubgraphState = {
        "input_data": ["task1", "task2"],
        "results": [],
    }

    try:
        import time
        start = time.time()

        final_state = await subgraph.ainvoke(initial_state)

        duration = time.time() - start

        print("\n" + "=" * 60)
        print("测试结果")
        print("=" * 60)
        print(f"执行时间: {duration:.2f}s")
        print(f"结果: {final_state.get('results', [])}")
        print(f"结果数量: {len(final_state.get('results', []))}")

        if duration < 0.7 and len(final_state.get("results", [])) == 2:
            print("\n✅✅✅ 并行执行成功！")
            print(f"   预期时间: ~0.5s (并行), 实际: {duration:.2f}s")
            print(f"   预期结果: 2 项, 实际: {len(final_state.get('results', []))} 项")
            return True
        else:
            print(f"\n❌ 有问题")
            print(f"   执行时间: {duration:.2f}s (预期 ~0.5s)")
            print(f"   结果数量: {len(final_state.get('results', []))} (预期 2)")
            return False

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_send_api())
    print("\n" + "=" * 60)
    if success:
        print("🎉🎉🎉 完美！子图中并行执行的正确方法：")
        print("   1. 使用条件边（不是节点返回值）")
        print("   2. 条件边函数返回 Send 列表")
        print("   3. State 中需要合并的字段用 Annotated[List, operator.add]")
    else:
        print("❌ 测试失败")
    print("=" * 60)
