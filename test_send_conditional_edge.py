"""
测试：使用条件边实现子图内并行执行
"""
import asyncio
import logging
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.types import Send

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubgraphState(TypedDict):
    """子图状态"""
    input_data: List[str]
    results: List[str]


async def worker_a(state: SubgraphState):
    """工作节点 A"""
    logger.info("🔵 Worker A started")
    await asyncio.sleep(0.5)
    results = state.get("results", [])
    return {"results": results + ["A completed"]}


async def worker_b(state: SubgraphState):
    """工作节点 B"""
    logger.info("🟢 Worker B started")
    await asyncio.sleep(0.5)
    results = state.get("results", [])
    return {"results": results + ["B completed"]}


def coordinator(state: SubgraphState):
    """
    Coordinator: 准备数据，但不直接路由
    """
    logger.info("📍 Coordinator: Preparing data...")
    return {"input_data": state.get("input_data", [])}


def route_to_workers(state: SubgraphState):
    """
    条件边函数：返回 Send 列表实现并行
    关键：这是条件边函数，不是节点！
    """
    logger.info("🔀 Conditional edge: Sending to workers in parallel...")

    # 返回 Send 列表 - 这在条件边中是允许的！
    return [
        Send("worker_a", state),
        Send("worker_b", state),
    ]


def merger(state: SubgraphState):
    """合并结果"""
    results = state.get("results", [])
    logger.info(f"✅ Merger: Collected {len(results)} results: {results}")
    return {"results": results}


def create_test_subgraph():
    """创建测试子图（使用条件边）"""
    graph = StateGraph(SubgraphState)

    graph.add_node("coordinator", coordinator)
    graph.add_node("worker_a", worker_a)
    graph.add_node("worker_b", worker_b)
    graph.add_node("merger", merger)

    graph.set_entry_point("coordinator")

    # 关键：使用条件边从 coordinator 到 workers
    # 条件边函数可以返回 Send 列表！
    graph.add_conditional_edges(
        "coordinator",
        route_to_workers,  # 这个函数返回 Send 列表
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
    print("测试：条件边 + Send API 在子图中实现并行")
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

        if duration < 0.7:
            # 如果两个 worker 并行执行（各 0.5s），总时间应该接近 0.5s
            print("✅ 并行执行成功！")
            print(f"   预期时间: ~0.5s (并行)")
            print(f"   实际时间: {duration:.2f}s")
            return True
        else:
            # 如果串行执行，总时间应该接近 1.0s
            print("❌ 退化为串行执行")
            print(f"   预期时间: ~0.5s (并行)")
            print(f"   实际时间: {duration:.2f}s")
            return False

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_send_api())
    print("\n" + "=" * 60)
    if success:
        print("✅ 结论：条件边 + Send API 可以在子图中实现并行！")
    else:
        print("❌ 结论：并行执行失败")
    print("=" * 60)
