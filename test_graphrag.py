#!/usr/bin/env python3
"""
Quick test script for GraphRAG integration
"""
import asyncio
import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.graphrag.services.zep_client import ZepGraphClient
from src.graphrag.config import get_config
from src.utils.logger import setup_colorful_logger, print_stage_header, print_success, print_error, print_info

setup_colorful_logger("INFO", verbose=False)


def test_graph_data():
    """Test retrieving graph data from existing graph."""
    print_stage_header("Graph Data Test", "Checking existing graph content")

    config = get_config()
    client = ZepGraphClient(api_key=config.ZEP_API_KEY)

    graph_id = "news_agent_news_graph"
    print_info(f"Checking graph: {graph_id}")

    try:
        graph_data = client.get_graph_data(graph_id)
        print_success(f"Nodes: {len(graph_data.nodes)}")
        print_success(f"Edges: {len(graph_data.edges)}")

        if graph_data.nodes:
            print_info("Sample nodes:")
            for node in graph_data.nodes[:5]:
                labels = ", ".join(node.labels) if node.labels else "Entity"
                print(f"  - {node.name} ({labels})")

        if graph_data.edges:
            print_info("Sample edges:")
            for edge in graph_data.edges[:5]:
                print(f"  - {edge.fact[:80]}...")

        return True
    except Exception as e:
        print_error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_zep_connection():
    """Test Zep Cloud connection and basic operations."""
    print_stage_header("GraphRAG Test", "Testing Zep Cloud connection")

    try:
        # Initialize client
        print_info("🔌 Initializing Zep client...")
        config = get_config()
        client = ZepGraphClient(api_key=config.ZEP_API_KEY)
        print_success("Zep client initialized")

        # Create or get graph
        print_info("📊 Creating/getting knowledge graph...")
        graph_id = client.get_or_create_graph("test_graph")
        print_success(f"Graph ready: {graph_id}")

        # Add test episode
        print_info("📝 Adding test news episode...")
        test_news = """
标题：人工智能技术突破
来源：科技日报
热度：9876
时间：2026-01-26
内容：科技公司发布新一代大语言模型，在多项基准测试中刷新记录。
"""
        success = client.add_episode(
            content=test_news,
            graph_id=graph_id,
            source="test"
        )

        if success:
            print_success("Test episode added successfully")
        else:
            print_error("Failed to add test episode")
            return False

        # Try to retrieve graph data
        print_info("🔍 Retrieving graph data...")
        graph_data = client.get_graph_data(graph_id)
        print_success(f"Graph retrieved - Nodes: {len(graph_data.nodes)}, Edges: {len(graph_data.edges)}")

        # Try search
        print_info("🔎 Testing semantic search...")
        results = client.search("人工智能", graph_id=graph_id, limit=5)
        print_success(f"Search completed - Found {len(results)} results")

        if results:
            print_info("📄 First result:")
            print(f"  Content: {results[0].get('content', '')[:100]}...")
            print(f"  Score: {results[0].get('score', 0)}")

        print_success("✅ All tests passed! GraphRAG is ready to use.")
        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check existing graph data")
    args = parser.parse_args()

    if args.check:
        result = test_graph_data()
    else:
        result = asyncio.run(test_zep_connection())

    sys.exit(0 if result else 1)
