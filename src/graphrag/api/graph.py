"""
Graph API Endpoints

Provides REST API for:
- Graph data retrieval
- News ingestion
- Entity/relationship queries
"""

import asyncio
import threading
import logging
from flask import request, jsonify

from . import graph_bp
from ..services.zep_client import ZepGraphClient
from ..agents.indexer import IndexerAgent
from ..agents.extractor import ExtractorAgent
from ..models.task import task_manager, TaskStatus
from ..config import get_config

logger = logging.getLogger(__name__)


def run_async(coro):
    """Helper to run async function in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@graph_bp.route('/data', methods=['GET'])
def get_graph_data():
    """
    Get graph data (nodes and edges).

    Query params:
    - graph_id: Optional graph ID (uses default if not provided)

    Returns:
    - nodes: List of graph nodes
    - edges: List of graph edges
    """
    try:
        graph_id = request.args.get('graph_id')

        client = ZepGraphClient()
        if not graph_id:
            graph_id = client.get_or_create_graph("news_graph")

        data = client.get_graph_data(graph_id)

        return jsonify({
            "success": True,
            "data": {
                "graph_id": data.graph_id,
                "name": data.name,
                "nodes": [
                    {
                        "uuid": n.uuid,
                        "name": n.name,
                        "labels": n.labels,
                        "summary": n.summary,
                        "attributes": n.attributes,
                        "created_at": n.created_at,
                    }
                    for n in data.nodes
                ],
                "edges": [
                    {
                        "uuid": e.uuid,
                        "name": e.name,
                        "fact": e.fact,
                        "source": e.source_node_uuid,
                        "target": e.target_node_uuid,
                        "created_at": e.created_at,
                    }
                    for e in data.edges
                ],
                "node_count": len(data.nodes),
                "edge_count": len(data.edges),
            }
        })

    except Exception as e:
        logger.error(f"Failed to get graph data: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@graph_bp.route('/entity/<entity_id>', methods=['GET'])
def get_entity(entity_id: str):
    """
    Get entity details by ID.

    Returns:
    - Entity information with related edges
    """
    try:
        graph_id = request.args.get('graph_id')

        client = ZepGraphClient()
        data = client.get_graph_data(graph_id)

        # Find entity
        entity = None
        for node in data.nodes:
            if node.uuid == entity_id:
                entity = node
                break

        if not entity:
            return jsonify({
                "success": False,
                "error": "Entity not found"
            }), 404

        # Find related edges
        related_edges = [
            e for e in data.edges
            if e.source_node_uuid == entity_id or e.target_node_uuid == entity_id
        ]

        return jsonify({
            "success": True,
            "data": {
                "entity": {
                    "uuid": entity.uuid,
                    "name": entity.name,
                    "labels": entity.labels,
                    "summary": entity.summary,
                    "attributes": entity.attributes,
                },
                "related_edges": [
                    {
                        "uuid": e.uuid,
                        "name": e.name,
                        "fact": e.fact,
                        "source": e.source_node_uuid,
                        "target": e.target_node_uuid,
                    }
                    for e in related_edges
                ]
            }
        })

    except Exception as e:
        logger.error(f"Failed to get entity: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@graph_bp.route('/timeline', methods=['GET'])
def get_timeline():
    """
    Get timeline data from graph.

    Query params:
    - graph_id: Optional graph ID
    - limit: Max items (default 50)

    Returns:
    - Timeline of events/facts sorted by time
    """
    try:
        graph_id = request.args.get('graph_id')
        limit = int(request.args.get('limit', 50))

        client = ZepGraphClient()
        data = client.get_graph_data(graph_id)

        # Sort edges by created_at
        timeline = sorted(
            data.edges,
            key=lambda e: e.created_at or "",
            reverse=True
        )[:limit]

        return jsonify({
            "success": True,
            "data": {
                "timeline": [
                    {
                        "uuid": e.uuid,
                        "fact": e.fact,
                        "relation": e.name,
                        "created_at": e.created_at,
                        "valid_at": e.valid_at,
                    }
                    for e in timeline
                ],
                "total": len(timeline)
            }
        })

    except Exception as e:
        logger.error(f"Failed to get timeline: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@graph_bp.route('/ingest', methods=['POST'])
def ingest_news():
    """
    Ingest news items into the graph.

    Request body:
    - news: List of news item objects
    - graph_id: Optional target graph ID
    - extract: Whether to use LLM extraction (default true)
    - async: Whether to run async (default false)

    Returns:
    - task_id if async
    - result if sync
    """
    try:
        data = request.get_json()
        news_items = data.get('news', [])
        graph_id = data.get('graph_id')
        extract = data.get('extract', True)
        run_async_flag = data.get('async', False)

        if not news_items:
            return jsonify({
                "success": False,
                "error": "No news items provided"
            }), 400

        if run_async_flag:
            # Create async task
            task_id = task_manager.create_task(f"Ingest {len(news_items)} news items")

            def ingest_task():
                try:
                    task_manager.update_task(
                        task_id,
                        status=TaskStatus.PROCESSING,
                        message="Starting ingestion..."
                    )

                    agent = IndexerAgent()

                    if extract:
                        # Extract first
                        task_manager.update_task(task_id, message="Extracting entities...")
                        extractor = ExtractorAgent()
                        extractions = run_async(extractor.extract_from_news_items(news_items))

                        task_manager.update_task(task_id, message="Indexing...", progress=50)
                        result = run_async(agent.index_extractions(extractions, graph_id, task_id))
                    else:
                        result = run_async(agent.index_news_items(news_items, graph_id, task_id))

                    task_manager.complete_task(
                        task_id,
                        result={
                            "indexed": result.indexed_count,
                            "failed": result.failed_count,
                            "graph_id": result.graph_id,
                        },
                        message=result.message
                    )

                except Exception as e:
                    task_manager.fail_task(task_id, str(e))

            thread = threading.Thread(target=ingest_task, daemon=True)
            thread.start()

            return jsonify({
                "success": True,
                "data": {
                    "task_id": task_id,
                    "message": "Ingestion started"
                }
            })

        else:
            # Sync execution
            agent = IndexerAgent()

            if extract:
                extractor = ExtractorAgent()
                extractions = run_async(extractor.extract_from_news_items(news_items))
                result = run_async(agent.index_extractions(extractions, graph_id))
            else:
                result = run_async(agent.index_news_items(news_items, graph_id))

            return jsonify({
                "success": True,
                "data": {
                    "indexed": result.indexed_count,
                    "failed": result.failed_count,
                    "graph_id": result.graph_id,
                    "message": result.message,
                }
            })

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@graph_bp.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id: str):
    """
    Get task status by ID.

    Returns:
    - Task status, progress, and result
    """
    task = task_manager.get_task(task_id)

    if not task:
        return jsonify({
            "success": False,
            "error": "Task not found"
        }), 404

    return jsonify({
        "success": True,
        "data": task.to_dict()
    })


@graph_bp.route('/search', methods=['GET'])
def search_graph():
    """
    Search the graph.

    Query params:
    - q: Search query
    - graph_id: Optional graph ID
    - limit: Max results (default 10)

    Returns:
    - Search results
    """
    try:
        query = request.args.get('q', '')
        graph_id = request.args.get('graph_id')
        limit = int(request.args.get('limit', 10))

        if not query:
            return jsonify({
                "success": False,
                "error": "Query parameter 'q' is required"
            }), 400

        client = ZepGraphClient()
        if not graph_id:
            graph_id = client.get_or_create_graph("news_graph")
        results = client.search(query, graph_id, limit)

        return jsonify({
            "success": True,
            "data": {
                "query": query,
                "results": results,
                "count": len(results)
            }
        })

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@graph_bp.route('/stats', methods=['GET'])
def get_graph_stats():
    """
    Get graph statistics.

    Returns:
    - Node count, edge count, etc.
    """
    try:
        graph_id = request.args.get('graph_id')

        client = ZepGraphClient()
        data = client.get_graph_data(graph_id)

        # Count by type
        node_types = {}
        for node in data.nodes:
            for label in node.labels:
                node_types[label] = node_types.get(label, 0) + 1

        edge_types = {}
        for edge in data.edges:
            edge_types[edge.name] = edge_types.get(edge.name, 0) + 1

        return jsonify({
            "success": True,
            "data": {
                "graph_id": data.graph_id,
                "node_count": len(data.nodes),
                "edge_count": len(data.edges),
                "node_types": node_types,
                "edge_types": edge_types,
            }
        })

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
