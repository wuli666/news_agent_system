"""
Zep Cloud Client for GraphRAG

Provides graph database operations:
- Graph creation and management
- Entity/relationship storage
- Semantic search and retrieval
"""

import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

try:
    from zep_cloud.client import Zep
    from zep_cloud.types import Message
    ZEP_AVAILABLE = True
except ImportError:
    ZEP_AVAILABLE = False
    Zep = None
    Message = None

from ..config import get_config, NEWS_ONTOLOGY

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    uuid: str
    name: str
    labels: List[str]
    summary: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None


@dataclass
class GraphData:
    """Complete graph data structure."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    graph_id: str
    name: str


class ZepGraphClient:
    """
    Zep Cloud client for knowledge graph operations.

    Provides methods for:
    - Creating and managing graphs
    - Adding news data as episodes
    - Retrieving graph data (nodes and edges)
    - Semantic search
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Zep client."""
        if not ZEP_AVAILABLE:
            raise ImportError(
                "zep-cloud package is not installed. "
                "Please install it with: pip install zep-cloud"
            )

        config = get_config()
        self.api_key = api_key or config.ZEP_API_KEY
        self.max_retries = config.MAX_RETRIES
        self.retry_delay = config.RETRY_DELAY
        self.batch_size = config.BATCH_SIZE

        if not self.api_key:
            raise ValueError("ZEP_API_KEY is required")

        # Create Zep client with httpx client that ignores proxy env vars
        import httpx
        httpx_client = httpx.Client(trust_env=False)
        self.client = Zep(api_key=self.api_key, httpx_client=httpx_client)
        self._current_graph_id: Optional[str] = None

    def _call_with_retry(
        self,
        func: Callable,
        operation_name: str,
        max_retries: Optional[int] = None
    ) -> Any:
        """Execute API call with retry logic."""
        max_retries = max_retries or self.max_retries
        delay = self.retry_delay

        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                logger.warning(
                    f"{operation_name} attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"{operation_name} failed after {max_retries} attempts")
                    raise

    def create_graph(self, name: str, graph_id: Optional[str] = None) -> str:
        """
        Create a new graph in Zep Cloud.

        Args:
            name: Graph name
            graph_id: Optional graph identifier (auto-generated if not provided)

        Returns:
            Graph ID
        """
        import uuid
        graph_id = graph_id or f"news_agent_{uuid.uuid4().hex[:16]}"

        def _create():
            # Use graph.create API
            self.client.graph.create(
                graph_id=graph_id,
                name=name,
                description=f"News Agent Knowledge Graph - {name}"
            )
            return graph_id

        try:
            created_id = self._call_with_retry(_create, "create_graph")
            self._current_graph_id = created_id
            logger.info(f"Created graph: {created_id}")
            return created_id
        except Exception as e:
            # If graph already exists, just use it
            if "already exists" in str(e).lower():
                self._current_graph_id = graph_id
                logger.info(f"Using existing graph: {graph_id}")
                return graph_id
            raise

    def get_or_create_graph(self, name: str = "news_graph") -> str:
        """Get existing graph or create new one."""
        graph_id = f"news_agent_{name}"
        try:
            # Try to get the graph
            self.client.graph.get(graph_id=graph_id)
            self._current_graph_id = graph_id
            logger.info(f"Using existing graph: {graph_id}")
            return graph_id
        except Exception:
            # Graph doesn't exist, create it
            return self.create_graph(name, graph_id=graph_id)

    def add_episode(
        self,
        content: str,
        graph_id: Optional[str] = None,
        source: str = "news",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a single episode (news content) to the graph.

        Zep automatically extracts entities and relationships from the content.

        Args:
            content: Text content to add
            graph_id: Target graph ID (uses current if not specified)
            source: Content source identifier
            metadata: Additional metadata

        Returns:
            Success status
        """
        graph_id = graph_id or self._current_graph_id
        if not graph_id:
            raise ValueError("No graph_id specified and no current graph set")

        def _add():
            # Use graph API instead of memory API
            self.client.graph.add(
                graph_id=graph_id,
                type='text',
                data=content
            )
            return True

        return self._call_with_retry(_add, "add_episode")

    def add_episodes_batch(
        self,
        contents: List[str],
        graph_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """
        Add multiple episodes in batches.

        Args:
            contents: List of text contents to add
            graph_id: Target graph ID
            progress_callback: Optional callback(current, total)

        Returns:
            Number of successfully added episodes
        """
        graph_id = graph_id or self._current_graph_id
        if not graph_id:
            raise ValueError("No graph_id specified")

        from zep_cloud import EpisodeData

        total = len(contents)
        success_count = 0

        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = contents[i:i + self.batch_size]

            try:
                # Create EpisodeData objects
                episodes = [EpisodeData(data=content, type='text') for content in batch]

                # Use add_batch API
                def _add_batch():
                    return self.client.graph.add_batch(
                        graph_id=graph_id,
                        episodes=episodes
                    )

                result = self._call_with_retry(_add_batch, f"add_batch_{i}")
                success_count += len(batch)

                if progress_callback:
                    progress_callback(i + len(batch), total)

            except Exception as e:
                logger.error(f"Failed to add batch {i}-{i+len(batch)}: {e}")

            # Small delay between batches to avoid rate limiting
            if i + self.batch_size < total:
                time.sleep(0.5)

        logger.info(f"Added {success_count}/{total} episodes to graph {graph_id}")
        return success_count

    def add_news_items(
        self,
        news_items: List[Dict[str, Any]],
        graph_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """
        Add news items to the graph.

        Converts news items to formatted text for entity extraction.

        Args:
            news_items: List of news item dictionaries
            graph_id: Target graph ID
            progress_callback: Optional progress callback

        Returns:
            Number of successfully added items
        """
        contents = []
        for item in news_items:
            # Format news item as structured text for better entity extraction
            text_parts = []

            title = item.get("title", "")
            if title:
                text_parts.append(f"标题：{title}")

            # 不添加来源信息，避免"百度热搜"等成为中心节点
            # source = item.get("source", item.get("platform", ""))
            # if source:
            #     text_parts.append(f"来源：{source}")

            # 不添加热度信息，避免噪声
            # hot_score = item.get("hot", item.get("hot_score", ""))
            # if hot_score:
            #     text_parts.append(f"热度：{hot_score}")

            url = item.get("url", "")
            if url:
                text_parts.append(f"链接：{url}")

            # Add timestamp if available
            timestamp = item.get("timestamp", item.get("published_at", ""))
            if timestamp:
                text_parts.append(f"时间：{timestamp}")

            # Add any additional content
            content = item.get("content", item.get("description", ""))
            if content:
                text_parts.append(f"内容：{content}")

            if text_parts:
                contents.append("\n".join(text_parts))

        return self.add_episodes_batch(contents, graph_id, progress_callback)

    def get_graph_data(self, graph_id: Optional[str] = None) -> GraphData:
        """
        Get complete graph data including nodes and edges.

        Args:
            graph_id: Graph ID to query

        Returns:
            GraphData with nodes and edges
        """
        graph_id = graph_id or self._current_graph_id
        if not graph_id:
            raise ValueError("No graph_id specified")

        nodes = []
        edges = []

        try:
            # Get nodes using get_by_graph_id (returns proper node names)
            def _get_nodes():
                return self.client.graph.node.get_by_graph_id(
                    graph_id=graph_id,
                    limit=10000
                )

            raw_nodes = self._call_with_retry(_get_nodes, "get_nodes")
            if raw_nodes:
                for node in raw_nodes:
                    label = getattr(node, 'label', 'Entity') or 'Entity'
                    nodes.append(GraphNode(
                        uuid=getattr(node, 'uuid_', getattr(node, 'uuid', str(hash(str(node))))),
                        name=getattr(node, 'name', 'Unknown'),
                        labels=[label] if isinstance(label, str) else (label or ['Entity']),
                        summary=getattr(node, 'summary', '') or '',
                        created_at=str(getattr(node, 'created_at', '')),
                    ))

            # Get edges using get_by_graph_id
            def _get_edges():
                return self.client.graph.edge.get_by_graph_id(
                    graph_id=graph_id,
                    limit=10000
                )

            raw_edges = self._call_with_retry(_get_edges, "get_edges")
            if raw_edges:
                for edge in raw_edges:
                    edges.append(GraphEdge(
                        uuid=getattr(edge, 'uuid_', getattr(edge, 'uuid', str(hash(str(edge))))),
                        name=getattr(edge, 'name', 'RELATED_TO'),
                        fact=getattr(edge, 'fact', ''),
                        source_node_uuid=getattr(edge, 'source_node_uuid', ''),
                        target_node_uuid=getattr(edge, 'target_node_uuid', ''),
                        created_at=str(getattr(edge, 'created_at', '')),
                        valid_at=str(getattr(edge, 'valid_at', '')),
                    ))

            logger.info(f"Graph data retrieved: {len(nodes)} nodes, {len(edges)} edges")

        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")

        return GraphData(
            nodes=nodes,
            edges=edges,
            graph_id=graph_id,
            name=graph_id
        )

    def search(
        self,
        query: str,
        graph_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Semantic search in the graph.

        Args:
            query: Search query
            graph_id: Graph to search in
            limit: Maximum results

        Returns:
            List of search results
        """
        graph_id = graph_id or self._current_graph_id
        if not graph_id:
            raise ValueError("No graph_id specified")

        def _search():
            # Use graph search API
            results = self.client.graph.search(
                graph_id=graph_id,
                query=query,
                limit=limit
            )
            return results

        try:
            results = self._call_with_retry(_search, "search")

            formatted_results = []
            if results:
                # Zep search returns object with edges and nodes attributes
                edges = getattr(results, 'edges', None) or []
                nodes = getattr(results, 'nodes', None) or []

                # Process edges (facts/relations)
                for edge in edges:
                    formatted_results.append({
                        "type": "edge",
                        "fact": getattr(edge, 'fact', ''),
                        "name": getattr(edge, 'name', 'RELATED_TO'),
                        "score": getattr(edge, 'score', 0.0),
                        "source_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_uuid": getattr(edge, 'target_node_uuid', ''),
                        "created_at": str(getattr(edge, 'created_at', '')),
                    })

                # Process nodes (entities)
                for node in nodes:
                    formatted_results.append({
                        "type": "node",
                        "name": getattr(node, 'name', ''),
                        "summary": getattr(node, 'summary', ''),
                        "label": getattr(node, 'label', 'Entity'),
                        "uuid": getattr(node, 'uuid_', getattr(node, 'uuid', '')),
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_facts(
        self,
        query: str,
        graph_id: Optional[str] = None,
        limit: int = 10
    ) -> List[str]:
        """
        Search for relevant facts in the graph.

        Args:
            query: Search query
            graph_id: Graph to search
            limit: Maximum results

        Returns:
            List of fact strings
        """
        results = self.search(query, graph_id, limit)
        return [r.get("content", "") for r in results if r.get("content")]

    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete a graph and all its data.

        Args:
            graph_id: Graph ID to delete

        Returns:
            Success status
        """
        def _delete():
            self.client.graph.delete(graph_id=graph_id)
            return True

        try:
            result = self._call_with_retry(_delete, "delete_graph")
            if self._current_graph_id == graph_id:
                self._current_graph_id = None
            logger.info(f"Deleted graph: {graph_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete graph: {e}")
            return False

    @property
    def current_graph_id(self) -> Optional[str]:
        """Get the current active graph ID."""
        return self._current_graph_id
