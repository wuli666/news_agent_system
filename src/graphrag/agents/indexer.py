"""
Indexer Agent

Stores extracted entities and relationships into Zep Cloud graph.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .extractor import ExtractionResult, ExtractedEntity, ExtractedRelation
from ..services.zep_client import ZepGraphClient
from ..models.task import task_manager, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of indexing operation."""
    success: bool
    indexed_count: int
    failed_count: int
    graph_id: str
    message: str = ""


class IndexerAgent:
    """
    Indexer Agent for storing extracted data in Zep.

    Handles:
    - Converting extraction results to Zep episodes
    - Batch indexing with progress tracking
    - Deduplication logic
    """

    def __init__(self, zep_client: Optional[ZepGraphClient] = None):
        """
        Initialize indexer agent.

        Args:
            zep_client: Optional ZepGraphClient instance
        """
        self.zep_client = zep_client
        self._seen_entities: set = set()

    def _get_client(self) -> ZepGraphClient:
        """Get or create Zep client."""
        if self.zep_client is None:
            self.zep_client = ZepGraphClient()
        return self.zep_client

    def _extraction_to_episode_text(self, result: ExtractionResult) -> str:
        """
        Convert extraction result to text format for Zep episode.

        Zep will automatically extract entities and relations from the text.
        We format it to help the extraction.
        """
        lines = []

        # Add source text context
        if result.metadata.get("title"):
            lines.append(f"新闻标题：{result.metadata['title']}")

        # Add entities section
        if result.entities:
            entity_descriptions = []
            for e in result.entities:
                desc = f"{e.name}（{e.type}）"
                if e.attributes:
                    attrs = ", ".join(f"{k}: {v}" for k, v in e.attributes.items())
                    desc += f" - {attrs}"
                entity_descriptions.append(desc)

            lines.append(f"涉及实体：{'; '.join(entity_descriptions)}")

        # Add relations section
        if result.relations:
            relation_descriptions = []
            for r in result.relations:
                desc = f"{r.source} {r.relation_type} {r.target}"
                if r.description:
                    desc += f"（{r.description}）"
                relation_descriptions.append(desc)

            lines.append(f"实体关系：{'; '.join(relation_descriptions)}")

        # Add original text snippet
        if result.source_text:
            # Truncate if too long
            text = result.source_text[:500]
            if len(result.source_text) > 500:
                text += "..."
            lines.append(f"原文摘要：{text}")

        return "\n".join(lines)

    async def index_extraction(
        self,
        result: ExtractionResult,
        graph_id: Optional[str] = None
    ) -> bool:
        """
        Index a single extraction result.

        Args:
            result: ExtractionResult to index
            graph_id: Target graph ID

        Returns:
            Success status
        """
        try:
            client = self._get_client()

            # Ensure graph exists
            if graph_id:
                client._current_graph_id = graph_id
            elif not client.current_graph_id:
                client.get_or_create_graph("news_graph")

            # Convert to episode text
            episode_text = self._extraction_to_episode_text(result)

            # Add to graph
            success = client.add_episode(
                content=episode_text,
                source="news_extraction",
                metadata=result.metadata
            )

            return success

        except Exception as e:
            logger.error(f"Failed to index extraction: {e}")
            return False

    async def index_extractions(
        self,
        results: List[ExtractionResult],
        graph_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> IndexResult:
        """
        Index multiple extraction results.

        Args:
            results: List of ExtractionResult objects
            graph_id: Target graph ID
            task_id: Optional task ID for progress tracking

        Returns:
            IndexResult summary
        """
        client = self._get_client()

        # Ensure graph exists
        if graph_id:
            client._current_graph_id = graph_id
        elif not client.current_graph_id:
            graph_id = client.get_or_create_graph("news_graph")
        else:
            graph_id = client.current_graph_id

        indexed = 0
        failed = 0
        total = len(results)

        for i, result in enumerate(results):
            try:
                success = await self.index_extraction(result, graph_id)
                if success:
                    indexed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to index result {i}: {e}")
                failed += 1

            # Update progress if task_id provided
            if task_id:
                progress = int((i + 1) / total * 100)
                task_manager.update_task(
                    task_id,
                    progress=progress,
                    message=f"Indexed {i + 1}/{total} items"
                )

        message = f"Indexed {indexed} items, {failed} failed"
        logger.info(message)

        return IndexResult(
            success=failed == 0,
            indexed_count=indexed,
            failed_count=failed,
            graph_id=graph_id,
            message=message
        )

    async def index_news_items(
        self,
        news_items: List[Dict[str, Any]],
        graph_id: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> IndexResult:
        """
        Index news items directly (without pre-extraction).

        This uses Zep's built-in entity extraction.

        Args:
            news_items: List of news item dictionaries
            graph_id: Target graph ID
            task_id: Optional task ID for progress tracking

        Returns:
            IndexResult summary
        """
        client = self._get_client()

        # Ensure graph exists
        if graph_id:
            client._current_graph_id = graph_id
        elif not client.current_graph_id:
            graph_id = client.get_or_create_graph("news_graph")
        else:
            graph_id = client.current_graph_id

        # Define progress callback
        def progress_callback(current: int, total: int):
            if task_id:
                progress = int(current / total * 100)
                task_manager.update_task(
                    task_id,
                    progress=progress,
                    message=f"Indexed {current}/{total} news items"
                )

        # Use Zep client's batch add
        indexed = client.add_news_items(
            news_items,
            graph_id,
            progress_callback
        )

        failed = len(news_items) - indexed

        return IndexResult(
            success=failed == 0,
            indexed_count=indexed,
            failed_count=failed,
            graph_id=graph_id,
            message=f"Indexed {indexed} items, {failed} failed"
        )


async def index_news(
    news_items: List[Dict[str, Any]],
    graph_id: Optional[str] = None,
    zep_client: Optional[ZepGraphClient] = None
) -> IndexResult:
    """
    Convenience function to index news items.

    Args:
        news_items: List of news item dictionaries
        graph_id: Optional target graph ID
        zep_client: Optional ZepGraphClient instance

    Returns:
        IndexResult
    """
    agent = IndexerAgent(zep_client=zep_client)
    return await agent.index_news_items(news_items, graph_id)
