"""
Retriever Agent

Multi-dimensional graph search inspired by MiroFish's ZepTools.
Provides three search modes:
- InsightForge: Deep analysis with sub-query decomposition
- PanoramaSearch: Broad context gathering
- QuickSearch: Fast keyword lookup
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from ..services.zep_client import ZepGraphClient

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result from graph."""
    content: str
    score: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsightResult:
    """Deep insight result."""
    query: str
    sub_queries: List[str]
    facts: List[str]
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    summary: str = ""


@dataclass
class RetrievalResult:
    """Combined retrieval result."""
    query: str
    results: List[SearchResult]
    insights: Optional[InsightResult] = None
    mode: str = "quick"


DECOMPOSE_PROMPT = """你是一个查询分解专家。给定一个复杂问题，将其分解为2-4个更简单的子问题，以便从知识图谱中检索相关信息。

原始问题：{query}

请输出子问题列表，每行一个问题。只输出问题，不要编号或其他内容。"""


class RetrieverAgent:
    """
    Multi-modal Graph Retriever Agent.

    Provides three search strategies:
    1. InsightForge - Deep analysis with query decomposition
    2. PanoramaSearch - Broad context search
    3. QuickSearch - Fast direct search
    """

    def __init__(
        self,
        zep_client: Optional[ZepGraphClient] = None,
        llm=None
    ):
        """
        Initialize retriever agent.

        Args:
            zep_client: Optional ZepGraphClient
            llm: Optional LLM for query decomposition
        """
        self.zep_client = zep_client
        self.llm = llm
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM if not provided."""
        if self.llm is None:
            from src.llms.llm import get_llm_by_type
            self.llm = get_llm_by_type("qwen")

    def _get_client(self) -> ZepGraphClient:
        """Get or create Zep client."""
        if self.zep_client is None:
            self.zep_client = ZepGraphClient()
        return self.zep_client

    async def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries.

        Args:
            query: Original query

        Returns:
            List of sub-queries
        """
        if self.llm is None:
            return [query]

        try:
            prompt = DECOMPOSE_PROMPT.format(query=query)
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse sub-queries (one per line)
            sub_queries = [
                q.strip()
                for q in response_text.strip().split('\n')
                if q.strip()
            ]

            return sub_queries if sub_queries else [query]

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]

    async def quick_search(
        self,
        query: str,
        graph_id: Optional[str] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """
        Fast direct search in graph.

        Args:
            query: Search query
            graph_id: Target graph ID
            limit: Max results

        Returns:
            RetrievalResult
        """
        client = self._get_client()

        try:
            raw_results = client.search(query, graph_id, limit)

            results = []
            for r in raw_results:
                # Handle both edge results (with 'fact') and node results (with 'name'/'summary')
                if r.get("type") == "edge":
                    content = r.get("fact", "")
                elif r.get("type") == "node":
                    content = f"{r.get('name', '')}: {r.get('summary', '')}"
                else:
                    content = r.get("fact", r.get("summary", r.get("content", "")))

                if content:
                    results.append(SearchResult(
                        content=content,
                        score=r.get("score", 0.0),
                        source=r.get("type", "unknown"),
                        metadata=r
                    ))

            return RetrievalResult(
                query=query,
                results=results,
                mode="quick"
            )

        except Exception as e:
            logger.error(f"Quick search failed: {e}")
            return RetrievalResult(query=query, results=[], mode="quick")

    async def panorama_search(
        self,
        query: str,
        graph_id: Optional[str] = None,
        limit: int = 20
    ) -> RetrievalResult:
        """
        Broad context search for comprehensive coverage.

        Args:
            query: Search query
            graph_id: Target graph ID
            limit: Max results

        Returns:
            RetrievalResult with broader context
        """
        client = self._get_client()

        all_results = []

        try:
            # Search with original query
            raw_results = client.search(query, graph_id, limit)

            for r in raw_results:
                # Extract content based on result type
                if r.get("type") == "edge":
                    content = r.get("fact", "")
                elif r.get("type") == "node":
                    content = f"{r.get('name', '')}: {r.get('summary', '')}"
                else:
                    content = r.get("fact", r.get("summary", ""))

                if content:
                    all_results.append(SearchResult(
                        content=content,
                        score=r.get("score", 0.0),
                        source=r.get("type", "direct"),
                        metadata=r
                    ))

            return RetrievalResult(
                query=query,
                results=all_results,
                mode="panorama"
            )

        except Exception as e:
            logger.error(f"Panorama search failed: {e}")
            return RetrievalResult(query=query, results=[], mode="panorama")

    async def insight_forge(
        self,
        query: str,
        graph_id: Optional[str] = None,
        limit: int = 15
    ) -> RetrievalResult:
        """
        Deep analysis with query decomposition.

        Inspired by MiroFish's InsightForge tool.

        Args:
            query: Complex query
            graph_id: Target graph ID
            limit: Max results per sub-query

        Returns:
            RetrievalResult with InsightResult
        """
        client = self._get_client()

        # Decompose query
        sub_queries = await self._decompose_query(query)
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries")

        all_results = []
        all_facts = []

        try:
            # Search for each sub-query
            for sub_q in sub_queries:
                raw_results = client.search(sub_q, graph_id, limit // len(sub_queries))

                for r in raw_results:
                    # Extract content based on result type
                    if r.get("type") == "edge":
                        content = r.get("fact", "")
                        if content:
                            all_facts.append(content)
                    elif r.get("type") == "node":
                        content = f"{r.get('name', '')}: {r.get('summary', '')}"
                    else:
                        content = r.get("fact", r.get("summary", ""))

                    if content and content not in [res.content for res in all_results]:
                        all_results.append(SearchResult(
                            content=content,
                            score=r.get("score", 0.0),
                            source=f"sub_query: {sub_q}",
                            metadata=r
                        ))

            # Build insight result
            insight = InsightResult(
                query=query,
                sub_queries=sub_queries,
                facts=list(set(all_facts))[:20],
                entities=[],
                relations=[],
            )

            return RetrievalResult(
                query=query,
                results=all_results,
                insights=insight,
                mode="insight"
            )

        except Exception as e:
            logger.error(f"Insight forge failed: {e}")
            return RetrievalResult(
                query=query,
                results=[],
                mode="insight"
            )

    async def search(
        self,
        query: str,
        mode: str = "quick",
        graph_id: Optional[str] = None,
        limit: int = 10
    ) -> RetrievalResult:
        """
        Unified search interface.

        Args:
            query: Search query
            mode: Search mode ('quick', 'panorama', 'insight')
            graph_id: Target graph ID
            limit: Max results

        Returns:
            RetrievalResult
        """
        if mode == "insight":
            return await self.insight_forge(query, graph_id, limit)
        elif mode == "panorama":
            return await self.panorama_search(query, graph_id, limit)
        else:
            return await self.quick_search(query, graph_id, limit)


async def search_graph(
    query: str,
    mode: str = "quick",
    graph_id: Optional[str] = None,
    limit: int = 10,
    zep_client: Optional[ZepGraphClient] = None
) -> RetrievalResult:
    """
    Convenience function to search graph.

    Args:
        query: Search query
        mode: Search mode
        graph_id: Target graph ID
        limit: Max results
        zep_client: Optional ZepGraphClient

    Returns:
        RetrievalResult
    """
    agent = RetrieverAgent(zep_client=zep_client)
    return await agent.search(query, mode, graph_id, limit)
