"""
搜索相关工具函数

包含：
- Tavily 搜索（已废弃）
- arXiv 论文获取
- Airiv 论文获取
"""
import asyncio
import logging
import os
from typing import Any, Dict, List
from pathlib import Path

import httpx
from langchain_community.utilities import ArxivAPIWrapper

logger = logging.getLogger(__name__)

# Configuration
AIRIV_API_BASE = os.getenv("AIRIV_API_BASE", "").strip()
AIRIV_API_KEY = os.getenv("AIRIV_API_KEY", "").strip()
ARXIV_DAILY_QUERY = os.getenv(
    "ARXIV_DAILY_QUERY",
    "cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.IR",
).strip()


async def run_tavily_search(
    query: str,
    max_results: int = 5,
    include_answer: bool = True
) -> Dict[str, Any]:
    """
    Deprecated: use create_react_agent with tavily_search instead.

    Args:
        query: Search query
        max_results: Maximum results to return
        include_answer: Whether to include answer

    Returns:
        Dict with success=False (deprecated)
    """
    return {"success": False, "error": "deprecated"}


async def fetch_daily_papers_from_airiv(limit: int = 6) -> List[Dict[str, Any]]:
    """
    拉取 Airiv 的每日论文列表（已分类）。

    Args:
        limit: 最多返回的论文数量

    Returns:
        论文列表，格式: [{title, category, url, summary}]
    """
    if not AIRIV_API_BASE:
        return []

    url = f"{AIRIV_API_BASE.rstrip('/')}/api/papers/daily"
    headers = {"Accept": "application/json"}
    if AIRIV_API_KEY:
        headers["Authorization"] = f"Bearer {AIRIV_API_KEY}"
    params = {"limit": max(1, min(limit, 20))}

    try:
        async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception:
        logger.debug("Airiv daily papers fetch failed", exc_info=True)
        return []

    papers: List[Dict[str, Any]] = []
    data = payload.get("papers") or payload.get("data") or payload.get("items") or []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        papers.append({
            "title": title,
            "category": str(item.get("category") or item.get("field") or "综合"),
            "url": item.get("url") or item.get("link"),
            "summary": str(item.get("summary") or item.get("abstract") or "")[:240],
        })
        if len(papers) >= limit:
            break
    return papers


async def fetch_daily_papers_from_arxiv(limit: int = 6) -> List[Dict[str, Any]]:
    """
    使用 ArxivAPIWrapper 获取每日论文。

    Args:
        limit: 最多返回的论文数量

    Returns:
        论文列表，格式: [{title, category, url, summary, tags, ts}]
    """
    try:
        wrapper = ArxivAPIWrapper(
            top_k_results=30,  # 先抓多一些，后续再筛选
            load_max_docs=30,
            load_all_available_meta=True,
        )

        def _query() -> List[Any]:
            # Use get_summaries_as_docs to get metadata without downloading PDFs
            return wrapper.get_summaries_as_docs(ARXIV_DAILY_QUERY)

        docs = await asyncio.to_thread(_query)
    except Exception:
        logger.debug("Arxiv daily papers fetch failed", exc_info=True)
        return []

    def _parse_ts(meta: dict) -> float:
        """解析时间戳"""
        from datetime import datetime
        for key in ("Published", "published", "updated", "Updated"):
            val = meta.get(key)
            if not val:
                continue
            try:
                return datetime.fromisoformat(str(val).replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
        return 0.0

    def _classify_tags(cats: List[str]) -> List[str]:
        """分类标签"""
        tags = []
        cat_lower = ",".join(cats).lower()
        if "cs.cv" in cat_lower:
            tags.append("计算机视觉")
        if "cs.cl" in cat_lower:
            tags.append("NLP")
        if "cs.lg" in cat_lower or "stat.ml" in cat_lower:
            tags.append("机器学习")
        if "cs.ir" in cat_lower:
            tags.append("信息检索")
        if "cs.ai" in cat_lower:
            tags.append("AI")
        if not tags and cats:
            tags.append(cats[0])
        return tags

    papers: List[Dict[str, Any]] = []
    for doc in docs or []:
        try:
            meta = getattr(doc, "metadata", {}) or {}
            title = meta.get("Title") or meta.get("title") or ""
            # Entry ID contains the arxiv URL
            entry_id = meta.get("Entry ID") or meta.get("entry_id") or ""
            url = entry_id if entry_id else (meta.get("pdf_url") or meta.get("link"))
            summary = getattr(doc, "page_content", "") or meta.get("Summary") or meta.get("summary") or ""
            if not title:
                continue
            categories_raw = meta.get("categories", []) or meta.get("Category", []) or []
            categories = categories_raw if isinstance(categories_raw, list) else [categories_raw]
            tags = _classify_tags(categories if isinstance(categories, list) else [categories])
            papers.append({
                "title": str(title).strip(),
                "category": ", ".join(categories) if categories else "arXiv",
                "url": url,
                "summary": str(summary).strip()[:240],
                "tags": tags,
                "ts": _parse_ts(meta),
            })
        except Exception:
            continue

    # 按时间倒序，取前 limit
    papers_sorted = sorted(papers, key=lambda x: x.get("ts", 0), reverse=True)
    return papers_sorted[:limit]


async def fetch_daily_papers(limit: int = 6) -> List[Dict[str, Any]]:
    """
    获取每日论文，优先用 arXiv，失败时回退到 Airiv（如配置了接口）。

    Args:
        limit: 最多返回的论文数量

    Returns:
        论文列表
    """
    papers = await fetch_daily_papers_from_arxiv(limit=limit)
    if papers:
        return papers
    return await fetch_daily_papers_from_airiv(limit=limit)
