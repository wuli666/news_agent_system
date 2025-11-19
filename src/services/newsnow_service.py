"""
NewsNow Service - Core news aggregation service.

This module provides the NewsNowService class for fetching news from multiple platforms.
No MCP overhead - direct Python API.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

logger = logging.getLogger(__name__)

DEFAULT_PLATFORM_CONFIG: List[Tuple[str, str]] = [
    ("toutiao", "今日头条"),
    ("baidu", "百度热搜"),
    ("weibo", "微博热搜"),
    ("douyin", "抖音热点"),
    ("zhihu", "知乎热榜"),
    ("bilibili-hot-search", "Bilibili 热搜"),
]

STOPWORDS = {
    "的", "了", "在", "是", "和", "有", "就", "不", "人", "都",
    "一个", "上", "也", "很", "到", "说", "要", "去", "会", "着",
    "没有", "看", "好", "自己", "这", "那", "来", "被", "与", "为",
    "等", "更", "最", "再",
}


def _load_platforms_from_env() -> List[Tuple[str, str]]:
    """
    Parse NEWSNOW_PLATFORMS env value.

    Accepts comma-separated entries such as:
    - "weibo,zhihu"
    - "weibo:微博,zhihu:知乎热榜"
    """
    raw = os.getenv("NEWSNOW_PLATFORMS", "").strip()
    if not raw:
        return DEFAULT_PLATFORM_CONFIG

    platforms: List[Tuple[str, str]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            pid, name = entry.split(":", 1)
            platforms.append((pid.strip(), name.strip() or pid.strip()))
        else:
            platforms.append((entry, entry))
    return platforms or DEFAULT_PLATFORM_CONFIG


@dataclass(frozen=True)
class Platform:
    platform_id: str
    display_name: str


class NewsNowService:
    """Light-weight wrapper around the NewsNow HTTP endpoints."""

    def __init__(self) -> None:
        self.base_url = os.getenv("NEWSNOW_BASE_URL", "https://newsnow.busiyi.world").rstrip("/")
        self.timeout = float(os.getenv("NEWSNOW_TIMEOUT_SECONDS", "15"))
        self.per_platform_limit = int(os.getenv("NEWSNOW_PER_PLATFORM_LIMIT", "30"))
        self.global_limit = int(os.getenv("NEWSNOW_MAX_ITEMS", "200"))
        self.platforms: List[Platform] = [
            Platform(platform_id=pid, display_name=name)
            for pid, name in _load_platforms_from_env()
        ]
        user_agent = os.getenv(
            "NEWSNOW_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0 Safari/537.36",
        )
        accept_language = os.getenv("NEWSNOW_ACCEPT_LANGUAGE", "zh-CN,zh;q=0.9,en;q=0.8")
        referer = os.getenv("NEWSNOW_REFERER", self.base_url)
        cookie = os.getenv("NEWSNOW_COOKIE", "").strip()
        self.default_headers = {
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": accept_language,
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Referer": referer,
        }
        if cookie:
            self.default_headers["Cookie"] = cookie

    def _resolve_platforms(self, requested: Optional[Sequence[str]]) -> List[Platform]:
        """Validate and resolve requested platform ids."""
        if requested:
            requested_set = {item.strip() for item in requested if item.strip()}
            resolved = [p for p in self.platforms if p.platform_id in requested_set]
            missing = requested_set - {p.platform_id for p in resolved}
            if missing:
                logger.warning("Unknown NewsNow platforms requested: %s", ", ".join(sorted(missing)))
            if resolved:
                return resolved
        return self.platforms

    async def _fetch_platform(
        self,
        client: httpx.AsyncClient,
        platform: Platform,
        include_url: bool,
        per_platform_limit: int,
    ) -> List[Dict[str, Any]]:
        """Fetch raw news list for a single platform."""
        url = f"{self.base_url}/api/s"
        params = {"id": platform.platform_id, "latest": ""}
        response = await client.get(url, params=params, headers=self.default_headers)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items", [])

        formatted: List[Dict[str, Any]] = []
        for rank, item in enumerate(items[:per_platform_limit], start=1):
            news_item = {
                "platform_id": platform.platform_id,
                "platform_name": platform.display_name,
                "title": item.get("title", "").strip(),
                "rank": rank,
                "source": item.get("source") or platform.display_name,
                "category": item.get("category") or "text",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
            if include_url:
                news_item["url"] = item.get("url") or item.get("mobileUrl")
                news_item["mobile_url"] = item.get("mobileUrl")
            formatted.append(news_item)

        return formatted

    async def _gather_platform_news(
        self,
        platforms: Sequence[Platform],
        include_url: bool,
        limit: int,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Fetch data for multiple platforms concurrently."""
        import asyncio

        per_platform_limit = max(1, min(self.per_platform_limit, limit))
        items: List[Dict[str, Any]] = []
        errors: List[str] = []

        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            tasks = [
                self._fetch_platform(client, platform, include_url, per_platform_limit)
                for platform in platforms
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for platform, result in zip(platforms, results, strict=False):
            if isinstance(result, Exception):
                logger.error("Failed to load %s: %s", platform.platform_id, result)
                errors.append(platform.platform_id)
                continue
            items.extend(result)

        items.sort(key=lambda item: (item["platform_id"], item["rank"]))
        return items[:limit], errors

    async def get_latest_news(
        self,
        platforms: Optional[Sequence[str]] = None,
        limit: int = 50,
        include_url: bool = False,
    ) -> Dict[str, Any]:
        """Return a combined list of news items."""
        limit = max(1, min(limit, self.global_limit))
        platform_entries = self._resolve_platforms(platforms)
        if not platform_entries:
            return {
                "success": False,
                "items": [],
                "error": "No NewsNow platforms configured.",
            }

        items, errors = await self._gather_platform_news(
            platform_entries,
            include_url=include_url,
            limit=limit,
        )

        return {
            "success": True,
            "items": items,
            "platforms": [p.platform_id for p in platform_entries],
            "errors": errors,
            "total": len(items),
        }

    async def search_news(
        self,
        query: str,
        platforms: Optional[Sequence[str]] = None,
        limit: int = 50,
        include_url: bool = False,
    ) -> Dict[str, Any]:
        """Simple keyword search over the latest batch of news."""
        query = (query or "").strip()
        if not query:
            return {
                "success": False,
                "items": [],
                "error": "Query must not be empty.",
            }

        latest = await self.get_latest_news(platforms, limit=self.global_limit, include_url=include_url)
        if not latest.get("success"):
            return latest

        lowered = query.lower()
        matched = [
            item
            for item in latest["items"]
            if lowered in item.get("title", "").lower()
        ]

        return {
            "success": True,
            "query": query,
            "items": matched[:limit],
            "total": len(matched),
        }

    async def get_trending_topics(
        self,
        platforms: Optional[Sequence[str]] = None,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Generate naive trending keywords."""
        latest = await self.get_latest_news(platforms, limit=self.global_limit, include_url=False)
        if not latest.get("success"):
            return latest

        keyword_counter: Dict[str, int] = {}
        for item in latest["items"]:
            for word in self._extract_keywords(item.get("title", "")):
                keyword_counter[word] = keyword_counter.get(word, 0) + 1

        sorted_keywords = sorted(keyword_counter.items(), key=lambda kv: kv[1], reverse=True)
        topics = [
            {"keyword": keyword, "count": count}
            for keyword, count in sorted_keywords[:max(1, top_n)]
        ]

        return {
            "success": True,
            "topics": topics,
            "total": len(topics),
        }

    @staticmethod
    def _extract_keywords(text: str) -> Iterable[str]:
        """Extract candidate keywords from the news title."""
        if not text:
            return []
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text)
        for token in tokens:
            token = token.strip()
            if len(token) <= 1:
                continue
            if token in STOPWORDS:
                continue
            yield token
