"""
News collection tools using @tool decorator.
Direct integration - no MCP overhead.
"""
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

import httpx
from langchain_core.tools import tool

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Import NewsNow service and utilities
from ..services.newsnow_service import NewsNowService
from ..utils.history_storage import get_storage
from ..utils.news_dedup import deduplicate_news


# Create a shared service instance
_news_service = NewsNowService()


@tool
async def get_latest_news(
    platforms: Optional[List[str]] = None,
    limit: int = 50,
    include_url: bool = True,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """
    获取最新新闻。

    从配置的新闻平台获取实时热点新闻。支持6个平台：
    今日头条、百度热搜、微博热搜、抖音热点、知乎热榜、Bilibili热搜。

    自动进行去重处理，将来自不同平台的相似新闻合并。

    Args:
        platforms: 平台ID列表，如 ['weibo', 'baidu']。不指定则返回所有平台。
        limit: 返回的最大新闻数量，默认50条
        include_url: 是否包含新闻链接
        deduplicate: 是否自动去重合并相似新闻（默认True）

    Returns:
        包含新闻列表的字典，格式：
        {
            "success": True,
            "items": [{"title": "...", "platform_name": "...", ...}],
            "total": 50,
            "platforms": ["weibo", "baidu"],
            "deduplicated": True (如果进行了去重)
        }
    """
    result = await _news_service.get_latest_news(platforms, limit, include_url)

    # 自动去重
    if deduplicate and result.get("success") and result.get("items"):
        dedup_result = deduplicate_news(
            result["items"],
            similarity_threshold=0.85,  # 更严格阈值，避免误删不同主题
            keep_duplicates_info=True,  # 保留重复信息以便追踪合并情况
        )
        result["items"] = dedup_result["items"]
        result["total"] = dedup_result["total"]
        result["original_total"] = dedup_result["original_total"]
        result["removed_duplicates"] = dedup_result["removed_duplicates"]
        result["deduplicated"] = True

    return result


@tool
async def search_news(
    query: str,
    platforms: Optional[List[str]] = None,
    limit: int = 30,
    include_url: bool = True,
) -> Dict[str, Any]:
    """
    搜索新闻标题。

    在最新的新闻批次中搜索包含指定关键词的新闻。

    Args:
        query: 搜索关键词
        platforms: 指定搜索的平台，不指定则搜索所有平台
        limit: 最多返回多少条结果
        include_url: 是否包含新闻链接

    Returns:
        匹配的新闻列表
    """
    return await _news_service.search_news(query, platforms, limit, include_url)


@tool
async def get_trending(
    platforms: Optional[List[str]] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    获取热门话题关键词。

    基于新闻标题的关键词频率分析，返回当前最热门的话题。

    Args:
        platforms: 指定分析的平台
        top_n: 返回前N个热门关键词

    Returns:
        热门关键词列表及其出现次数
    """
    return await _news_service.get_trending_topics(platforms, top_n)


@tool
async def get_historical_news(
    date: str,
    platforms: Optional[List[str]] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    获取历史新闻数据。

    查询指定日期的历史新闻记录。需要先运行 crawler.py 采集历史数据。

    Args:
        date: 日期，格式为 "YYYY-MM-DD"，如 "2025-11-19"
        platforms: 筛选的平台ID列表
        limit: 最多返回多少条

    Returns:
        历史新闻数据，包含采集时间戳
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {
            "success": False,
            "error": f"日期格式错误: {date}. 请使用 YYYY-MM-DD 格式"
        }

    storage = get_storage()
    data = storage.load_news_data(target_date)

    if not data:
        return {
            "success": False,
            "error": f"未找到日期 {date} 的数据",
            "suggestion": "运行 crawler.py 采集历史数据"
        }

    news_items = data.get("data", {}).get("items", [])

    if platforms:
        platform_set = set(platforms)
        news_items = [
            item for item in news_items
            if item.get("platform_id") in platform_set
        ]

    return {
        "success": True,
        "date": date,
        "items": news_items[:limit],
        "total": len(news_items),
        "fetched_at": data.get("timestamp")
    }


@tool
def get_available_dates() -> Dict[str, Any]:
    """
    查看可用的历史数据日期。

    返回所有已采集的历史数据日期列表，以及日期范围和统计信息。

    Returns:
        可用日期列表和统计信息
    """
    storage = get_storage()
    dates = storage.get_available_dates()
    date_range = storage.get_date_range()
    stats = storage.get_stats()

    return {
        "success": True,
        "available_dates": [d.strftime("%Y-%m-%d") for d in dates],
        "total_days": len(dates),
        "date_range": {
            "earliest": date_range[0].strftime("%Y-%m-%d") if date_range else None,
            "latest": date_range[1].strftime("%Y-%m-%d") if date_range else None,
        },
        "stats": stats
    }


@tool
async def fetch_article(url: str, max_length: int = 3000) -> Dict[str, Any]:
    """
    抓取网页文章内容。

    使用简单的网页抓取器获取文章的完整内容。适用于需要分析新闻详情的场景。

    Args:
        url: 文章URL
        max_length: 内容最大长度，超过会截断

    Returns:
        文章标题和内容
    """
    if not HAS_BS4:
        return {
            "success": False,
            "error": "需要安装 BeautifulSoup4: pip install beautifulsoup4"
        }

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()

            # Extract title and content
            title = soup.title.get_text(strip=True) if soup.title else "无标题"
            content_tags = soup.find_all(["p", "article", "main"])
            content = "\n".join(
                tag.get_text(strip=True)
                for tag in content_tags
                if tag.get_text(strip=True)
            )

            # Clean and limit
            content = re.sub(r'\s+', ' ', content)
            if len(content) > max_length:
                content = content[:max_length] + "...[已截断]"

            return {
                "success": True,
                "url": url,
                "title": title,
                "content": content,
                "length": len(content)
            }

    except httpx.TimeoutException:
        return {"success": False, "url": url, "error": "请求超时"}
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}


# Export all tools as a list for easy import
ALL_NEWS_TOOLS = [
    get_latest_news,  # 已内置去重功能
    search_news,
    get_trending,
    get_historical_news,
    get_available_dates,
    fetch_article,
]
