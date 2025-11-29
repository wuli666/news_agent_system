"""
Refactored agent nodes for improved multi-agent architecture.

Key improvements:
- Separated data collection from decision-making
- Team coordinators for parallel task distribution
- Cleaner state management without temporary control fields
- Better separation of concerns
"""
import asyncio
import base64
import io
import json
import logging
import os
import re
import smtplib
import ssl
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formataddr
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langgraph.types import Command

from src.graph.types import NewsItem, State, NewsReport, ReflectionResult, InformationGap
from src.llms.llm import get_llm_by_type
from langgraph.prebuilt import create_react_agent
from src.config.agents import AGENT_LLM_MAP
from src.config.settings import settings
from src.prompts import get_system_prompt
from src.services.newsnow_service import STOPWORDS, NewsNowService
from src.utils.news_dedup import deduplicate_news
from src.tools.news_tools import tavily_search, tavily_extract
from langchain_community.utilities import ArxivAPIWrapper


async def _run_tavily_search(query: str, max_results: int = 5, include_answer: bool = True) -> Dict[str, Any]:
    """Deprecated: use create_react_agent with tavily_search instead."""
    return {"success": False, "error": "deprecated"}

# Shared output root
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AIRIV_API_BASE = os.getenv("AIRIV_API_BASE", "").strip()
AIRIV_API_KEY = os.getenv("AIRIV_API_KEY", "").strip()
ARXIV_DAILY_QUERY = os.getenv(
    "ARXIV_DAILY_QUERY",
    "cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.IR",
).strip()


_SENSITIVE_PATTERNS = [
    r"自杀", r"枪击", r"爆炸", r"袭击", r"恐怖", r"极端", r"性侵", r"色情",
    r"赌博", r"毒品", r"暴力", r"谋杀", r"分裂", r"煽动", r"恐吓", r"ISIS",
]


def _mask_sensitive_text(text: str) -> str:
    """Basic masking to avoid DashScope data inspection triggers."""
    masked = text or ""
    for pattern in _SENSITIVE_PATTERNS:
        masked = re.sub(pattern, "[敏感内容]", masked, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", masked).strip()


async def _fetch_daily_papers_from_airiv(limit: int = 6) -> List[Dict[str, Any]]:
    """
    拉取 Airiv 的每日论文列表（已分类）。
    返回格式: [{title, category, url, summary}]
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


async def _fetch_daily_papers_from_arxiv(limit: int = 6) -> List[Dict[str, Any]]:
    """使用 ArxivAPIWrapper 获取每日论文。"""
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


async def _fetch_daily_papers(limit: int = 6) -> List[Dict[str, Any]]:
    """优先用 arXiv，失败时回退到 Airiv（如配置了接口）。"""
    papers = await _fetch_daily_papers_from_arxiv(limit=limit)
    if papers:
        return papers
    return await _fetch_daily_papers_from_airiv(limit=limit)


def _build_safe_news_list(news_items: Sequence[Dict[str, Any]]) -> str:
    """Render news titles with light redaction to reduce moderation failures."""
    safe_lines = []
    for idx, item in enumerate(news_items):
        title = _mask_sensitive_text(item.get("title", ""))[:200]
        safe_lines.append(f"{idx + 1}. {title}")
    return "\n".join(safe_lines)


def _stringify_message_content(content: Any) -> str:
    """Convert LangChain message content into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                text_value = chunk.get("text") or chunk.get("content")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _coerce_tool_payload(content: Any) -> Any:
    """Convert tool message content back to Python objects."""
    if isinstance(content, str):
        raw = content.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(content, list):
        merged = "".join(
            chunk if isinstance(chunk, str) else chunk.get("text", "")
            for chunk in content
            if isinstance(chunk, (str, dict))
        ).strip()
        if merged:
            try:
                return json.loads(merged)
            except json.JSONDecodeError:
                return None
    if isinstance(content, dict):
        return content
    return None


def _extract_news_items_from_payload(payload: Any, tool_name: str) -> List[Dict[str, Any]]:
    """Find news entries from a tool payload."""
    if payload is None:
        return []

    if isinstance(payload, dict):
        if tool_name in {"fetch_article"}:
            return [payload]
        for key in ("items", "news", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if all(isinstance(v, dict) for v in payload.values()):
            return list(payload.values())
        return []

    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    return []


def _collect_news_from_tool_messages(messages: Sequence[Any]) -> List[NewsItem]:
    """Extract news items from tool outputs produced by the agent."""
    collected: List[NewsItem] = []
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (message.name or "").strip()
        if tool_name not in {"get_latest_news", "search_news", "get_historical_news", "fetch_article"}:
            continue

        payload = _coerce_tool_payload(message.content)
        if payload is None and getattr(message, "artifact", None) is not None:
            payload = message.artifact

        raw_items = _extract_news_items_from_payload(payload, tool_name)
        if not raw_items:
            continue

        normalized = _normalize_news_items(raw_items, default_category="text")
        if normalized:
            collected.extend(normalized)
    return collected


def _coerce_timestamp(raw_ts: Any) -> Optional[str]:
    """Parse timestamp and clamp stale values to retrieval time."""
    if not raw_ts:
        return None

    shanghai_tz = timezone(timedelta(hours=8))

    try:
        if isinstance(raw_ts, datetime):
            ts = raw_ts
        else:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
    except Exception:
        return None

    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=shanghai_tz)
    else:
        ts = ts.astimezone(shanghai_tz)

    now = datetime.now(shanghai_tz)
    if ts < now.replace(microsecond=0) and (now - ts).days > 60:
        return None
    return ts.isoformat()


def _normalize_news_items(items: Sequence[Dict[str, Any]], default_category: str = "text") -> List[NewsItem]:
    """Normalize heterogeneous tool outputs into NewsItem structures."""
    normalized: List[NewsItem] = []
    now_iso = datetime.now().isoformat()
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        title = str(
            raw.get("title")
            or raw.get("headline")
            or raw.get("name")
            or "未命名新闻"
        ).strip()
        content = raw.get("content") or raw.get("summary") or raw.get("description") or title
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content if part)
        content = str(content).strip()
        source = str(
            raw.get("source")
            or raw.get("platform_name")
            or raw.get("platform")
            or "未知来源"
        ).strip()
        timestamp = _coerce_timestamp(
            raw.get("timestamp")
            or raw.get("published_at")
            or raw.get("retrieved_at")
            or raw.get("time")
        ) or now_iso
        category = str(raw.get("category") or default_category or "text")
        url = raw.get("url") or raw.get("link") or raw.get("mobile_url")

        enriched = dict(raw)
        enriched.update(
            NewsItem(
                title=title,
                content=content or title,
                source=source or "未知来源",
                timestamp=str(timestamp),
                category=category or "text",
                url=url,
            )
        )
        normalized.append(enriched)
    return normalized


def _bm25_keywords(news_items: Sequence[NewsItem], top_k: int = 30, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
    """BM25 粗选关键词，支持比分/型号等特殊 token。"""
    docs: List[List[str]] = []
    df: Dict[str, int] = {}
    for item in news_items or []:
        title = str(item.get("title") or "").lower()
        token_set: set[str] = set()
        token_set.update(re.findall(r"\b\d+:\d+\b", title))          # 比分如 0:2
        token_set.update(re.findall(r"\b[a-z]+\d{2,4}\b", title))    # 型号如 mate80
        token_set.update(re.findall(r"[\w\u4e00-\u9fff]+", title))
        tokens: List[str] = []
        for tok in token_set:
            if len(tok) <= 1:
                continue
            if tok.isdigit() or re.fullmatch(r"\d+(\.\d+)?", tok):
                continue
            if len(tok) <= 3 and tok[0].isalpha() and tok[1:].isdigit():
                continue
            if tok in STOPWORDS:
                continue
            tokens.append(tok)
            df[tok] = df.get(tok, 0) + 1
        docs.append(tokens)

    N = max(1, len(docs))
    avg_len = sum(len(d) for d in docs) / N
    scores: Dict[str, float] = {}
    for doc in docs:
        tf: Dict[str, int] = {}
        for tok in doc:
            tf[tok] = tf.get(tok, 0) + 1
        doc_len = max(1, len(doc))
        for tok, freq in tf.items():
            idf = max(0.0, (N - df.get(tok, 1) + 0.5) / (df.get(tok, 1) + 0.5))
            idf = max(idf, 1e-6)  # avoid zero
            tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_len))
            scores[tok] = scores.get(tok, 0.0) + idf * tf_norm

    sorted_tokens = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"keyword": tok, "count": round(score, 4)} for tok, score in sorted_tokens[:max(1, top_k)]]


def _extract_keywords_from_news(news_items: Sequence[NewsItem], top_n: int = 10) -> List[Dict[str, Any]]:
    """回退方案：简单加权词频。"""
    keyword_counter: Dict[str, float] = {}
    platform_totals: Dict[str, int] = {}

    def _item_platforms(item: NewsItem) -> List[str]:
        platforms: List[str] = []
        pid = item.get("platform_id")
        if pid:
            platforms.append(str(pid).strip().lower())
        for p in item.get("platforms") or []:
            if p:
                platforms.append(str(p).strip().lower())
        if not platforms and item.get("source"):
            platforms.append(str(item.get("source")).strip().lower())
        return [p for p in platforms if p]

    for item in news_items or []:
        for pid in _item_platforms(item):
            platform_totals[pid] = platform_totals.get(pid, 0) + 1

    total_items = max(1, len(news_items or []))
    for item in news_items or []:
        title = str(item.get("title", "")).lower()
        token_set: set[str] = set()
        token_set.update(re.findall(r"\b\d+:\d+\b", title))
        token_set.update(re.findall(r"\b[a-z]+\d{2,4}\b", title))
        token_set.update(re.findall(r"[\w\u4e00-\u9fff]+", title))
        platforms = _item_platforms(item)
        if platforms:
            pid_weight = sum(1.0 / max(1, platform_totals.get(pid, 1)) for pid in platforms) / len(platforms)
        else:
            pid_weight = 1.0 / total_items
        for token in token_set:
            if len(token) <= 1:
                continue
            if token.isdigit() or re.fullmatch(r"\d+(\.\d+)?", token):
                continue
            if len(token) <= 3 and token[0].isalpha() and token[1:].isdigit():
                continue
            if token in STOPWORDS:
                continue
            keyword_counter[token] = keyword_counter.get(token, 0.0) + pid_weight

    sorted_keywords = sorted(keyword_counter.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"keyword": kw, "count": round(count, 4)}
        for kw, count in sorted_keywords[:max(1, top_n)]
    ]


def _apply_hotness_scores(items: List[NewsItem], trending_keywords: Optional[List[Dict[str, Any]]]) -> List[NewsItem]:
    """根据关键词命中次数为新闻打一个粗略 hot_score。"""
    kw_map = {kw["keyword"]: kw.get("count", 1) for kw in trending_keywords or []}
    scored: List[NewsItem] = []
    for item in items or []:
        score = 1.0
        title = item.get("title") or ""
        for kw, cnt in kw_map.items():
            if kw in title:
                score += cnt
        new_item = dict(item)
        new_item["hot_score"] = score
        scored.append(new_item)  # type: ignore[arg-type]
    return scored


def _sample_batch_diverse(
    items: List[NewsItem],
    start: int,
    batch_size: int,
) -> List[NewsItem]:
    """
    从 items[start:] 选择一个平台多样化的批次。
    策略：按平台分桶后轮询抽取，保持热度排序，剩余不足补齐。
    """
    slice_items = sorted(items[start:], key=lambda x: x.get("hot_score", 1), reverse=True)
    if batch_size <= 0 or not slice_items:
        return []

    def _item_platforms(item: NewsItem) -> List[str]:
        platforms: List[str] = []
        pid = item.get("platform_id")
        if pid:
            platforms.append(str(pid).strip().lower())
        for p in item.get("platforms") or []:
            if p:
                platforms.append(str(p).strip().lower())
        if not platforms and item.get("source"):
            platforms.append(str(item.get("source")).strip().lower())
        return platforms or ["unknown"]

    buckets: Dict[str, List[NewsItem]] = defaultdict(list)
    for it in slice_items:
        pid = _item_platforms(it)[0]
        buckets[pid].append(it)

    selected: List[NewsItem] = []
    while len(selected) < batch_size and buckets:
        for pid in list(buckets.keys()):
            if len(selected) >= batch_size:
                break
            bucket = buckets.get(pid) or []
            if bucket:
                selected.append(bucket.pop(0))
            if not bucket:
                buckets.pop(pid, None)

    if len(selected) < batch_size:
        for it in slice_items:
            if len(selected) >= batch_size:
                break
            if it not in selected:
                selected.append(it)

    return selected[:batch_size]


def _persist_step_snapshot(date_str: str, iteration: int, step: str, data: Dict[str, Any]) -> None:
    """把每步的关键信息写到 output/steps 方便排查。"""
    root = OUTPUT_DIR / "steps" / date_str
    root.mkdir(parents=True, exist_ok=True)
    file_path = root / f"{iteration:02d}_{step}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to persist step snapshot")


async def _extract_keywords_via_llm(news_items: Sequence[NewsItem], top_n: int = 10) -> List[Dict[str, Any]]:
    """调用 LLM 生成关键词（JSON 数组），失败则返回空列表让上层回退。"""
    if not news_items:
        return []
    titles = [
        str(item.get("title") or "").strip()
        for item in news_items
        if item.get("title")
    ][:40]
    llm = get_llm_by_type(AGENT_LLM_MAP.get("summary_trend", "qwen"))
    prompt = f"""请从以下新闻标题中提取 TOP {top_n} 关键词，输出 JSON 数组，不要任何额外文字。
每个元素格式: {{"keyword": "词语", "count": 频次或热度整数}}
新闻标题（共 {len(titles)} 条，编号仅用于参考）:
""" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles))

    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = _coerce_tool_payload(resp.content)
        if raw is None and isinstance(resp.content, str):
            try:
                raw = json.loads(resp.content)
            except Exception:
                raw = None
        if not isinstance(raw, list):
            return []
        parsed: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            kw = item.get("keyword")
            if not kw or len(str(kw).strip()) <= 1:
                continue
            parsed.append(
                {
                    "keyword": str(kw).strip(),
                    "count": item.get("count") or 1,
                }
            )
        return parsed[:top_n]
    except Exception:
        logger.exception("LLM keyword extraction failed")
        return []


def _save_report_to_html(
    date_str: str,
    task: str,
    final_report: str,
    text_news: Sequence[NewsItem],
    video_news: Sequence[NewsItem],
    timeline_analysis: Optional[str],
    trend_analysis: Optional[str],
    trending_keywords: Optional[List[Dict[str, Any]]],
    relationship_graph: Optional[str],
    news_images: Optional[Dict[str, List[str]]] = None,
    daily_papers: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """Persist a styled HTML report to output/reports/<date>.html."""
    try:
        import json
        reports_dir = OUTPUT_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = reports_dir / f"{date_str}.html"

        def _safe(val: str | None) -> str:
            return (val or "").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

        # Parse structured JSON report (now guaranteed to be valid JSON from Pydantic)
        report_data = None
        try:
            report_data = json.loads(final_report)
            logger.info("Successfully parsed structured report JSON")
        except Exception as e:
            logger.warning(f"Failed to parse report JSON: {e}, using fallback rendering")
            report_data = None

        def _render_source_bars(items: Sequence[NewsItem]) -> str:
            """渲染来源数量条形图（仅文字新闻去重来源）。"""
            if not items:
                return '<div class="bar-group"><div class="empty">暂无来源统计</div></div>'
            counts = Counter([str(it.get("source") or "未知来源") for it in items])
            total = sum(counts.values()) or 1
            bars = []
            for src, cnt in counts.most_common():
                pct = round(cnt / total * 100, 2)
                bars.append(f"""
                <div class="bar-row">
                  <span class="bar-label">{_safe(src)}</span>
                  <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
                  <span class="bar-value">{cnt} ({pct}%)</span>
                </div>
                """)
            return "<div class='bar-group'>" + "".join(bars[:8]) + "</div>"

        def _render_timeline_graph(timeline_text: Optional[str], key_events: Optional[List[Dict[str, Any]]] = None) -> str:
            """
            将时间线渲染为节点式横线展示。
            优先使用key_events（重点事件），如果没有则使用timeline_text。
            """
            items = []

            # 优先使用重点事件生成时间线
            if key_events:
                # 按时间排序事件（尝试解析时间）
                def parse_event_time(event: Dict[str, Any]) -> tuple:
                    """解析事件时间用于排序"""
                    time_str = str(event.get("time", "")).strip()
                    # 尝试提取日期时间
                    import re
                    # 匹配 YYYY-MM-DD 或 YYYY-MM-DD HH:MM
                    match = re.search(r'(\d{4})[-年/](\d{1,2})[-月/](\d{1,2})', time_str)
                    if match:
                        year, month, day = match.groups()
                        return (int(year), int(month), int(day))
                    # 匹配 MM月DD日
                    match = re.search(r'(\d{1,2})[月/](\d{1,2})', time_str)
                    if match:
                        month, day = match.groups()
                        return (2025, int(month), int(day))  # 假设当前年份
                    return (0, 0, 0)  # 无法解析的放最后

                sorted_events = sorted(key_events, key=parse_event_time)

                for event in sorted_events[:10]:  # 最多显示10个事件
                    time_label = _safe(event.get("time", "时间未知"))
                    title = _safe(event.get("title", ""))
                    # 截取标题，避免过长
                    if len(title) > 40:
                        title = title[:37] + "..."
                    items.append({
                        "time": time_label,
                        "desc": title,
                    })

            # 如果没有key_events，回退到使用timeline_text
            if not items and timeline_text:
                raw_lines = [ln.strip() for ln in timeline_text.splitlines() if ln.strip()]
                for idx, line in enumerate(raw_lines[:10]):
                    # 尝试拆分"时间 + 描述"
                    match = re.match(
                        r"^(?:-|\d+\.)?\s*(?P<time>(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{2})?)|(\d{1,2}[:：]\d{2})|([上下]午\d{1,2}[:：]\d{2})|(今天|昨日|此前|当日|稍早|稍晚))[\s，,:：-]+(?P<desc>.+)$",
                        line,
                    )
                    if match:
                        time_label = match.group("time")
                        desc = match.group("desc")
                    else:
                        time_label = f"T{idx + 1}"
                        desc = line
                    items.append({
                        "time": _safe(time_label),
                        "desc": _safe(desc),
                    })

            if not items:
                return '<div class="empty">暂无时间线</div>'

            nodes_html = "".join(
                f"""
              <div class="timeline-item">
                <div class="timeline-node"></div>
                <div class="timeline-meta">{item['time']}</div>
                <div class="timeline-desc">{item['desc']}</div>
              </div>
                """
                for item in items
            )
            return f"""
            <div class="timeline-wrap">
              <div class="timeline-track"></div>
              <div class="timeline-items">
                {nodes_html}
              </div>
            </div>
            """

        def _render_hero_overview(overview: Dict[str, Any]) -> str:
            """紧凑的总体概览，放在标题区域展示核心指标与情感饼图。"""
            if not overview:
                return ""

            total = overview.get("total_news", len(text_news) + len(video_news))
            text_cnt = overview.get("text_news", len(text_news))
            video_cnt = overview.get("video_news", len(video_news))
            summary = _safe(executive_summary or overview.get("summary", "今日信号平稳，重点关注少数热点事件"))

            sentiment = overview.get("sentiment_distribution", {})
            pos = sentiment.get("positive", 20)
            neu = sentiment.get("neutral", 50)
            neg = sentiment.get("negative", 30)
            pie_style = f"background: conic-gradient(#4ade80 0% {pos}%, #e2e8f0 {pos}% {pos+neu}%, #f87171 {pos+neu}% 100%);"

            return f"""
    <div class="hero-overview">
      <div class="hero-overview-top">
        <div class="overview-summary">
          <div class="icon">📊</div>
          <div>
            <h3>总体概览</h3>
            <p>{summary}</p>
            <div class="stat-chips">
              <div class="stat-chip">新闻总数<strong>{total}</strong></div>
              <div class="stat-chip">文字新闻<strong>{text_cnt}</strong></div>
              <div class="stat-chip">视频新闻<strong>{video_cnt}</strong></div>
            </div>
          </div>
        </div>
        <div class="sentiment-pie-box">
          <div class="pie-chart">
            <div class="pie" style="{pie_style}"></div>
            <div class="pie-center">情感</div>
          </div>
          <div class="pie-legend">
            <div><span class="dot pos"></span>正面 {pos}%</div>
            <div><span class="dot neu"></span>中性 {neu}%</div>
            <div><span class="dot neg"></span>负面 {neg}%</div>
          </div>
        </div>
      </div>
      <div>
        <h4 style="margin: 0 0 8px; color: var(--muted); font-size: 13px;">来源分布（文字新闻）</h4>
        {_render_source_bars(text_news)}
      </div>
    </div>
            """

        def _render_core_themes(themes: List[Dict[str, Any]]) -> str:
            """Render core themes table"""
            if not themes:
                return '<p class="empty">暂无主题分类</p>'

            rows = []
            for theme in themes[:6]:
                name = _safe(theme.get("theme", "未知主题"))
                heat = theme.get("heat_level", 3)
                stars = "⭐" * heat
                examples = theme.get("examples", [])[:2]
                examples_text = "、".join([_safe(ex) for ex in examples]) if examples else "无"
                attention = _safe(theme.get("attention", "中等"))

                rows.append(f"""
                <tr>
                  <td><strong>{name}</strong></td>
                  <td>{stars} ({heat}/5)</td>
                  <td class="examples-cell">{examples_text}</td>
                  <td><span class="attention-badge {attention}">{attention}</span></td>
                </tr>
                """)

            return f"""
            <table class="theme-table">
              <thead>
                <tr>
                  <th>主题类别</th>
                  <th>热度指数</th>
                  <th>典型新闻</th>
                  <th>关注度</th>
                </tr>
              </thead>
              <tbody>
                {"".join(rows)}
              </tbody>
            </table>
            """

        def _render_papers(papers: Optional[List[Dict[str, Any]]]) -> str:
            if not papers:
                return '<p class="empty">暂无重点论文</p>'
            cards = []
            for paper in papers[:10]:
                title = _safe(paper.get("title", "未命名论文"))
                cat = _safe(paper.get("category", "综合"))
                url = paper.get("url")
                summary = _safe(paper.get("summary", ""))
                link_html = f'<a href="{_safe(str(url))}" target="_blank" rel="noopener noreferrer">{title}</a>' if url else title
                tags = paper.get("tags") or []
                tags_html = "".join(f'<span class="paper-tag">{_safe(str(t))}</span>' for t in tags[:3])
                cards.append(f"""
                <div class="paper-card">
                  <div class="paper-cat">{cat}</div>
                  <div class="paper-title">{link_html}</div>
                  <div class="paper-tags">{tags_html}</div>
                  <div class="paper-summary">{summary}</div>
                </div>
                """)
            return "<div class='papers-grid'>" + "".join(cards) + "</div>"

        def _render_key_events(events: List[Dict[str, Any]]) -> str:
            """Render key events as cards"""
            if not events:
                return '<p class="empty">暂无重点事件</p>'

            def _related_news(event_title: str, news_items: Sequence[NewsItem]) -> List[Dict[str, str]]:
                """
                使用关键词匹配查找相关新闻。
                如果没有匹配到相关新闻，返回空列表。
                """
                if not event_title:
                    return []

                # 精简的停用词列表（只过滤最常见的虚词）
                stopwords = {
                    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
                    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会"
                }

                # 提取关键词（分词 + 去停用词），同时保留较长的短语片段
                keywords = []

                # 1. 先按标点分词，保留>=3字符的有意义词段
                for seg in re.split(r"[\s·：:，,。！!？?；;、——\-]+", event_title):
                    seg_clean = seg.strip()
                    if len(seg_clean) >= 3 and seg_clean not in stopwords:
                        keywords.append(seg_clean.lower())

                # 2. 如果关键词太少，再添加>=2字符的词（更宽松）
                if len(keywords) < 3:
                    for seg in re.split(r"[\s·：:，,。！!？?；;、——\-]+", event_title):
                        seg_clean = seg.strip()
                        if len(seg_clean) >= 2 and seg_clean not in stopwords and seg_clean.lower() not in keywords:
                            keywords.append(seg_clean.lower())

                if not keywords:
                    return []

                # 匹配相关新闻，按相关度排序
                matches = []
                seen = set()

                for n in news_items:
                    # 提取新闻标题
                    t = ""
                    link = None
                    if isinstance(n, dict):
                        t = str(n.get("title") or "").strip()
                        link = n.get("url") or n.get("link")
                    else:
                        t = str(getattr(n, "title", "") or "").strip()
                        link = getattr(n, "url", None) or getattr(n, "link", None)

                    if not t:
                        continue

                    t_lower = t.lower()

                    # 计算匹配的关键词数量
                    match_count = sum(1 for kw in keywords if kw in t_lower)

                    # 至少匹配1个关键词就算相关（降低阈值）
                    if match_count >= 1:
                        key = (t, link)
                        if key in seen:
                            continue
                        seen.add(key)
                        matches.append({"title": t, "url": link, "score": match_count})

                # 按匹配分数降序排序，取前3条
                matches.sort(key=lambda x: x["score"], reverse=True)
                related = [{"title": m["title"], "url": m["url"]} for m in matches[:3]]

                # 没匹配到相关新闻，返回空列表
                return related

            # 按关注度排序（高->中->低）
            priority = {"高": 0, "中": 1, "低": 2}
            events_sorted = sorted(events[:8], key=lambda e: priority.get(e.get("risk_level", "中"), 1))
            cards = []
            risk_colors = {"高": "danger", "中": "warning", "低": "success"}
            category_icons = {
                "国家安全": "🛡️",
                "外交关系": "🌐",
                "社会热点": "🔥",
                "经济动态": "💰",
                "文化娱乐": "🎭"
            }

            for event in events_sorted:
                title = _safe(event.get("title", "未知事件"))
                time = _safe(event.get("time", "时间未知"))
                summary = _safe(event.get("summary", ""))
                impact = _safe(event.get("impact", ""))
                risk = event.get("risk_level", "中")  # 关注度
                category = event.get("category", "社会热点")
                event_link = event.get("url") or event.get("link")
                icon = category_icons.get(category, "📌")
                risk_class = risk_colors.get(risk, "warning")
                related_items = _related_news(title, text_news)
                if not event_link and related_items and related_items[0].get("url"):
                    event_link = related_items[0]["url"]
                related_html = ""
                if related_items:
                    links_html = "".join(
                        f'<li><a href="{_safe(r["url"])}" target="_blank" rel="noopener">{_safe(r["title"])}</a></li>'
                        if r.get("url") else f'<li>{_safe(r["title"])}</li>'
                        for r in related_items
                    )
                    related_html = f"""
                    <details class="related-news">
                      <summary>相关新闻（{len(related_items)}）</summary>
                      <ul>{links_html}</ul>
                    </details>
                    """

                cards.append(f"""
                <div class="event-card">
                  <div class="event-header">
                    <div class="event-category">{icon} {_safe(category)}</div>
                    <span class="risk-badge {risk_class}">关注度 {risk}</span>
                  </div>
                  <h4 class="event-title">{f'<a href="{_safe(event_link)}" target="_blank" rel="noopener">{title}</a>' if event_link else title}</h4>
                  <div class="event-meta">📅 {time}</div>
                  <p class="event-summary">{summary}</p>
                  {f'<div class="event-impact"><strong>影响：</strong>{impact}</div>' if impact else ''}
                  {related_html}
                </div>
                """)

            return f'<div class="events-grid">{"".join(cards)}</div>'

        def _render_sentiment_risk(risks: List[Dict[str, Any]]) -> str:
            """Render sentiment &关注度 表格"""
            if not risks:
                return '<p class="empty">暂无风险分析</p>'

            rows = []
            sentiment_icons = {"正面": "😊", "中性": "😐", "负面": "😟"}
            severity_colors = {"高": "danger", "中": "warning", "低": "success"}

            for item in risks[:10]:
                title = _safe(item.get("title", ""))
                sentiment = item.get("sentiment", "中性")
                risk = _safe(item.get("risk", ""))
                severity = item.get("severity", "中")  # 关注度：高/中/低
                icon = sentiment_icons.get(sentiment, "📰")
                severity_class = severity_colors.get(severity, "warning")

                rows.append(f"""
                <tr>
                  <td class="title-cell">{title}</td>
                  <td><span class="sentiment-tag">{icon} {sentiment}</span></td>
                  <td class="risk-cell">{risk}</td>
                  <td><span class="severity-badge {severity_class}">{severity}</span></td>
                </tr>
                """)

            return f"""
            <table class="risk-table">
              <thead>
                <tr>
                  <th>新闻标题</th>
                  <th>情感倾向</th>
                  <th>关注点/潜在风险</th>
                  <th>关注度</th>
                </tr>
              </thead>
              <tbody>
                {"".join(rows)}
              </tbody>
            </table>
            """

        def _render_stocks(stocks: List[Dict[str, Any]]) -> str:
            """Render stock market analysis table"""
            if not stocks:
                return '<p class="empty">暂无股票数据（需要雪球新闻源）</p>'

            rows = []
            for stock in stocks[:15]:
                name = _safe(stock.get("stock_name", ""))
                code = _safe(stock.get("stock_code") or "-")
                change = stock.get("change_percent", "")
                trend = stock.get("trend", "横盘")
                heat = stock.get("heat_score", 0)
                summary = _safe(stock.get("news_summary", ""))

                # Determine trend color and icon
                if trend == "上涨" or (change and str(change).startswith("+")):
                    trend_icon = "📈"
                    badge_class = "stock-badge-up"
                elif trend == "下跌" or (change and str(change).startswith("-")):
                    trend_icon = "📉"
                    badge_class = "stock-badge-down"
                else:
                    trend_icon = "➡️"
                    badge_class = "stock-badge-flat"

                # Heat stars
                heat_val = float(heat) if heat else 0
                heat_display = int(min(heat_val, 10))
                heat_stars = "⭐" * heat_display

                change_display = _safe(str(change)) if change else "-"

                rows.append(f"""
                <tr>
                  <td class="stock-name"><strong>{name}</strong></td>
                  <td class="stock-code">{code}</td>
                  <td class="stock-change"><span class="stock-badge {badge_class}">{trend_icon} {change_display}</span></td>
                  <td class="stock-heat">{heat_stars} <span class="heat-num">({heat_display})</span></td>
                  <td class="stock-summary">{summary}</td>
                </tr>
                """)

            return f"""
            <div style="overflow-x: auto;">
              <table class="stock-table">
                <thead>
                  <tr>
                    <th>股票名称</th>
                    <th>代码</th>
                    <th>涨跌幅</th>
                    <th>热度</th>
                    <th>相关新闻摘要</th>
                  </tr>
                </thead>
                <tbody>
                  {"".join(rows)}
                </tbody>
              </table>
            </div>
            """

        def _render_trends(trends: List[Dict[str, Any]]) -> str:
            """Render trend predictions"""
            if not trends:
                return '<p class="empty">暂无趋势预测</p>'

            cards = []
            for trend in trends[:6]:
                dimension = _safe(trend.get("dimension", ""))
                short = _safe(trend.get("short_term", ""))
                long_term = _safe(trend.get("long_term", ""))

                cards.append(f"""
                <div class="trend-card">
                  <h4 class="trend-dimension">📈 {dimension}</h4>
                  <div class="trend-item">
                    <div class="trend-label">短期（1周）</div>
                    <p>{short}</p>
                  </div>
                  <div class="trend-item">
                    <div class="trend-label">中长期（1月）</div>
                    <p>{long_term}</p>
                  </div>
                </div>
                """)

            return f'<div class="trends-grid">{"".join(cards)}</div>'

        def _extract_stock_quote(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """提取股票价格/涨跌信息，兼容常见字段。"""
            def _to_float(val: Any) -> Optional[float]:
                if val in (None, ""):
                    return None
                try:
                    return float(str(val).replace("%", "").strip())
                except Exception:
                    return None

            price = None
            for key in ("price", "current", "last_price", "last", "quote"):
                price = _to_float(item.get(key))
                if price is not None:
                    break

            change = None
            for key in ("change", "chg", "delta"):
                change = _to_float(item.get(key))
                if change is not None:
                    break

            pct = None
            for key in ("change_percent", "pct_change", "percent", "chg_percent"):
                pct = _to_float(item.get(key))
                if pct is not None:
                    break

            symbol = item.get("symbol") or item.get("stock_code") or item.get("ticker")
            name = item.get("stock_name") or item.get("name")
            label = symbol or name

            # 如果结构化字段为空，尝试从标题粗略识别涨跌百分比
            if price is None and pct is None and change is None:
                title = str(item.get("title") or "").strip()
                if title:
                    dir_hint = 0
                    if re.search(r"(大涨|上涨|飙升|走高|回升|反弹|涨停)", title):
                        dir_hint = 1
                    elif re.search(r"(下跌|大跌|暴跌|跳水|走低|回落|跌停)", title):
                        dir_hint = -1

                    pct_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", title)
                    if pct_match:
                        try:
                            pct_val = float(pct_match.group(1))
                            if dir_hint != 0 and pct_val > 0:
                                pct_val = pct_val * dir_hint
                            pct = pct_val
                        except Exception:
                            pct = None
                    direction = dir_hint if dir_hint else (1 if (pct or 0) > 0 else -1 if (pct or 0) < 0 else 0)
                    if pct is None and direction == 0:
                        return None
                else:
                    return None

            direction = 0
            for val in (pct, change):
                if val is None:
                    continue
                if val > 0:
                    direction = 1
                    break
                if val < 0:
                    direction = -1
                    break

            return {
                "price": price,
                "change": change,
                "pct": pct,
                "symbol": _safe(str(symbol)) if symbol else None,
                "name": _safe(str(name)) if name else None,
                "label": _safe(str(label)) if label else None,
                "direction": direction,
            }

        def _merge_quote(old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            """优先保留已有数据，若旧值缺失则补充新值。"""
            if not old:
                return new
            if not new:
                return old
            merged = dict(old)
            for k, v in new.items():
                if merged.get(k) in (None, "") and v not in (None, ""):
                    merged[k] = v
            return merged

        def _render_quote(quote: Optional[Dict[str, Any]]) -> str:
            """渲染股价气泡，带涨跌色彩。"""
            if not quote:
                return ""
            direction = quote.get("direction", 0)
            cls = "up" if direction > 0 else "down" if direction < 0 else "flat"
            price = quote.get("price")
            change = quote.get("change")
            pct = quote.get("pct")
            label = quote.get("label") or quote.get("symbol") or quote.get("name") or "股价"

            parts: List[str] = []
            if change is not None:
                parts.append(f"{change:+.2f}")
            if pct is not None:
                parts.append(f"{pct:+.2f}%")
            change_part = " ".join(parts)
            price_part = f"{price:.2f}" if price is not None else "--"

            return f'<span class="quote-pill {cls}">{label} <span class="price">{price_part}</span><span>{change_part}</span></span>'

        def _render_news_items(items: Sequence[NewsItem], empty_placeholder: str, images_map: Optional[Dict[str, List[str]]] = None) -> str:
            if not items:
                return f'<div class="empty">{empty_placeholder}</div>'

            def _classify_topic(title: str) -> tuple[str, str]:
                title_lower = title.lower()
                if any(k in title_lower for k in ["火灾", "事故", "撞", "误杀", "遇难", "疫情", "谣言", "拘留"]):
                    return "社会民生", "tag-society"
                if any(k in title_lower for k in ["科技", "电池", "芯片", "ai", "算法", "科研", "科幻"]):
                    return "科技要闻", "tag-tech"
                if any(k in title_lower for k in ["美", "日", "中", "总统", "总理", "外交", "台海", "军", "政客", "政策"]):
                    return "时政外事", "tag-politics"
                if any(k in title_lower for k in ["股", "消费", "市场", "经济", "企业", "融资", "财报"]):
                    return "财经商业", "tag-biz"
                if any(k in title_lower for k in ["电影", "演唱会", "综艺", "动漫", "游戏", "电竞", "体育", "比赛", "球"]):
                    return "文体娱乐", "tag-ent"
                return "综合", "tag-default"

            # merge by title, aggregate sources
            merged: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []
            for it in items:
                raw_title = str(it.get("title") or "未命名").strip()
                key = raw_title.lower()
                if key not in merged:
                    merged[key] = {
                        "title": _safe(raw_title),
                        "link": it.get("url") or it.get("link"),
                        "hot": it.get("hot_score"),
                        "sources": [],
                        "images": images_map.get(key, []) if images_map else [],
                        "quote": None,
                    }
                    order.append(key)
                entry = merged[key]
                if entry.get("link") is None:
                    entry["link"] = it.get("url") or it.get("link")
                entry["hot"] = max(entry.get("hot") or 0, it.get("hot_score") or 0)
                src = _safe(str(it.get("source") or "未知来源"))
                if src not in entry["sources"]:
                    entry["sources"].append(src)
                entry["quote"] = _merge_quote(entry.get("quote"), _extract_stock_quote(it))

            grouped: Dict[str, List[str]] = {}
            topic_class_map: Dict[str, str] = {}

            for key in order:
                entry = merged[key]
                title = entry["title"]
                topic_label, topic_class = _classify_topic(title)
                hot = entry.get("hot")
                hot_str = f"{hot:.2f}" if isinstance(hot, (int, float)) else "-"
                imgs_html = ""
                urls = entry.get("images") or []
                if urls:
                    cover = urls[0]
                    thumbs = "".join(
                        f'<img src="{_safe(u)}" alt="img" loading="lazy" />'
                        for u in urls[:3]
                    )
                    imgs_html = f'<div class="card-cover"><img src="{_safe(cover)}" alt="cover" loading="lazy" /></div><div class="img-row">{thumbs}</div>'

                link = entry.get("link")
                safe_link = _safe(str(link)) if link else ""
                title_html = (
                    f'<a class="tag title-tag" href="{safe_link}" target="_blank" rel="noopener noreferrer">{title}</a>'
                    if link else f'<span class="tag title-tag">{title}</span>'
                )
                sources_text = " / ".join(entry["sources"][:6])
                quote_html = _render_quote(entry.get("quote"))
                link_html = f'<a class="src-pill" href="{safe_link}" target="_blank" rel="noopener noreferrer">来源链接</a>' if link else ""
                meta_html = f'<div class="meta-line"><span class="src-pill">{sources_text}</span><span class="hot-pill">热度 {hot_str}</span>{quote_html}{link_html}</div>'

                card = f"""
                <div class="card">
                    {title_html}
                    {meta_html}
                    {imgs_html}
                </div>
                """
                grouped.setdefault(topic_label, []).append(card)
                topic_class_map[topic_label] = topic_class

            groups_html = []
            for topic, cards in grouped.items():
                topic_class = topic_class_map.get(topic, "tag-default")
                groups_html.append(f"""
                <details class="topic-group">
                  <summary class="topic-header">
                    <span class="badge-pill {topic_class}"><span class="dot"></span>{topic}</span>
                    <span class="topic-count">{len(cards)} 条</span>
                  </summary>
                  <div class="grid-cards">
                    {"".join(cards)}
                  </div>
                </details>
                """)

            return "\n".join(groups_html) if groups_html else '<div class="empty">暂无新闻</div>'

        def _render_chips(items: Sequence[Dict[str, Any]] | None) -> str:
            if not items:
                return '<div class="empty">暂无关键词</div>'
            chips: List[str] = []
            for kw in items:
                if isinstance(kw, dict):
                    raw = str(kw.get("keyword") or kw.get("word") or kw.get("name") or "关键词")
                    count = kw.get("count")
                else:
                    raw = str(kw)
                    count = None
                clean = _safe(raw.replace("\\n", " ").replace("\n", " ").strip())
                count_str = f"{count}" if count not in (None, "") else ""
                chips.append(f'<span class="chip">{clean}{f" ({count_str})" if count_str else ""}</span>')
            return "\n".join(chips)

        def _to_paragraphs(text: str | None) -> str:
            if not text:
                return "<p class=\"empty\">暂无内容</p>"
            paragraphs = [p.strip() for p in str(text).split("\\n") if p.strip()]
            return "\\n".join(f"<p>{_safe(p)}</p>" for p in paragraphs)

        def _extract_relationship_edges(graph_text: str | None) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
            """
            优先解析 JSON nodes/links，其次 mermaid，再退化简单箭头行。
            返回 (nodes, edges, remaining_text)。
            """
            if not graph_text:
                return [], [], ""
            content = str(graph_text).strip()

            # 1) JSON
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "nodes" in data and "links" in data:
                    nodes = []
                    for n in data.get("nodes") or []:
                        if isinstance(n, dict) and n.get("id"):
                            nodes.append({"id": str(n.get("id")), "label": str(n.get("label") or n.get("id"))})
                    links = []
                    for e in data.get("links") or []:
                        if isinstance(e, dict) and e.get("source") and e.get("target"):
                            links.append({"source": str(e["source"]), "target": str(e["target"])})
                    nodes = sorted(nodes, key=lambda x: x["id"])
                    links = sorted(links, key=lambda x: (x["source"], x["target"]))
                    if links:
                        return nodes, links, ""
            except Exception:
                pass

            # 2) Mermaid 块
            mermaid_match = re.search(r"```mermaid\s*(graph[\s\S]*?)```", content, re.DOTALL | re.IGNORECASE)
            if not mermaid_match:
                mermaid_match = re.search(r"graph\s+(?:TD|LR|BT|RL)[\s\S]+", content, re.IGNORECASE)
            mermaid_code = mermaid_match.group(0) if mermaid_match else None
            if mermaid_code:
                nodes_set: set[str] = set()
                links: list[dict[str, str]] = []
                for line in mermaid_code.splitlines():
                    matches = re.findall(r"([A-Za-z0-9_\u4e00-\u9fa5]+)\s*-{1,}-\>+\s*([A-Za-z0-9_\u4e00-\u9fa5]+)", line)
                    for src, dst in matches:
                        nodes_set.add(src)
                        nodes_set.add(dst)
                        links.append({"source": src, "target": dst})
                if links:
                    remaining = content.replace(mermaid_code, "")
                    nodes = [{"id": n, "label": n} for n in sorted(nodes_set)]
                    links = sorted(links, key=lambda x: (x["source"], x["target"]))
                    return nodes, links, remaining.strip()

            # 3) 简单箭头行 A->B
            nodes_set: set[str] = set()
            links: list[dict[str, str]] = []
            lines = [ln for ln in content.splitlines() if "->" in ln]
            for ln in lines:
                matches = re.findall(r"([A-Za-z0-9_\u4e00-\u9fa5]+)\s*-{1,}-\>+\s*([A-Za-z0-9_\u4e00-\u9fa5]+)", ln)
                for src, dst in matches:
                    nodes_set.add(src)
                    nodes_set.add(dst)
                    links.append({"source": src, "target": dst})
            nodes_final = [{"id": n, "label": n} for n in sorted(nodes_set)]
            links = sorted(links, key=lambda x: (x["source"], x["target"]))
            return nodes_final, links, content

        def _build_pyvis_from_edges(nodes: list[dict[str, str]], links: list[dict[str, str]], graph_path: Path) -> Optional[str]:
            """使用 pyvis 渲染交互式图谱，返回文件名"""
            if not links:
                return None
            try:
                from pyvis.network import Network
            except Exception:
                logger.info("pyvis not installed; skip pyvis graph rendering.")
                return None
            try:
                net = Network(height="520px", width="100%", bgcolor="#050816", font_color="#e5e7eb", directed=True)
                for n in nodes:
                    net.add_node(n.get("id"), label=n.get("label") or n.get("id"))
                for e in links:
                    net.add_edge(e.get("source"), e.get("target"))
                net.set_options("""
                {
                  "physics": {"enabled": true, "solver": "forceAtlas2Based"},
                  "nodes": {"color": {"background": "#0f172a", "border": "#38bdf8"}},
                  "edges": {"color": {"color": "#38bdf8"}}
                }
                """)
                net.write_html(str(graph_path), notebook=False)
                return graph_path.name
            except Exception:
                logger.warning("pyvis graph rendering failed; fallback to force-graph", exc_info=True)
                return None

        if isinstance(report_data, dict):
            timeline_analysis = report_data.get("timeline_analysis") or timeline_analysis
            trend_analysis = report_data.get("trend_analysis") or trend_analysis
            relationship_graph = report_data.get("relationship_graph") or relationship_graph

        nodes_list, edges, relationship_plain = _extract_relationship_edges(relationship_graph)
        if not nodes_list and (trending_keywords or []):
            # fallback: use trending keywords as nodes
            nodes_list = [{"id": kw.get("keyword") or str(kw), "label": kw.get("keyword") or str(kw)} for kw in (trending_keywords or [])[:12]]
        edges = edges or []
        pyvis_iframe = None  # 仅使用前端 force-graph 渲染

        overview = report_data.get("overview") if isinstance(report_data, dict) else None
        core_themes = report_data.get("core_themes") if isinstance(report_data, dict) else []
        key_events = report_data.get("key_events") if isinstance(report_data, dict) else []
        sentiment_risk = report_data.get("sentiment_risk") if isinstance(report_data, dict) else []
        trends = report_data.get("trends") if isinstance(report_data, dict) else []
        executive_summary = report_data.get("executive_summary") if isinstance(report_data, dict) else final_report
        daily_papers = daily_papers or []

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{date_str} 新闻报告</title>
  <style>
    :root {{
      --bg: #050816;
      --bg-soft: #070b1c;
      --panel: #0b1220;
      --panel-soft: #0f172a;
      --card: #0f172a;
      --accent: #22d3ee;
      --accent-soft: rgba(34, 211, 238, 0.14);
      --accent-strong: #38bdf8;
      --muted: #9ca3af;
      --muted-soft: #6b7280;
      --text: #e5e7eb;
      --border: #1f2937;
      --border-soft: rgba(148, 163, 184, 0.35);
      --danger: #fb7185;
      --success: #4ade80;
      --warning: #facc15;
      --radius-lg: 18px;
      --radius-md: 14px;
      --radius-sm: 999px;
      --shadow-strong: 0 28px 60px rgba(0, 0, 0, 0.6);
      --shadow-soft: 0 16px 40px rgba(15, 23, 42, 0.85);
    }}

    * {{ box-sizing: border-box; -webkit-tap-highlight-color: transparent; }}
    html, body {{ margin: 0; padding: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", "Inter", sans-serif;
      background: radial-gradient(circle at 20% 15%, rgba(56, 189, 248, 0.16), transparent 55%),
        radial-gradient(circle at 80% 0%, rgba(147, 51, 234, 0.28), transparent 50%),
        radial-gradient(circle at 0% 80%, rgba(34, 197, 94, 0.18), transparent 55%), var(--bg);
      color: var(--text); -webkit-font-smoothing: antialiased; scroll-behavior: smooth; }}
    ::selection {{ background: rgba(56, 189, 248, 0.32); color: #f9fafb; }}

    .shell {{ max-width: 1220px; margin: 28px auto 40px; padding: 0 18px 48px; }}
    .top-bar {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; padding: 8px 4px; gap: 12px; font-size: 12px; color: var(--muted-soft); }}
    .brand {{ display: inline-flex; align-items: center; gap: 8px; font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: #c4d4ff; }}
    .brand-dot {{ width: 8px; height: 8px; border-radius: 999px; background: radial-gradient(circle at 30% 30%, #e0f2fe, #38bdf8); box-shadow: 0 0 12px rgba(56,189,248,0.9); }}
    .pill-soft {{ padding: 4px 10px; border-radius: 999px; border: 1px solid rgba(148, 163, 184, 0.35); background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(8px); display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 11px; }}
    .pill-soft-dot {{ width: 6px; height: 6px; border-radius: 999px; background: var(--success); box-shadow: 0 0 10px rgba(74, 222, 128, 0.9); }}

    .hero {{ position: relative; padding: 22px 22px 20px; border-radius: var(--radius-lg);
      background: linear-gradient(145deg, rgba(56, 189, 248, 0.26), rgba(129, 140, 248, 0.22)),
        radial-gradient(circle at 15% 0%, rgba(8, 47, 73, 0.9), transparent 62%),
        radial-gradient(circle at 80% 0%, rgba(46, 16, 101, 0.75), transparent 55%), var(--panel);
      border: 1px solid rgba(148, 163, 184, 0.35); box-shadow: var(--shadow-strong); overflow: hidden; }}
    .hero::before {{ content: ""; position: absolute; inset: 0;
      background: radial-gradient(circle at 0% 100%, rgba(56, 189, 248, 0.3), transparent 55%);
      opacity: 0.14; pointer-events: none; }}
    .hero-inner {{ position: relative; z-index: 1; }}
    .hero-top {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; justify-content: space-between; }}
    .hero-overview {{ margin-top: 16px; padding: 14px; border-radius: var(--radius-md); background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(148, 163, 184, 0.35); display: flex; flex-direction: column; gap: 12px; }}
    .hero-overview-top {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; align-items: center; }}
    @media (max-width: 820px) {{
      .hero-overview-top {{ grid-template-columns: 1fr; }}
    }}
    .hero-overview .overview-summary {{ display: flex; gap: 10px; align-items: flex-start; }}
    .hero-overview .overview-summary .icon {{ font-size: 26px; }}
    .hero-overview .overview-summary h3 {{ margin: 0 0 6px; font-size: 16px; }}
    .hero-overview .overview-summary p {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .hero-overview .stat-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .hero-overview .stat-chip {{ padding: 8px 10px; border-radius: var(--radius-sm); border: 1px solid rgba(148, 163, 184, 0.35); background: rgba(8, 47, 73, 0.6); font-size: 12px; color: var(--muted); }}
    .hero-overview .stat-chip strong {{ color: #e0ecff; font-size: 14px; margin-left: 4px; }}
    .timeline-wrap {{ position: relative; padding: 12px 10px 4px; overflow-x: auto; }}
    .timeline-track {{ position: absolute; top: 30px; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, rgba(56,189,248,0.5), rgba(99,102,241,0.5)); border-radius: 999px; min-width: 560px; }}
    .timeline-items {{ display: grid; grid-auto-flow: column; grid-auto-columns: minmax(180px, 1fr); gap: 14px; position: relative; z-index: 1; min-width: 560px; }}
    .timeline-item {{ position: relative; padding-top: 12px; }}
    .timeline-node {{ width: 14px; height: 14px; border-radius: 50%; background: #38bdf8; border: 3px solid #0b1220; box-shadow: 0 0 12px rgba(56,189,248,0.8); margin-bottom: 6px; }}
    .timeline-meta {{ font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
    .timeline-desc {{ font-size: 13px; line-height: 1.5; color: #e5e7eb; }}
    .pill {{ padding: 6px 13px; border-radius: var(--radius-sm); background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(148, 163, 184, 0.6); color: #e0f2fe; font-size: 12px; letter-spacing: 0.16em;
      text-transform: uppercase; display: inline-flex; align-items: center; gap: 8px; backdrop-filter: blur(10px); }}
    .pill::before {{ content: "●"; font-size: 9px; color: var(--accent); }}
    .hero h1 {{ margin: 0; font-size: clamp(24px, 3vw, 30px); letter-spacing: 0.02em; font-weight: 650; }}
    .meta {{ color: var(--muted); font-size: 13px; margin-top: 2px; display: flex; align-items: center; gap: 10px; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 18px; }}
    .stat-card {{ padding: 12px 13px; border-radius: var(--radius-md);
      background: radial-gradient(circle at 0% 0%, rgba(148, 163, 253, 0.32), transparent 55%), rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(148, 163, 184, 0.5); box-shadow: 0 14px 35px rgba(15, 23, 42, 0.85);
      display: flex; flex-direction: column; gap: 5px; position: relative; overflow: hidden; }}
    .stat-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em; }}
    .stat-value {{ font-size: 24px; font-weight: 700; margin-top: 2px; color: #e0ecff; }}

    .section {{ margin-top: 26px; padding: 18px 18px 18px; border-radius: var(--radius-lg);
      background: radial-gradient(circle at 0% 0%, rgba(15, 23, 42, 0.85), transparent 55%),
        radial-gradient(circle at 100% 0%, rgba(30, 64, 175, 0.7), transparent 50%), var(--panel-soft);
      border: 1px solid var(--border); box-shadow: var(--shadow-soft); position: relative; overflow: hidden; }}
    .section::before {{ content: ""; position: absolute; inset: 0;
      background: linear-gradient(120deg, rgba(56, 189, 248, 0.08), transparent 40%);
      opacity: 0.7; pointer-events: none; }}
    .section-inner {{ position: relative; z-index: 1; }}
    .section-eyebrow {{ font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted-soft); }}
    .section h2 {{ margin: 8px 0 12px; font-size: 18px; font-weight: 600; letter-spacing: 0.03em; }}
    .section-desc {{ font-size: 12px; color: var(--muted-soft); margin-bottom: 12px; }}

    .overview-cards {{ display: grid; grid-template-columns: 1fr auto; gap: 14px; margin-bottom: 12px; }}
    .summary-card-large {{ padding: 16px; border-radius: var(--radius-md);
      background: radial-gradient(circle at 0 0, rgba(56,189,248,.16), transparent), #020617;
      border: 1px solid rgba(30, 64, 175, 0.7); display: flex; gap: 14px; }}
    .summary-icon {{ font-size: 32px; }}
    .summary-content h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .summary-content p {{ margin: 0 0 12px; font-size: 13px; line-height: 1.5; }}
    .sentiment-bar {{ display: flex; height: 26px; border-radius: 999px; overflow: hidden; font-size: 11px; font-weight: 600; }}
    .sentiment-item {{ display: flex; align-items: center; justify-content: center; padding: 0 8px; }}
    .sentiment-item.positive {{ background: rgba(74, 222, 128, 0.4); color: #bbf7d0; }}
    .sentiment-item.neutral {{ background: rgba(148, 163, 184, 0.4); color: #e2e8f0; }}
    .sentiment-item.negative {{ background: rgba(251, 113, 133, 0.4); color: #fecdd3; }}
    .mini-stats {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .mini-stat {{ padding: 10px; border-radius: var(--radius-md); background: rgba(15, 23, 42, 0.8); border: 1px solid var(--border); text-align: center; }}
    .mini-stat-value {{ font-size: 20px; font-weight: 700; color: #e0ecff; }}
    .mini-stat-label {{ font-size: 10px; color: var(--muted); margin-top: 4px; text-transform: uppercase; }}
    .keywords-cloud {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}

    .theme-table, .risk-table, .stock-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
    .theme-table th, .risk-table th, .stock-table th {{ background: rgba(15, 23, 42, 0.8); color: var(--muted); font-weight: 600;
      text-align: left; padding: 10px; border: 1px solid var(--border); }}
    .theme-table td, .risk-table td, .stock-table td {{ padding: 10px; border: 1px solid var(--border); background: rgba(15, 23, 42, 0.4); }}
    .examples-cell {{ font-size: 12px; color: var(--muted-soft); }}
    .attention-badge {{ padding: 3px 8px; border-radius: 999px; font-size: 11px; display: inline-block; }}
    .attention-badge.极高 {{ background: rgba(251, 113, 133, 0.3); color: #fca5a5; border: 1px solid var(--danger); }}
    .attention-badge.高 {{ background: rgba(251, 191, 36, 0.3); color: #fcd34d; border: 1px solid var(--warning); }}
    .attention-badge.中高, .attention-badge.中等 {{ background: rgba(56, 189, 248, 0.3); color: #7dd3fc; border: 1px solid var(--accent); }}
    .attention-badge.低 {{ background: rgba(74, 222, 128, 0.3); color: #86efac; border: 1px solid var(--success); }}

    /* Stock table specific styles */
    .stock-table {{ background: rgba(15, 23, 42, 0.2); }}
    .stock-name {{ min-width: 120px; }}
    .stock-code {{ min-width: 80px; color: var(--muted); }}
    .stock-change {{ min-width: 100px; text-align: center; }}
    .stock-heat {{ min-width: 120px; }}
    .stock-heat .heat-num {{ color: var(--muted-soft); font-size: 11px; }}
    .stock-summary {{ font-size: 12px; color: var(--text); max-width: 300px; }}

    /* Stock badge styles - 徽章样式涨跌色块 */
    .stock-badge {{
      display: inline-block;
      padding: 6px 14px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 13px;
      white-space: nowrap;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }}
    .stock-badge-up {{
      background: linear-gradient(135deg, rgba(74, 222, 128, 0.28), rgba(34, 197, 94, 0.22));
      color: #86efac;
      border: 1px solid rgba(74, 222, 128, 0.5);
    }}
    .stock-badge-down {{
      background: linear-gradient(135deg, rgba(251, 113, 133, 0.28), rgba(239, 68, 68, 0.22));
      color: #fca5a5;
      border: 1px solid rgba(251, 113, 133, 0.5);
    }}
    .stock-badge-flat {{
      background: linear-gradient(135deg, rgba(148, 163, 184, 0.28), rgba(100, 116, 139, 0.22));
      color: #cbd5e1;
      border: 1px solid rgba(148, 163, 184, 0.5);
    }}

    .events-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin-top: 10px; }}
    .event-card {{ padding: 14px; border-radius: var(--radius-md);
      background: radial-gradient(circle at 0 0, rgba(56,189,248,.12), transparent), #020617;
      border: 1px solid var(--border); transition: transform 0.2s, box-shadow 0.2s; }}
    .event-card:hover {{ transform: translateY(-2px); box-shadow: 0 12px 30px rgba(0,0,0,0.5); }}
    .event-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .event-category {{ font-size: 13px; color: var(--accent); font-weight: 600; }}
    .risk-badge {{ padding: 3px 8px; border-radius: 999px; font-size: 11px; }}
    .risk-badge.danger {{ background: rgba(251, 113, 133, 0.3); color: #fca5a5; border: 1px solid var(--danger); }}
    .risk-badge.warning {{ background: rgba(251, 191, 36, 0.3); color: #fcd34d; border: 1px solid var(--warning); }}
    .risk-badge.success {{ background: rgba(74, 222, 128, 0.3); color: #86efac; border: 1px solid var(--success); }}
    .event-title {{ margin: 0 0 8px; font-size: 15px; font-weight: 600; color: #e5e7eb; }}
    .event-meta {{ font-size: 12px; color: var(--muted-soft); margin-bottom: 8px; }}
    .event-summary {{ margin: 0; font-size: 13px; line-height: 1.5; color: var(--text); }}
    .event-impact {{ margin-top: 10px; padding: 8px; background: rgba(15, 23, 42, 0.6); border-left: 3px solid var(--accent); font-size: 12px; }}
    .related-news {{ margin-top: 8px; font-size: 12px; color: var(--muted); }}
    .related-news summary {{ cursor: pointer; color: #c7d2fe; }}
    .related-news ul {{ margin: 6px 0 0 0; padding-left: 16px; }}
    .related-news a {{ color: #93c5fd; text-decoration: none; }}
    .related-news a:hover {{ text-decoration: underline; }}

    .sentiment-tag {{ padding: 4px 10px; border-radius: 999px; font-size: 12px; background: rgba(15, 23, 42, 0.8); display: inline-block; }}
    .severity-badge {{ padding: 3px 8px; border-radius: 999px; font-size: 11px; }}
    .severity-badge.danger {{ background: rgba(251, 113, 133, 0.3); color: #fca5a5; border: 1px solid var(--danger); }}
    .severity-badge.warning {{ background: rgba(251, 191, 36, 0.3); color: #fcd34d; border: 1px solid var(--warning); }}
    .severity-badge.success {{ background: rgba(74, 222, 128, 0.3); color: #86efac; border: 1px solid var(--success); }}
    .title-cell {{ max-width: 300px; }}
    .risk-cell {{ font-size: 12px; color: var(--muted-soft); }}

    .trends-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; margin-top: 10px; }}
    .trend-card {{ padding: 14px; border-radius: var(--radius-md);
      background: radial-gradient(circle at 100% 0, rgba(129,140,248,.12), transparent), #020617;
      border: 1px solid var(--border); }}
    .trend-dimension {{ margin: 0 0 12px; font-size: 15px; font-weight: 600; color: var(--accent-strong); }}
    .trend-item {{ margin-bottom: 10px; }}
    .trend-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }}
    .trend-item p {{ margin: 0; font-size: 13px; line-height: 1.5; }}

    .papers-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; }}
    .paper-card {{ padding: 12px; border-radius: 12px; border: 1px solid var(--border); background: rgba(15,23,42,0.8); box-shadow: var(--shadow-soft); }}
    .paper-cat {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: rgba(56,189,248,0.16); color: #7dd3fc; font-size: 12px; margin-bottom: 6px; border: 1px solid rgba(59,130,246,0.35); }}
    .paper-title {{ font-size: 14px; font-weight: 600; margin-bottom: 6px; color: #e0ecff; line-height: 1.4; }}
    .paper-title a {{ color: #c7d2fe; text-decoration: none; }}
    .paper-title a:hover {{ text-decoration: underline; }}
    .paper-tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }}
    .paper-tag {{ padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.35); font-size: 11px; color: #e5e7eb; background: rgba(15,23,42,0.6); }}
    .paper-summary {{ font-size: 12px; color: var(--muted); line-height: 1.5; }}

    .content-box {{ padding: 14px 14px 12px; border-radius: 14px; border: 1px solid rgba(15, 23, 42, 0.9);
      background: radial-gradient(circle at 0 0, rgba(56,189,248,.12), transparent 55%),
        radial-gradient(circle at 100% 0, rgba(129,140,248,.12), transparent 55%), linear-gradient(180deg, #020617, #020617);
      min-height: 120px; box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.85); }}
    .content-box p {{ margin: 0 0 8px; line-height: 1.6; font-size: 13px; color: #e5e7eb; }}
    .mermaid {{ background: rgba(15,23,42,0.7); border: 1px solid rgba(56,189,248,0.2); border-radius: 12px; padding: 12px; }}

    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }}
    .chip {{ padding: 6px 12px; border-radius: var(--radius-sm); background: transparent;
      color: #e5e7eb; border: 1px solid rgba(148, 163, 184, 0.35); font-size: 12px; display: inline-flex;
      align-items: center; gap: 6px; white-space: nowrap; }}
    .chip::before {{ content: "#"; color: #a5b4fc; font-size: 11px; }}
    .empty {{ color: var(--muted); font-style: italic; font-size: 13px; }}

    .grid-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 8px; }}
    .card {{ padding: 10px 10px 12px; border-radius: 14px;
      background: radial-gradient(circle at 0% 0%, rgba(56,189,248,0.16), transparent 55%),
        radial-gradient(circle at 100% 0%, rgba(129,140,248,0.16), transparent 55%), #020617;
      border: 1px solid rgba(30, 64, 175, 0.7); box-shadow: 0 18px 40px rgba(15, 23, 42, 0.95);
      display: flex; flex-direction: column; gap: 8px; transition: transform 0.14s, box-shadow 0.14s, border-color 0.14s; }}
    .card:hover {{ transform: translateY(-3px); box-shadow: 0 22px 50px rgba(15, 23, 42, 0.98);
      border-color: rgba(56, 189, 248, 0.9); }}
    .meta-line {{ color: var(--muted); font-size: 11px; margin-bottom: 2px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
    .src-pill, .hot-pill {{ padding: 4px 8px; border-radius: 999px; border: 1px solid rgba(148, 163, 184, 0.35); background: transparent; color: #e5e7eb; font-size: 11px; }}
    .quote-pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 8px; border-radius: 999px; font-size: 12px; border: 1px solid rgba(148, 163, 184, 0.4); background: transparent; }}
    .quote-pill .price {{ font-weight: 700; color: #e0ecff; }}
    .quote-pill.up {{ color: #86efac; border-color: rgba(74, 222, 128, 0.6); }}
    .quote-pill.down {{ color: #fca5a5; border-color: rgba(251, 113, 133, 0.6); }}
    .quote-pill.flat {{ color: #e5e7eb; border-color: rgba(148, 163, 184, 0.5); }}
    .tag-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .tag {{ display: inline-flex; align-items: center; gap: 6px; padding: 0 2px; border-radius: 6px;
      background: transparent; color: #e5e7eb; border: 1px solid rgba(56, 189, 248, 0.45);
      font-size: 13px; text-decoration: none; transition: color 0.12s, border-color 0.12s;
      max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .tag:hover {{ background: transparent; transform: none;
      box-shadow: none; border-color: rgba(56, 189, 248, 0.8); text-decoration: underline; }}
    .tag .hot {{ color: var(--danger); font-weight: 600; }}
    .tag .src {{ color: #a5b4fc; font-size: 11px; }}
    .tag .time {{ color: var(--muted-soft); font-size: 11px; }}
    .tag span.sep {{ color: rgba(148, 163, 184, 0.6); }}
    p {{ margin: 0 0 8px; line-height: 1.6; font-size: 13px; }}
    iframe.graph-frame {{ width: 100%; height: 560px; border: 1px solid var(--border); border-radius: 12px; background: rgba(15,23,42,0.6); }}
    .img-row {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }}
    .img-row img {{ width: 90px; height: 60px; object-fit: cover; border-radius: 10px; border: 1px solid rgba(148,163,184,0.4); background: rgba(15,23,42,0.8); }}
    .card-cover {{ width: 100%; height: 140px; border-radius: 12px; overflow: hidden; border: 1px solid rgba(148,163,184,0.35); background: rgba(15,23,42,0.7); }}
    .card-cover img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
    .bar-group {{ display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }}
    .bar-row {{ display: grid; grid-template-columns: 120px 1fr 80px; gap: 8px; align-items: center; font-size: 12px; color: var(--muted-soft); }}
    .bar-track {{ background: rgba(148,163,184,0.15); border-radius: 999px; height: 8px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, #38bdf8, #6366f1); }}
    #rel-graph {{ width: 100%; height: 520px; background: rgba(15,23,42,0.6); border: 1px solid var(--border); border-radius: 12px; }}
    .tag-list .tag {{ background: transparent; border-color: rgba(59,130,246,0.35); color: #e5e7eb; }}
    .tag-list .tag:hover {{ background: transparent; }}
    .tag.title-tag {{ font-size: 16px; font-weight: 700; color: #e0e7ff; background: transparent; border: none; padding: 0; }}
    .badge-pill.tag-society {{ background: transparent; border-color: rgba(251, 146, 60, 0.55); color: #fdba74; }}
    .badge-pill.tag-tech {{ background: transparent; border-color: rgba(34, 197, 235, 0.55); color: #7dd3fc; }}
    .badge-pill.tag-politics {{ background: transparent; border-color: rgba(248, 113, 113, 0.55); color: #fecdd3; }}
    .badge-pill.tag-biz {{ background: transparent; border-color: rgba(250, 204, 21, 0.55); color: #facc15; }}
    .badge-pill.tag-ent {{ background: transparent; border-color: rgba(129, 140, 248, 0.55); color: #c7d2fe; }}
    .badge-pill.tag-default {{ background: transparent; border-color: rgba(148, 163, 184, 0.45); color: #e5e7eb; }}
    .sentiment-pie-box {{ display: flex; align-items: center; gap: 14px; margin-top: 12px; flex-wrap: wrap; }}
    .pie-chart {{ width: 120px; height: 120px; position: relative; }}
    .pie {{ width: 100%; height: 100%; border-radius: 50%; }}
    .pie-center {{ position: absolute; inset: 0; margin: auto; width: 54px; height: 54px; border-radius: 50%; background: #0b1220; display: flex; align-items: center; justify-content: center; color: #e5e7eb; font-size: 13px; border: 1px solid rgba(148,163,184,0.35); }}
    .pie-legend {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 6px 10px; font-size: 12px; color: var(--muted); }}
    .pie-legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
    .pie-legend .dot.pos {{ background: #4ade80; }}
    .pie-legend .dot.neu {{ background: #e2e8f0; }}
    .pie-legend .dot.neg {{ background: #f87171; }}
    .topic-group {{ border: 1px solid rgba(148,163,184,0.2); border-radius: 14px; background: rgba(15,23,42,0.5); margin-bottom: 12px; padding: 10px; }}
    .topic-group[open] summary::marker {{ display: none; }}
    .topic-header {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; cursor: pointer; }}
    .topic-count {{ color: var(--muted); font-size: 12px; }}
    summary {{ list-style: none; }}

    @media (max-width: 880px) {{
      .overview-cards {{ grid-template-columns: 1fr; }}
      .mini-stats {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 600px) {{
      .grid-3 {{ grid-template-columns: minmax(0, 1fr); }}
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <script src="https://unpkg.com/force-graph"></script>
  <script>
    if (window.mermaid) {{
      mermaid.initialize({{ startOnLoad: true, theme: "dark" }});
    }}
  </script>
</head>
<body>
  <div class="shell">
    <div class="top-bar">
      <div class="brand">
        <span class="brand-dot"></span>
        <span>Daily Signal Board</span>
      </div>
      <div class="pill-soft">
        <span class="pill-soft-dot"></span>
        <span>实时舆情 · 在线分析</span>
      </div>
    </div>

        <div class="hero">
      <div class="hero-inner">
        <div class="hero-top">
          <div class="hero-main">
            <div class="pill">新闻收集报告</div>
            <h1>{date_str} · {_safe(task)} <span class="date">Daily Snapshot</span></h1>
            <div class="meta">生成时间：{generated_at} <span class="dot"></span> 时区：{settings.TIMEZONE}</div>
          </div>
          <div class="hero-badge">
            <div class="hero-badge-label">今日概览</div>
            <div class="hero-badge-main">
              <span class="hero-badge-value">每日要闻</span>
              <span class="hero-badge-tag">{len(text_news) + len(video_news)} 条信号</span>
            </div>
            <div class="hero-badge-sub">文字 {len(text_news)} · 视频 {len(video_news)}</div>
          </div>
        </div>
        {_render_hero_overview(overview) if report_data else ""}
      </div>
    </div>

    {"" if not report_data else f'''
    <div class="section">
      <div class="section-inner">
        <div class="section-header">
            <div class="section-title">
            <div class="section-eyebrow">Section 02</div>
            <h2>重点事件</h2>
          </div>
        </div>
        {_render_key_events(key_events)}
        <div style="margin-top:14px;">
          <div class="section-eyebrow">Research</div>
          <h3 style="margin:6px 0 8px;">重点论文</h3>
          {_render_papers(daily_papers)}
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-title">
            <div class="section-eyebrow">Section 03</div>
            <h2>时间线与趋势洞察</h2>
          </div>
        </div>
        <div class="two-col">
          <div><div class="content-box">{_render_timeline_graph(timeline_analysis, key_events)}</div></div>
          <div><div class="content-box">{_to_paragraphs(trend_analysis)}</div></div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-title">
            <div class="section-eyebrow">Section 04</div>
            <h2>关系图谱</h2>
          </div>
        </div>
        <div class="content-box">
          { '<div id="rel-graph"></div>' if (edges or nodes_list) else _to_paragraphs(relationship_plain) }
        </div>
      </div>
    </div>

    '''}

    {"" if report_data else f'''
    <div class="section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-title">
            <div class="section-eyebrow">Section 01</div>
            <h2>最终报告</h2>
          </div>
        </div>
        <div class="content-box">{_to_paragraphs(final_report)}</div>
      </div>
    </div>
    '''}

    <div class="section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-title">
            <div class="section-eyebrow">Section 05</div>
            <h2>文字新闻详情</h2>
          </div>
          <div class="section-desc">共收集 {len(text_news)} 条文字新闻</div>
        </div>
        <div class="grid-cards">{_render_news_items(text_news, "暂无文字新闻", news_images)}</div>
      </div>
    </div>

    <div class="section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-title">
            <div class="section-eyebrow">Section 06</div>
            <h2>视频新闻 <span class="badge-soft">多模态信号</span></h2>
          </div>
          <div class="section-desc">共收集 {len(video_news)} 条视频新闻</div>
        </div>
        <div class="grid-cards">{_render_news_items(video_news, "暂无视频新闻", news_images)}</div>
      </div>
    </div>

    {f'''
    <div class="section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-title">
            <div class="section-eyebrow">Section 07</div>
            <h2>股票市场分析 <span class="badge-soft">雪球热榜</span></h2>
          </div>
          <div class="section-desc">从雪球新闻中提取的热门股票涨跌情况</div>
        </div>
        {_render_stocks(report_data.get("stocks", []))}
      </div>
    </div>
    ''' if report_data and report_data.get("stocks") else ''}

    <div class="section">
      <div class="section-inner">
        <div class="section-header">
          <div class="section-title">
            <div class="section-eyebrow">Section 08</div>
            <h2>趋势预测</h2>
          </div>
        </div>
        {_render_trends(trends)}
      </div>
    </div>
  </div>
</body>
</html>
<script>
  (function() {{
    const palette = ['#38bdf8', '#a5b4fc', '#f472b6', '#34d399', '#facc15', '#f97316', '#f87171', '#22d3ee', '#c084fc', '#67e8f9'];
    const pickColor = (id) => {{
      const code = Array.from(String(id || '')).reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
      return palette[code % palette.length];
    }};

    const rootId = '今日要闻';
    let edges = {json.dumps(edges if 'edges' in locals() else [], ensure_ascii=False)};
    const presetNodes = {json.dumps(nodes_list if 'nodes_list' in locals() else [], ensure_ascii=False)};

    // Build node map with a fixed root
    const nodesMap = new Map();
    nodesMap.set(rootId, {{ id: rootId, label: rootId, color: '#facc15', group: 'root', val: 10 }});

    if (!Array.isArray(edges)) edges = [];
    edges.forEach(e => {{
      if (!e) return;
      nodesMap.set(e.source, {{ id: e.source, label: e.source }});
      nodesMap.set(e.target, {{ id: e.target, label: e.target }});
    }});

    if (Array.isArray(presetNodes)) {{
      presetNodes.forEach(n => {{
        if (n && n.id) {{
          nodesMap.set(n.id, {{
            id: n.id,
            label: n.label || n.id,
            color: n.color,
            group: n.type,
            val: n.val || 4
          }});
        }}
      }});
    }}

    // Always connect every node to the root for可读性
    const rootLinks = [];
    nodesMap.forEach((_, id) => {{
      if (id !== rootId) {{
        rootLinks.push({{ source: rootId, target: id }});
      }}
    }});
    const links = edges.concat(rootLinks);

    if (links.length === 0 || !window.ForceGraph) return;

    // Apply palette colors
    nodesMap.forEach((node, id) => {{
      if (!node.color) node.color = pickColor(id);
      if (!node.val) node.val = id === rootId ? 10 : 5;
    }});

    const nodes = Array.from(nodesMap.values());
    const data = {{ nodes, links }};
    const container = document.getElementById('rel-graph');
    if (!container) return;

    const fg = ForceGraph()(container)
      .graphData(data)
      .nodeId('id')
      .nodeLabel(node => node.label || node.id)
      .nodeRelSize(6)
      .nodeAutoColorBy('group')
      .linkColor(() => 'rgba(56,189,248,0.6)')
      .nodeColor(node => node.color || pickColor(node.id))
      .backgroundColor('transparent')
      .linkWidth(() => 1.1)
      .cooldownTime(1200);

    fg.nodeCanvasObject((node, ctx, globalScale) => {{
      const label = node.label || node.id;
      const fontSize = 12 / globalScale;
      const r = (node.val || 5);

      // Node circle
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
      ctx.fillStyle = node.color || pickColor(node.id);
      ctx.fill();
      ctx.strokeStyle = 'rgba(15,23,42,0.35)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label above the node
      ctx.font = `${{fontSize}}px "Inter","Segoe UI",sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = '#e5e7eb';
      ctx.fillText(label, node.x, node.y - r - 4);
    }});

    // 初始视角聚焦中心节点
    setTimeout(() => {{
      try {{
        fg.zoom(1.4, 400);
        fg.centerAt(0, 0, 400);
      }} catch (err) {{
        console.error(err);
      }}
    }}, 200);
  }})();
</script>
"""
        with report_path.open("w", encoding="utf-8") as fp:
            fp.write(html)
        logger.info("HTML report saved to %s", report_path)
        return report_path
    except Exception:
        logger.exception("Failed to save HTML report")
        return None


def _send_report_email(
    date_str: str,
    task: str,
    html_path: Optional[Path],
    summary_text: str,
) -> None:
    """Send the HTML report via SMTP if email is enabled and configured."""
    if not settings.EMAIL_ENABLED:
        logger.info("Email notification is disabled; skip sending report email.")
        return

    recipients = [addr.strip() for addr in (settings.EMAIL_TO or "").split(",") if addr.strip()]
    if not recipients:
        logger.warning("Email notification enabled but EMAIL_TO is empty; skip sending.")
        return

    host = settings.EMAIL_SMTP_HOST
    user = settings.EMAIL_SMTP_USER
    password = settings.EMAIL_SMTP_PASSWORD
    port = settings.EMAIL_SMTP_PORT
    sender = settings.EMAIL_FROM or settings.EMAIL_SMTP_USER

    if not (host and user and password and sender):
        logger.warning("Email notification missing SMTP config (host/user/password/from); skip sending.")
        return

    msg = EmailMessage()
    msg["Subject"] = f"{date_str} 新闻报告"
    msg["From"] = sender if "<" in sender else formataddr(("News Agent", sender))
    msg["To"] = ", ".join(recipients)

    # Plain text fallback for email clients that don't support HTML
    html_path_str = str(html_path) if html_path else "未生成 HTML 报告"
    plain_text = (
        f"{date_str} 新闻报告\n"
        f"任务: {task}\n\n"
        f"本地 HTML 路径: {html_path_str}\n"
        "若无法访问本地路径，请查收附件或在服务端查看 output/reports 目录。\n\n"
        f"摘要预览:\n{summary_text[:800]}\n\n"
        "请使用支持 HTML 的邮件客户端查看完整报告。"
    )
    msg.set_content(plain_text)

    # Add HTML version as the primary email body
    if html_path and html_path.exists():
        try:
            with html_path.open("r", encoding="utf-8") as fp:
                html_content = fp.read()

            # Set HTML as the preferred alternative
            msg.add_alternative(html_content, subtype="html")

            # Also attach HTML file for download
            with html_path.open("rb") as fp:
                html_bytes = fp.read()
            msg.add_attachment(
                html_bytes,
                maintype="text",
                subtype="html",
                filename=html_path.name,
            )
        except Exception:
            logger.warning("Attach HTML report failed", exc_info=True)

    try:
        context = ssl.create_default_context()
        if settings.EMAIL_USE_SSL:
            with smtplib.SMTP_SSL(host, port, timeout=15, context=context) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                if settings.EMAIL_USE_TLS:
                    server.starttls(context=context)
                server.login(user, password)
                server.send_message(msg)
        logger.info("Report email sent to: %s", recipients)
    except Exception:
        logger.warning("Send report email failed (ignored)", exc_info=True)

logger = logging.getLogger(__name__)

# Shared news service instance
_news_service = NewsNowService()


# ===== Coordinator Node =====

async def coordinator_node(state: State):
    """
    Initialize the system and preload tools.
    """
    logger.info("=== Coordinator: Initializing system ===")

    from src.tools import ALL_NEWS_TOOLS

    # Preload tools to avoid re-initialization
    text_tools = [tool for tool in ALL_NEWS_TOOLS if tool.name in {
        "get_latest_news", "search_news", "get_historical_news",
        "get_trending", "fetch_article",
        "deduplicate_news_items"
    }]

    video_tools = text_tools  # Video can use same tools

    research_tools = [tool for tool in ALL_NEWS_TOOLS if tool.name in {
        "tavily_search", "fetch_article"
    }]

    return Command(update={
        "started_at": datetime.now(),
        "iteration": 0,
        "quality_score": 0.0,
        "supervisor_decision": "collect",
        "last_agent": "coordinator",
        "text_tools": text_tools,
        "video_tools": video_tools,
        "research_tools": research_tools,
        "supervisor_questions": [],
        "research_notes": [],
        "text_news": [],
        "video_news": [],
        "news_pool": [],
        "news_pool_cursor": 0,
        "latest_news_batch": [],
        "supervisor_feedback": "",
    }, goto="main_supervisor")


# ===== News Collector Node =====

async def news_collector_node(state: State):
    """
    Dedicated node for fetching and preparing news data.
    Responsibilities:
    - Fetch news from NewsNowService
    - Deduplicate news
    - Extract trending keywords
    - Prepare news pool
    """
    logger.info("=== News Collector: Fetching news data ===")

    try:
        # Fetch daily papers (optional)
        daily_papers = await _fetch_daily_papers(limit=10)

        # Fetch latest news from all platforms
        news_result = await _news_service.get_latest_news(
            platforms=None,
            limit=_news_service.global_limit,
            include_url=True,
        )

        raw_items = news_result.get("items") or []
        logger.info(f"Fetched {len(raw_items)} raw news items from {len(news_result.get('platforms', []))} platforms")

        # Deduplicate
        dedup_result = deduplicate_news(
            raw_items,
            similarity_threshold=0.85,
            keep_duplicates_info=True
        )
        dedup_items = dedup_result.get("items", [])
        logger.info(f"After deduplication: {len(dedup_items)} items (removed {dedup_result.get('removed_duplicates', 0)} duplicates)")

        # 若上游无时间或仅有统一的 retrieved_at，按排名轻微错开时间，避免时间线全相同
        base_ts = datetime.now(timezone.utc).replace(microsecond=0)
        for idx, item in enumerate(dedup_items):
            raw_ts = item.get("timestamp") or item.get("published_at") or item.get("time")
            ts_source = "origin" if raw_ts else ""
            if not raw_ts:
                raw_ts = item.get("retrieved_at")
                ts_source = "retrieved"

            parsed_ts = None
            if raw_ts:
                try:
                    parsed_ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
                except Exception:
                    parsed_ts = None

            if parsed_ts is None:
                parsed_ts = base_ts - timedelta(seconds=idx)
                item["timestamp"] = parsed_ts.isoformat()
            else:
                if not item.get("timestamp"):
                    # 仅提供了其他字段时，统一写回 timestamp
                    item["timestamp"] = parsed_ts.isoformat()
                if ts_source == "retrieved":
                    # 上游只有抓取时间，则按排名增加轻微偏移
                    item["timestamp"] = (parsed_ts + timedelta(seconds=idx)).isoformat()

            if not item.get("retrieved_at"):
                item["retrieved_at"] = item["timestamp"]

        # Normalize to NewsItem format
        normalized = _normalize_news_items(dedup_items, default_category="text")

        # Extract keywords：BM25 粗选 → LLM 精排 → 回退
        bm25_candidates = _bm25_keywords(normalized, top_k=30)
        trending_keywords = await _extract_keywords_via_llm(normalized, top_n=10) or bm25_candidates[:10]
        if not trending_keywords:
            trending_keywords = _extract_keywords_from_news(normalized, top_n=10)

        # Apply hotness scores
        news_pool = _apply_hotness_scores(normalized, trending_keywords)

        logger.info(f"News pool prepared with {len(news_pool)} items")
        logger.info(f"Top keywords: {[kw.get('keyword') for kw in (trending_keywords or [])[:5]]}")

        return Command(update={
            "news_pool": news_pool,
            "news_pool_cursor": 0,
            "trending_keywords": trending_keywords,
            "last_agent": "collector",
            "daily_papers": daily_papers,
        }, goto="main_supervisor")

    except Exception as exc:
        logger.error(f"News collector failed: {exc}", exc_info=True)
        return Command(update={
            "supervisor_feedback": f"新闻收集失败: {str(exc)}",
            "last_agent": "collector",
            "daily_papers": [],
        }, goto="main_supervisor")


# ===== Main Supervisor Node =====

async def main_supervisor_node(state: State):
    """
    Main supervisor - makes high-level decisions only.

    Responsibilities:
    - Evaluate current state quality
    - Decide next team to execute
    - Control iteration flow

    NOT responsible for:
    - Data fetching (done by collector)
    - Detailed routing (done by team coordinators)
    """
    logger.info(f"=== Main Supervisor: Evaluating (Iteration {state['iteration']}) ===")

    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    news_pool = state.get("news_pool", [])
    news_pool_cursor = state.get("news_pool_cursor", 0)
    text_news_count = len(state.get("text_news", []))
    video_news_count = len(state.get("video_news", []))
    last_agent = state.get("last_agent", "coordinator")
    gap_pending = bool(state.get("analysis_gap_pending", False))
    analysis_cycle_count = state.get("text_team_cycle_count", 0)
    analysis_cycle_limit = getattr(settings, "ANALYSIS_MAX_CYCLES", 3)

    # Calculate quality score
    min_text = max(1, settings.MIN_TEXT_NEWS)
    min_video = max(0, settings.MIN_VIDEO_NEWS)
    target_total = min_text + min_video
    quality_score = min(1.0, (text_news_count + video_news_count) / float(target_total))

    logger.info(f"Quality: {quality_score:.2f} | Text: {text_news_count}/{min_text} | Video: {video_news_count}/{min_video}")

    # Decision logic
    decision = ""
    feedback = ""
    next_node = ""

    # Pool collected, need to distribute to collection team
    if last_agent == "collector":
        # Just finished collecting, now distribute to collection team
        # Prepare batch for collection team
        remaining = max(0, len(news_pool) - news_pool_cursor)
        batch_size = settings.SUPERVISOR_BATCH_SIZE
        if batch_size <= 0 or batch_size > remaining:
            batch_size = remaining
        batch = _sample_batch_diverse(news_pool, news_pool_cursor, batch_size)
        new_cursor = min(news_pool_cursor + batch_size, len(news_pool))

        decision = "collection_team"
        next_node = "collection_team"
        feedback = f"分派 {len(batch)} 条新闻给收集团队处理"

        return Command(update={
            "latest_news_batch": batch,
            "news_pool_cursor": new_cursor,
            "supervisor_decision": decision,
            "supervisor_feedback": feedback,
            "quality_score": quality_score,
        }, goto=next_node)

    # Collection team finished, check if need analysis
    elif last_agent == "collection_team":
        # Check iteration limit first
        if iteration >= max_iterations:
            # Hit iteration limit, move to summarize
            decision = "summarize"
            next_node = "summary_team"
            feedback = f"达到最大迭代次数 ({max_iterations})，开始总结"
        # Check if we have enough basic collection
        elif text_news_count >= min_text and video_news_count >= min_video:
            # Move to analysis team
            decision = "analysis_team"
            next_node = "analysis_team"
            feedback = "基础收集完成，进入深度分析"
        else:
            # Need more collection, check if pool exhausted
            if news_pool_cursor >= len(news_pool):
                # Pool exhausted but not enough news - fetch more
                decision = "collect"
                next_node = "news_collector"
                feedback = "新闻池已用完，需要获取更多数据"
                iteration += 1
            else:
                # Continue collecting from pool
                decision = "collection_team"
                next_node = "collection_team"
                feedback = "继续从新闻池分派下一批数据"
                iteration += 1

    # Analysis team finished, check quality
    elif last_agent == "analysis_team":
        # Check final quality
        has_research = len(state.get("research_notes", [])) > 0
        has_sentiment = state.get("text_sentiment") is not None
        has_relationship = state.get("text_relationship_graph") is not None

        quality_ok = quality_score >= settings.QUALITY_THRESHOLD
        iteration_limit = iteration >= max_iterations

        # 如果内层循环已到上限，直接进入总结避免无限重入
        if gap_pending and analysis_cycle_count >= analysis_cycle_limit:
            decision = "summarize"
            next_node = "summary_team"
            feedback = f"内层分析循环已达上限 ({analysis_cycle_limit})，直接输出总结"
        elif not gap_pending and (quality_ok or iteration_limit or has_research or has_sentiment or has_relationship):
            decision = "summarize"
            next_node = "summary_team"
            feedback = f"分析完成（质量: {quality_score:.2f}），开始生成报告"
        elif gap_pending and iteration_limit:
            decision = "summarize"
            next_node = "summary_team"
            feedback = "达到迭代上限，信息缺口仍存在，直接输出总结"
        else:
            # Need another analysis round guided by reflect
            decision = "analysis_team"
            next_node = "analysis_team"
            feedback = "信息缺口未补全，继续分析循环"
            iteration += 1

    # Default: first run or need to continue
    else:
        # Check if we need to fetch news first
        if len(news_pool) == 0:
            decision = "collect"
            next_node = "news_collector"
            feedback = "新闻池为空，需要收集数据"
        elif iteration >= max_iterations:
            decision = "summarize"
            next_node = "summary_team"
            feedback = f"达到最大迭代次数 ({max_iterations})"
        else:
            decision = "collection_team"
            next_node = "collection_team"
            feedback = "继续收集"
            iteration += 1

    logger.info(f"Decision: {decision} -> {next_node} | Feedback: {feedback}")

    return Command(update={
        "iteration": iteration,
        "supervisor_decision": decision,
        "supervisor_feedback": feedback,
        "quality_score": quality_score,
    }, goto=next_node)


# ===== Collection Team =====

async def collection_team_coordinator(state: State):
    """
    Collection team coordinator - routes to text collector first.

    Note: Attempted to use Send API for parallel execution, but current LangGraph
    version/setup doesn't support returning Send lists from nodes in flat workflows.
    Using sequential execution via Command for now.
    """
    logger.info("=== Collection Team: Coordinator starting ===")

    news_batch = state.get("latest_news_batch", [])

    if not news_batch:
        logger.warning("No news batch provided to collection team")
        return Command(update={
            "last_agent": "collection_team"
        }, goto="main_supervisor")

    # Split news by category
    text_batch = [item for item in news_batch if item.get("category") == "text"]
    video_batch = [item for item in news_batch if item.get("category") == "video"]

    logger.info(f"Batch size: {len(text_batch)} text, {len(video_batch)} video")

    # Store batches and route to text collector first
    return Command(update={
        "text_batch": text_batch,
        "video_batch": video_batch,
        "assigned_news": text_batch,
    }, goto="text_collector_agent")


async def text_collector_agent(state: State):
    """
    Text collector agent - collects and analyzes text news.
    """
    logger.info("=== Text Collector Agent: Processing ===")

    assigned_news = state.get("assigned_news", [])

    llm = get_llm_by_type(AGENT_LLM_MAP.get("text_agent", "qwen"))

    # Build context message
    news_context = "\n".join([
        f"- {item.get('title')} ({item.get('source')})"
        for item in assigned_news[:20]
    ])

    message = f"""任务：{state.get('task', '分析新闻')}

当前批次有 {len(assigned_news)} 条文字新闻。

新闻列表：
{news_context}

请分析这些新闻的关键主题和趋势，提取要点。"""

    try:
        response = await llm.ainvoke([HumanMessage(content=message)])
        analysis = _stringify_message_content(response.content) or "无分析结果"

        # Use assigned news
        all_text_news = (state.get("text_news") or []) + list(assigned_news)

        logger.info(f"Text collector processed {len(all_text_news)} news items")

        _persist_step_snapshot(
            state.get("date", datetime.now().strftime("%Y-%m-%d")),
            state.get("iteration", 0),
            "text_collector",
            {"news_count": len(all_text_news), "analysis_length": len(analysis)}
        )

        return Command(update={
            "text_news": all_text_news,
            "text_analysis": analysis,
            "assigned_news": state.get("video_batch", []),
        }, goto="video_collector_agent")

    except Exception as exc:
        logger.error(f"Text collector failed: {exc}", exc_info=True)
        return Command(update={
            "text_news": assigned_news,
            "text_analysis": f"分析失败: {str(exc)}",
            "assigned_news": state.get("video_batch", []),
        }, goto="video_collector_agent")


async def video_collector_agent(state: State):
    """
    Video collector agent - collects and analyzes video news.
    """
    logger.info("=== Video Collector Agent: Processing ===")

    assigned_news = state.get("assigned_news", [])

    # Skip if no video news
    if not assigned_news:
        logger.info("No video news to process")
        return Command(update={
            "video_news": [],
            "video_analysis": None,
        }, goto="collection_team_merger")

    llm = get_llm_by_type(AGENT_LLM_MAP.get("video_agent", "qwen_vl"))

    news_context = "\n".join([
        f"- {item.get('title')} ({item.get('source')})"
        for item in assigned_news[:20]
    ])

    message = f"""任务：{state.get('task', '分析新闻')}

当前批次有 {len(assigned_news)} 条视频新闻。

新闻列表：
{news_context}

请分析这些视频新闻的关键主题。"""

    try:
        response = await llm.ainvoke([HumanMessage(content=message)])
        analysis = _stringify_message_content(response.content) or "无分析结果"
        all_video_news = (state.get("video_news") or []) + list(assigned_news)

        logger.info(f"Video collector processed {len(all_video_news)} news items")

        _persist_step_snapshot(
            state.get("date", datetime.now().strftime("%Y-%m-%d")),
            state.get("iteration", 0),
            "video_collector",
            {"news_count": len(all_video_news), "analysis_length": len(analysis)}
        )

        return Command(update={
            "video_news": all_video_news,
            "video_analysis": analysis,
        }, goto="collection_team_merger")

    except Exception as exc:
        logger.error(f"Video collector failed: {exc}", exc_info=True)
        return Command(update={
            "video_news": assigned_news,
            "video_analysis": f"分析失败: {str(exc)}",
        }, goto="collection_team_merger")


def collection_team_merger(state: State):
    """
    Merger node for collection team results.
    LangGraph automatically merges results from parallel agents.
    """
    logger.info("=== Collection Team: Merging results ===")

    text_count = len(state.get("text_news", []))
    video_count = len(state.get("video_news", []))

    logger.info(f"Collection team completed: {text_count} text, {video_count} video")

    return Command(update={
        "last_agent": "collection_team",
    }, goto="main_supervisor")


# ===== Analysis Team =====

async def analysis_team_coordinator(state: State):
    """
    Analysis team coordinator - routes to first analysis agent.

    Note: Sequential execution (research -> sentiment -> relationship) instead of
    parallel Send API due to current LangGraph limitations in flat workflows.
    """
    logger.info("=== Analysis Team: Coordinator starting ===")

    text_news = state.get("text_news", [])

    if not text_news:
        logger.warning("No text news for analysis team")
        return Command(update={
            "last_agent": "analysis_team"
        }, goto="main_supervisor")

    logger.info("Starting sequential analysis: research -> sentiment -> relationship")

    gap_fill_mode = bool(state.get("analysis_gap_pending"))
    return Command(update={"gap_fill_mode": gap_fill_mode}, goto="research_agent")


async def research_agent_node(state: State):
    """
    Research agent - enriches hot news with background information.
    """
    logger.info("=== Research Agent: Starting ===")

    text_news = state.get("text_news", [])

    if not text_news:
        return {"research_notes": []}

    # Pick top news for research
    top_news = sorted(
        text_news,
        key=lambda x: x.get("hot_score", 0),
        reverse=True
    )[:5]

    llm = get_llm_by_type(AGENT_LLM_MAP.get("research", "qwen"))
    agent = create_react_agent(llm, [tavily_search])

    news_list = _build_safe_news_list(top_news)
    message = f"""请对以下热点新闻（标题已脱敏）做背景研究，必要时调用 tavily_search 补充信息，最终简洁总结要点：

{news_list}
"""

    try:
        result = await agent.ainvoke({
            "messages": [
                SystemMessage(content="你是新闻研究员，必要时用 tavily_search 查背景，输出中文要点，简洁列出事实。"),
                HumanMessage(content=message),
            ]
        })
        last_msg = result.get("messages", result)
        if isinstance(last_msg, list):
            research_text = _stringify_message_content(last_msg[-1].content)
        else:
            research_text = _stringify_message_content(last_msg)
        research_text = research_text or "无研究结果"

        # 【双层循环策略】根据reflect的指令进行针对性Extract
        enriched_notes = []
        information_gaps = state.get("information_gaps", [])

        # 如果reflect发现了信息缺口，优先Extract这些新闻
        if information_gaps:
            logger.info(f"=== Reflect指导Extract: 发现{len(information_gaps)}个信息缺口 ===")
            target_news = []

            # 从gaps中提取需要Extract的新闻
            for gap in information_gaps[:5]:  # 最多Extract 5条（提高覆盖率）
                gap_title = gap.get("news_title", "")
                if not gap_title:
                    continue

                # 清理标题（去掉【】和...）
                gap_title_clean = gap_title.replace("【", "").replace("】", "").replace("...", "").strip()

                # 在text_news中找到对应的新闻（使用模糊匹配）
                best_match = None
                best_score = 0

                for news_item in text_news:
                    news_title = news_item.get("title", "")
                    if not news_title:
                        continue

                    # 简单相似度：计算共同字符比例
                    common_chars = sum(1 for c in gap_title_clean if c in news_title)
                    score = common_chars / max(len(gap_title_clean), 1)

                    if score > best_score:
                        best_score = score
                        best_match = news_item

                # 如果相似度 > 0.6，认为匹配成功
                if best_match and best_score > 0.6:
                    if best_match not in target_news:  # 避免重复
                        target_news.append(best_match)

            logger.info(f"找到{len(target_news)}条需要Extract的新闻")
        else:
            # 如果没有reflect指令，回退到默认策略：Extract热度最高的前3条
            logger.info("=== 默认策略: Extract热度最高的3条新闻 ===")
            target_news = top_news[:3]

        for idx, news_item in enumerate(target_news):
            try:
                news_title = news_item.get("title", "")
                if not news_title:
                    continue

                # Step 1: Search找到最相关的URL
                search_result = await tavily_search.ainvoke({
                    "query": news_title,
                    "max_results": 2,
                    "include_answer": False,
                    "include_images": False,
                })

                if not isinstance(search_result, dict) or not search_result.get("success"):
                    continue

                results = search_result.get("data", {}).get("results") or []
                if not results:
                    continue

                top_result = max(results, key=lambda x: x.get("score", 0))
                url = top_result.get("url")
                if not url:
                    continue

                # Step 2: Extract完整内容
                logger.info(f"Extracting detailed content for: {news_title[:50]}...")
                extract_result = await tavily_extract.ainvoke({
                    "urls": url,
                    "include_images": False,
                    "extract_depth": "basic",  # basic足够，控制成本
                    "fmt": "markdown",
                })

                if not isinstance(extract_result, dict) or not extract_result.get("success"):
                    continue

                data = extract_result.get("data", {})
                results_list = data.get("results", [])
                if not results_list:
                    continue

                raw_content = results_list[0].get("raw_content", "")
                if not raw_content or len(raw_content) < 100:
                    continue

                # Step 3: 用LLM从完整内容提取关键信息（时间、事件细节等）
                content_snippet = raw_content[:3000]  # 限制长度
                extraction_prompt = f"""从以下网页内容中提取关键信息：

新闻标题：{news_title}

网页内容：
{content_snippet}

请提取并输出（中文，简洁）：
1. 事件发生的准确时间（精确到日期或时间）
2. 事件的关键细节和背景
3. 涉及的重要人物或机构
4. 事件影响和后续发展

输出格式：
时间：[时间]
细节：[关键细节]
相关方：[人物/机构]
影响：[影响评估]"""

                try:
                    detail_response = await llm.ainvoke([HumanMessage(content=extraction_prompt)])
                    extracted_details = _stringify_message_content(detail_response.content).strip()

                    if extracted_details and len(extracted_details) > 50:
                        enriched_notes.append({
                            "agent": "research_extract",
                            "news_title": news_title,
                            "content": extracted_details,
                            "source_url": url,
                            "timestamp": datetime.now().isoformat()
                        })
                        logger.info(f"Successfully enriched: {news_title[:40]}...")

                except Exception as e:
                    logger.debug(f"LLM extraction failed for {news_title}: {e}")

            except Exception as e:
                logger.warning(f"Failed to enrich news item {idx}: {e}")

        # 合并原始research和增强的notes
        notes = [{
            "agent": "research",
            "content": research_text,
            "timestamp": datetime.now().isoformat()
        }]
        notes.extend(enriched_notes)
        existing_notes = state.get("research_notes") or []
        merged_notes = existing_notes + notes

        logger.info(f"Research completed: {len(research_text)} chars, {len(enriched_notes)} items enriched with Extract")

        _persist_step_snapshot(
            state.get("date", datetime.now().strftime("%Y-%m-%d")),
            state.get("iteration", 0),
            "research",
            {"research_length": len(research_text)}
        )

        def _collect_tavily_images(payload: Dict[str, Any]) -> List[str]:
            """从 Tavily 结果提取图片 URL，兼容不同字段。"""
            urls: List[str] = []
            root_imgs = payload.get("images") or payload.get("image_results") or []
            for img in root_imgs:
                if isinstance(img, str):
                    urls.append(img)
                elif isinstance(img, dict):
                    url = img.get("url") or img.get("src")
                    if isinstance(url, str):
                        urls.append(url)
                    for k in ("image_url", "thumbnail"):
                        v = img.get(k)
                        if isinstance(v, str):
                            urls.append(v)

            for res in payload.get("results") or []:
                if not isinstance(res, dict):
                    continue
                for key in ("images", "image_results"):
                    for img in res.get(key) or []:
                        if isinstance(img, str):
                            urls.append(img)
                        elif isinstance(img, dict):
                            url = img.get("url") or img.get("src")
                            if isinstance(url, str):
                                urls.append(url)
                            for k in ("image_url", "thumbnail"):
                                v = img.get(k)
                                if isinstance(v, str):
                                    urls.append(v)
                for key in ("image", "thumbnail", "image_url"):
                    val = res.get(key)
                    if isinstance(val, str):
                            urls.append(val)

        def _is_search_page(url: str) -> bool:
            """粗略判断是否是搜索结果页，避免 Extract 白跑。"""
            try:
                parsed = urlparse(url)
                host = (parsed.netloc or "").lower()
                path = (parsed.path or "").lower()
                qs = (parsed.query or "").lower()
                if any(h in host for h in ["baidu.", "bing.", "google.", "search.bilibili", "sogou.", "sm.cn"]):
                    return True
                if "search" in path or "query" in path:
                    return True
                if any(k in qs for k in ["wd=", "q=", "keyword="]):
                    return True
            except Exception:
                return False
            return False

        async def _collect_images_via_extract(url: Optional[str]) -> List[str]:
            if not url:
                return []
            try:
                resp = await tavily_extract.ainvoke({
                    "urls": [url],
                    "include_images": True,
                    "include_favicon": False,
                    "extract_depth": "advanced",
                    "fmt": "markdown",
                })
            except Exception:
                logger.debug("Tavily Extract failed for %s", url, exc_info=True)
                return []
            if not isinstance(resp, dict) or not resp.get("success"):
                return []
            data = resp.get("data", {}) or {}
            results = data.get("results") or []
            urls: List[str] = []
            for res in results:
                if not isinstance(res, dict):
                    continue
                for img in res.get("images") or []:
                    if isinstance(img, str):
                        urls.append(img)
                    elif isinstance(img, dict):
                        u = img.get("url") or img.get("src") or img.get("image_url")
                        if isinstance(u, str):
                            urls.append(u)
            # 去重
            seen = set()
            uniq: List[str] = []
            for u in urls:
                if u in seen:
                    continue
                seen.add(u)
                uniq.append(u)
            return uniq

            # 去重保持顺序
            seen = set()
            uniq: List[str] = []
            for u in urls:
                if u in seen:
                    continue
                seen.add(u)
                uniq.append(u)
            return uniq

        async def _fetch_images_for_news(items: Sequence[NewsItem]) -> Dict[str, List[str]]:
            images_map: Dict[str, List[str]] = {}
            for item in items:
                title = str(item.get("title") or "").strip()
                if not title:
                    continue
                try:
                    urls: List[str] = []
                    target_url = item.get("url") or item.get("link") or item.get("mobile_url")
                    if target_url and not _is_search_page(target_url):
                        logger.info("Tavily Extract fetching images for title: %s | url: %s", title, target_url)
                        urls.extend(await _collect_images_via_extract(target_url))
                    else:
                        logger.debug("Skip direct extract (search page or missing url) for title: %s | url: %s", title, target_url)

                    # 如果原始链接是搜索页或无图，回退直接按标题搜索图片，再对搜索结果链接尝试 extract
                    if not urls:
                        resp = await tavily_search.ainvoke({
                            "query": title,
                            "max_results": 6,
                            "include_answer": False,
                            "include_images": True,
                            "include_image_descriptions": False,
                        })
                        if isinstance(resp, dict) and resp.get("success"):
                            data_block = resp.get("data", {}) or {}
                            imgs = _collect_tavily_images(data_block)
                            urls.extend(str(u) for u in imgs if isinstance(u, str))
                            search_results = data_block.get("results") or []
                            for r in search_results[:3]:
                                if not isinstance(r, dict):
                                    continue
                                link = r.get("url")
                                if not isinstance(link, str):
                                    continue
                                if _is_search_page(link):
                                    continue
                                logger.info("Tavily Extract fallback via search result for title: %s | url: %s", title, link)
                                urls.extend(await _collect_images_via_extract(link))
                                if urls:
                                    break
                    if urls:
                        uniq = []
                        seen = set()
                        for u in urls:
                            if u in seen:
                                continue
                            seen.add(u)
                            uniq.append(u)
                        images_map[title.lower()] = uniq[:3]
                        logger.info("Images found for %s: %s", title, len(uniq))
                    else:
                        logger.debug("No images found via Tavily for %s", title)
                except Exception:
                    logger.debug("Fetch images failed for %s", title, exc_info=True)
                    continue
            if images_map:
                summary = {k: len(v) for k, v in images_map.items()}
                logger.info("Tavily images summary (title->count): %s", summary)
            else:
                logger.info("Tavily images summary: none found")
            return images_map

        images_map = await _fetch_images_for_news(top_news)

        # 【智能路由】根据skip标志决定下一步
        skip_sentiment_relationship = bool(state.get("skip_sentiment_relationship", False))

        if skip_sentiment_relationship:
            # 优化路径：跳过sentiment和relationship，直接到reflect
            logger.info("⚡ 优化路径：跳过sentiment/relationship，直接到reflect")
            next_node = "reflect_agent"
        else:
            # 正常路径：继续sentiment分析
            next_node = "sentiment_agent"

        return Command(update={"research_notes": merged_notes, "news_images": images_map}, goto=next_node)

    except Exception as exc:
        if "data_inspection_failed" in str(exc):
            logger.warning("Research agent input hit DashScope moderation; titles redacted: %s", news_list)
            notes = [{
                "agent": "research",
                "content": "因模型内容安全校验，研究步骤已跳过（标题已脱敏仍被拦截）。",
                "timestamp": datetime.now().isoformat()
            }]
            return Command(update={"research_notes": notes}, goto="sentiment_agent")

        logger.error(f"Research agent failed: {exc}", exc_info=True)
        return Command(update={"research_notes": []}, goto="sentiment_agent")


async def sentiment_agent_node(state: State):
    """
    Sentiment agent - analyzes sentiment and risk.
    """
    logger.info("=== Sentiment Agent: Starting ===")

    text_news = state.get("text_news", [])

    if not text_news:
        return {"text_sentiment": None}

    llm = get_llm_by_type(AGENT_LLM_MAP.get("text_sentiment", "qwen"))
    system_prompt = get_system_prompt("text_sentiment")
    # Removed create_agent

    news_list = "\n".join([
        f"- {item.get('title')}"
        for item in text_news[:20]
    ])

    message = f"""分析以下新闻的情感倾向和关注度：

{news_list}

请输出表格，每行包含：标题、情感（正面/中性/负面）、关注度（高/中/低）、简短说明（避免夸张或煽动措辞）。"""

    try:
        response = await llm.ainvoke([HumanMessage(content=message)])

        sentiment = _stringify_message_content(response.content) or "无结果"

        logger.info(f"Sentiment analysis completed: {len(sentiment)} chars")

        _persist_step_snapshot(
            state.get("date", datetime.now().strftime("%Y-%m-%d")),
            state.get("iteration", 0),
            "sentiment",
            {"sentiment_length": len(sentiment)}
        )

        return Command(update={"text_sentiment": sentiment}, goto="relationship_agent")

    except Exception as exc:
        logger.error(f"Sentiment agent failed: {exc}", exc_info=True)
        return Command(update={"text_sentiment": None}, goto="relationship_agent")


async def relationship_agent_node(state: State):
    """
    Relationship agent - builds causal/relationship graph.
    """
    logger.info("=== Relationship Agent: Starting ===")

    text_news = state.get("text_news", [])

    if not text_news:
        return {"text_relationship_graph": None}

    llm = get_llm_by_type(AGENT_LLM_MAP.get("relationship_graph", "qwen"))
    system_prompt = get_system_prompt("relationship_graph")
    # Removed create_agent

    news_list = "\n".join([
        f"- {item.get('title')}"
        for item in text_news[:15]
    ])

    message = f"""基于以下新闻，生成关系图谱，请只输出 JSON，不要额外文字：

新闻列表：
{news_list}

输出格式（严格遵守）：
{{
  "nodes": [
    {{"id": "节点ID", "label": "显示名(简短中文)", "type": "人物/机构/事件/地点"}}
  ],
  "links": [
    {{"source": "节点ID", "target": "节点ID", "label": "关系描述(因果/协同/冲突/影响)" }}
  ]
}}

约束：
- 节点需同时提供 id 和 label(简短中文)，节点数 10-20，边数不少于 12。
- 只输出合法 JSON，不要 Markdown、不要 mermaid、不要换行注释，不要附加文本。"""

    try:
        response = await llm.ainvoke([HumanMessage(content=message)])

        graph = _stringify_message_content(response.content) or "无结果"

        logger.info(f"Relationship graph completed: {len(graph)} chars")

        _persist_step_snapshot(
            state.get("date", datetime.now().strftime("%Y-%m-%d")),
            state.get("iteration", 0),
            "relationship",
            {"graph_length": len(graph)}
        )

        return Command(update={"text_relationship_graph": graph}, goto="reflect_agent")

    except Exception as exc:
        logger.error(f"Relationship agent failed: {exc}", exc_info=True)
        return Command(update={"text_relationship_graph": None}, goto="reflect_agent")


async def reflect_agent_node(state: State):
    """
    Reflect agent - 审查信息完整性，识别需要深入研究的新闻缺口。

    双层循环的关键节点（使用结构化输出）：
    - 检查哪些新闻缺少时间、细节、证据等信息
    - 使用Pydantic模型精确识别信息缺口类型
    - 输出structured information_gaps列表，指导research_agent用Extract补充
    - 判断是否需要继续内层循环
    """
    logger.info("=== Reflect Agent: Starting information gap detection (structured) ===")

    text_news = state.get("text_news", [])
    text_analysis = state.get("text_analysis") or ""
    research_notes = state.get("research_notes", [])

    if not text_news:
        return Command(update={
            "information_gaps": [],
            "text_reflection": "无新闻数据"
        }, goto="analysis_team_merger")

    llm = get_llm_by_type(AGENT_LLM_MAP.get("text_reflect", "qwen"))
    system_prompt = get_system_prompt("text_reflect")

    # 构建审查上下文
    news_list = "\n".join([
        f"{idx+1}. 【{item.get('title', '未知标题')}】- 来源:{item.get('platform', '未知')}"
        for idx, item in enumerate(text_news[:15])  # 审查前15条
    ])

    research_summary = "\n".join([
        f"- {note.get('content', '')[:200]}"  # 只取前200字
        for note in research_notes[:3]
    ]) if research_notes else "暂无深入研究"

    message = f"""审查以下新闻的信息完整性：

【已收集新闻】
{news_list}

【已有分析】
{text_analysis[:500]}

【已有研究】
{research_summary}

请仔细识别信息缺口。对于每个有缺口的新闻，明确指出：
1. 缺失的信息类型（时间信息/背景信息/人物信息/事件细节/数据证据）
2. 具体缺少什么（例如：缺少事件发生的准确日期和时间）
3. 优先级（1-5，5最高）
4. 是否需要使用Tavily Extract深度提取

返回结果应包含：
- overall_assessment: 总体信息完整度评估
- information_gaps: 具体的信息缺口列表
- needs_continue: 是否建议继续补充
- suggestion: 下一步建议"""

    try:
        # 使用结构化输出
        structured_llm = llm.with_structured_output(ReflectionResult)
        response = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ])

        # response 现在是 ReflectionResult 对象
        reflection_result: ReflectionResult = response

        # 转换为字典格式（保持兼容性）
        information_gaps = [gap.model_dump() for gap in reflection_result.information_gaps]

        # 判断是否需要继续循环
        cycle_count = state.get("text_team_cycle_count", 0)
        should_continue = reflection_result.needs_continue and cycle_count < 3

        logger.info(f"Reflect completed (structured): {len(information_gaps)} gaps found, cycle={cycle_count}, continue={should_continue}")
        logger.info(f"Assessment: {reflection_result.overall_assessment}")

        # 详细记录每个gap
        for gap in reflection_result.information_gaps:
            logger.info(f"  Gap: {gap.news_title[:30]}... | Type: {gap.missing_type} | Priority: {gap.priority}")

        _persist_step_snapshot(
            state.get("date", datetime.now().strftime("%Y-%m-%d")),
            state.get("iteration", 0),
            "reflect",
            {
                "assessment": reflection_result.overall_assessment,
                "gaps_count": len(information_gaps),
                "cycle_count": cycle_count,
                "needs_continue": reflection_result.needs_continue
            }
        )

        return Command(update={
            "text_reflection": f"{reflection_result.overall_assessment}\n\n建议：{reflection_result.suggestion}",
            "information_gaps": information_gaps,
        }, goto="analysis_team_merger")

    except Exception as exc:
        logger.error(f"Reflect agent failed: {exc}", exc_info=True)
        logger.warning("Falling back to non-structured reflection")

        # 降级方案：使用简单文本分析
        return Command(update={
            "text_reflection": f"反思失败，错误: {str(exc)[:100]}",
            "information_gaps": []
        }, goto="analysis_team_merger")


def analysis_team_merger(state: State):
    """
    Merger node for analysis team results.

    双层循环智能控制（内层循环）：
    - 检查reflect是否发现信息缺口
    - 根据缺口类型智能选择路径：
      * 仅时间/背景信息 → research Extract → reflect（跳过sentiment/relationship）
      * 涉及事件细节/人物/数据 → research Extract → sentiment → relationship → reflect（完整重新分析）
    - 循环次数<3时继续，否则返回supervisor
    """
    logger.info("=== Analysis Team: Merging results (intelligent routing) ===")

    has_research = len(state.get("research_notes", [])) > 0
    has_sentiment = state.get("text_sentiment") is not None
    has_relationship = state.get("text_relationship_graph") is not None
    has_reflection = state.get("text_reflection") is not None

    information_gaps = state.get("information_gaps", [])
    cycle_count = state.get("text_team_cycle_count", 0)
    max_cycles = 3  # 内层循环最多3次

    logger.info(f"Analysis status: research={has_research}, sentiment={has_sentiment}, relationship={has_relationship}, reflection={has_reflection}")
    logger.info(f"Cycle status: count={cycle_count}/{max_cycles}, gaps={len(information_gaps)}")

    # 【内层循环决策】
    # 如果reflect发现信息缺口，且还没超过循环次数限制，继续循环
    if information_gaps and cycle_count < max_cycles:
        new_cycle_count = cycle_count + 1

        # 【智能路径选择】分析缺口类型
        gap_types = [gap.get("missing_type", "") for gap in information_gaps]

        # 只需要research的类型：时间信息、背景信息
        research_only_types = {"时间信息", "背景信息"}

        # 需要重新分析的类型：事件细节、人物信息、数据证据
        full_reanalysis_types = {"事件细节", "人物信息", "数据证据"}

        # 判断是否所有gaps都只需要research
        only_research_needed = all(
            gap_type in research_only_types
            for gap_type in gap_types
            if gap_type
        )

        if only_research_needed and gap_types:
            # 路径1：只回research Extract，然后直接到reflect
            logger.info(f"🔄 内层循环第{new_cycle_count}轮：仅需补充时间/背景信息，优化路径 research → reflect")
            logger.info(f"   Gap types: {set(gap_types)}")

            return Command(update={
                "text_team_cycle_count": new_cycle_count,
                "analysis_gap_pending": True,
                "gap_fill_mode": True,
                "skip_sentiment_relationship": True,  # 新增标志：跳过sentiment和relationship
            }, goto="research_agent")
        else:
            # 路径2：完整重新分析 research → sentiment → relationship → reflect
            logger.info(f"🔄 内层循环第{new_cycle_count}轮：需要深度分析，完整路径 research → sentiment → relationship → reflect")
            logger.info(f"   Gap types: {set(gap_types)}")

            return Command(update={
                "text_team_cycle_count": new_cycle_count,
                "analysis_gap_pending": True,
                "gap_fill_mode": True,
                "skip_sentiment_relationship": False,  # 不跳过
            }, goto="research_agent")
    else:
        # 循环结束，返回supervisor
        if information_gaps and cycle_count >= max_cycles:
            logger.info(f"⚠️ 达到内层循环上限({max_cycles}次)，尽管还有{len(information_gaps)}个缺口，仍返回supervisor")
        else:
            logger.info(f"✅ 内层循环完成：无信息缺口或已补充完成")

        # 重置循环计数，为下一次迭代准备
        return Command(update={
            "last_agent": "analysis_team",
            "text_team_cycle_count": 0,  # 重置计数
            "information_gaps": information_gaps if information_gaps else [],
            "analysis_gap_pending": False,
            "gap_fill_mode": False,
            "skip_sentiment_relationship": False,  # 重置标志
        }, goto="main_supervisor")


# ===== Summary Team =====

async def summary_team_coordinator(state: State):
    """
    Summary team coordinator - orchestrates summary generation.

    Unlike collection/analysis teams, summary runs sequentially:
    timeline -> trend -> chart -> writer
    """
    logger.info("=== Summary Team: Coordinator starting ===")

    return Command(goto="timeline_agent")


async def timeline_agent_node(state: State):
    """
    Timeline agent - extracts temporal sequence of events.
    """
    logger.info("=== Timeline Agent: Starting ===")

    text_news = state.get("text_news", [])

    if not text_news:
        return Command(update={"timeline_analysis": None}, goto="trend_agent")

    llm = get_llm_by_type(AGENT_LLM_MAP.get("summary_timeline", "qwen"))
    system_prompt = get_system_prompt("summary_timeline")
    # Removed create_agent

    async def _extract_time_via_tavily(title: str) -> Optional[str]:
        """调用 Tavily 为新闻推测发布时间。"""
        try:
            resp = await tavily_search.ainvoke({
                "query": title,
                "max_results": 3,
                "include_answer": False,
                "include_images": False,
                "include_image_descriptions": False,
            })
            if not isinstance(resp, dict) or not resp.get("success"):
                return None
            results = resp.get("data", {}).get("results") or []
            for r in results:
                if not isinstance(r, dict):
                    continue
                for key in ("publishedDate", "published_date", "date", "created_at", "time"):
                    ts = r.get(key)
                    parsed = _coerce_timestamp(ts)
                    if parsed:
                        return parsed
        except Exception:
            logger.debug("Timeline tavily lookup failed for %s", title, exc_info=True)
        return None

    async def _enrich_timestamps(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for item in items:
            cloned = dict(item)
            ts = _coerce_timestamp(cloned.get("timestamp"))
            if not ts:
                title = str(cloned.get("title") or "").strip()
                if title:
                    ts = await _extract_time_via_tavily(title)
            if ts:
                cloned["timestamp"] = ts
            enriched.append(cloned)
        return enriched

    top_for_timeline = text_news[:20]
    enriched_news = await _enrich_timestamps(top_for_timeline)

    news_list = "\n".join([
        f"- {item.get('title')} ({item.get('timestamp', 'N/A')})"
        for item in enriched_news
    ])

    message = f"""提取以下新闻的时间线：

{news_list}

按时间顺序梳理关键事件，保持中性，不写评价，不使用夸张/煽动词汇，不描述血腥或敏感细节。"""

    try:
        response = await llm.ainvoke([HumanMessage(content=message)])

        timeline = _stringify_message_content(response.content) or "无结果"

        logger.info(f"Timeline extracted: {len(timeline)} chars")

        _persist_step_snapshot(
            state.get("date", datetime.now().strftime("%Y-%m-%d")),
            state.get("iteration", 0),
            "timeline",
            {"timeline_length": len(timeline)}
        )

        return Command(update={"timeline_analysis": timeline}, goto="trend_agent")

    except Exception as exc:
        if "data_inspection_failed" in str(exc):
            logger.warning("Timeline agent hit content moderation, retrying with stricter prompt")
            try:
                news_list_safe = "\n".join([
                    f"- {item.get('title')} ({item.get('timestamp', 'N/A')})"
                    for item in text_news[:12]
                ])
                safe_msg = f"""仅列出时间线，不做评价，不含敏感词或血腥细节，保持客观中性。

{news_list_safe}

按时间顺序输出，每行“时间 - 事件标题/要点”。"""
                response = await llm.ainvoke([HumanMessage(content=safe_msg)])
                timeline = _stringify_message_content(response.content) or "无结果"
                _persist_step_snapshot(
                    state.get("date", datetime.now().strftime("%Y-%m-%d")),
                    state.get("iteration", 0),
                    "timeline_retry",
                    {"timeline_length": len(timeline)}
                )
                return Command(update={"timeline_analysis": timeline}, goto="trend_agent")
            except Exception:
                logger.error("Timeline agent retry still failed", exc_info=True)
        logger.error(f"Timeline agent failed: {exc}", exc_info=True)
        return Command(update={"timeline_analysis": None}, goto="trend_agent")


async def trend_agent_node(state: State):
    """
    Trend agent - analyzes trends and patterns.
    """
    logger.info("=== Trend Agent: Starting ===")

    trending_keywords = state.get("trending_keywords", [])
    text_news = state.get("text_news", [])

    llm = get_llm_by_type(AGENT_LLM_MAP.get("summary_trend", "qwen"))
    system_prompt = get_system_prompt("summary_trend")
    # Removed create_agent

    keywords_text = ", ".join([kw.get("keyword", "") for kw in (trending_keywords or [])[:10]])

    message = f"""分析新闻趋势：

收集到 {len(text_news)} 条文字新闻
热门关键词: {keywords_text}

识别主要趋势和未来走向。"""

    try:
        response = await llm.ainvoke([HumanMessage(content=message)])

        trend = _stringify_message_content(response.content) or "无结果"

        logger.info(f"Trend analysis completed: {len(trend)} chars")

        _persist_step_snapshot(
            state.get("date", datetime.now().strftime("%Y-%m-%d")),
            state.get("iteration", 0),
            "trend",
            {"trend_length": len(trend)}
        )

        return Command(update={"trend_analysis": trend}, goto="summary_chart_agent")

    except Exception as exc:
        error_str = str(exc)

        # 检查是否是内容审核错误
        if "data_inspection_failed" in error_str or "inappropriate content" in error_str:
            logger.warning(f"⚠️ Content moderation triggered, using fallback analysis")

            # Fallback: 提供基础的趋势分析（不调用 LLM）
            fallback_trend = f"""## 趋势分析（基于关键词统计）

收集到 {len(text_news)} 条新闻

**热门关键词**: {keywords_text}

**注**: 由于内容审核限制，本次分析采用基础统计方法。主要关注以下几个维度的热点话题。

（系统提示：部分内容因审核机制未能生成详细分析）
"""
            logger.info(f"Using fallback trend analysis: {len(fallback_trend)} chars")
            return Command(update={"trend_analysis": fallback_trend}, goto="summary_chart_agent")

        else:
            # 其他错误：记录并继续
            logger.error(f"Trend agent failed: {exc}", exc_info=True)
            return Command(update={"trend_analysis": "趋势分析暂时不可用"}, goto="summary_chart_agent")


async def summary_chart_agent_node(state: State):
    """
    Chart agent - generates visualization.
    """
    logger.info("=== Summary Chart Agent: Starting ===")

    # Chart generation temporarily disabled (function not available)
    # TODO: Implement chart generation or import from original nodes when available

    logger.info("Chart generation skipped (not implemented in v2)")

    return Command(update={"timeline_chart_path": None}, goto="summary_writer_agent")


async def summary_writer_agent_node(state: State):
    """
    Summary writer - generates final report using structured output.
    """
    logger.info("=== Summary Writer Agent: Starting ===")

    llm = get_llm_by_type(AGENT_LLM_MAP.get("summary", "qwen"))
    system_prompt = get_system_prompt("summary")

    # Gather all analysis (with safe defaults)
    text_analysis = state.get("text_analysis") or ""
    video_analysis = state.get("video_analysis") or ""
    research_notes = state.get("research_notes") or []
    sentiment = state.get("text_sentiment") or ""
    relationship = state.get("text_relationship_graph") or ""
    timeline = state.get("timeline_analysis") or ""
    trend = state.get("trend_analysis") or ""

    text_count = len(state.get("text_news", []))
    video_count = len(state.get("video_news", []))

    research_text = "\n".join([note.get("content", "") for note in research_notes])

    # Extract xueqiu (stock) news for analysis
    text_news = state.get("text_news", [])
    xueqiu_news = [
        item for item in text_news
        if item.get("source") == "雪球热榜" or "xueqiu" in str(item.get("url", "")).lower()
    ]
    xueqiu_titles = "\n".join([f"- {item.get('title', '')}" for item in xueqiu_news[:15]]) if xueqiu_news else "暂无雪球数据"

    # Prepare context for structured output
    context_message = f"""生成最终新闻报告：

统计数据（必须使用以下精确数字）：
- 文字新闻: {text_count} 条
- 视频新闻: {video_count} 条
- 新闻总数: {text_count + video_count} 条
- 雪球股票新闻: {len(xueqiu_news)} 条

分析内容：
【文字分析】
{text_analysis[:500]}

【视频分析】
{video_analysis[:500]}

【背景研究】
{research_text[:500]}

【情感分析】
{sentiment[:300]}

【关系图】
{relationship}

【时间线】
{timeline[:300]}

【趋势分析】
{trend[:300]}

【雪球股票热榜】（从以下标题中提取股票名称、涨跌趋势、热度）
{xueqiu_titles}

请根据以上信息生成一份结构化的新闻报告。
重要提示：
1. overview.total_news 必须填写 {text_count + video_count}
2. overview.text_news 必须填写 {text_count}
3. overview.video_news 必须填写 {video_count}
4. timeline_analysis、trend_analysis、relationship_graph 字段直接复用或凝练上面的对应分析，避免留空。
5. stocks 字段：如果有雪球数据，请从标题中提取股票信息（股票名称、涨跌趋势、热度评分等）；如果没有雪球数据，返回空数组 []。"""

    try:
        # Use structured output with Pydantic schema
        structured_llm = llm.with_structured_output(NewsReport)

        # Invoke with system prompt + user message
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context_message)
        ]

        report_obj: NewsReport = await structured_llm.ainvoke(messages)
        if report_obj is None:
            raise ValueError("Structured report is empty")

        # 补充关键分析字段，避免模型遗漏
        # 同时强制修正 overview 中的新闻统计数字（LLM 可能填错）
        corrected_overview = report_obj.overview.model_copy(
            update={
                "total_news": text_count + video_count,
                "text_news": text_count,
                "video_news": video_count,
            }
        )

        report_obj = report_obj.model_copy(
            update={
                "overview": corrected_overview,
                "timeline_analysis": getattr(report_obj, "timeline_analysis", None) or timeline,
                "trend_analysis": getattr(report_obj, "trend_analysis", None) or trend,
                "relationship_graph": getattr(report_obj, "relationship_graph", None) or relationship,
            }
        )

        # Convert Pydantic object to JSON string for storage
        report_json = json.dumps(report_obj.model_dump(), indent=2, ensure_ascii=False)

        logger.info(f"Structured report generated: {len(report_json)} chars")
        logger.info(f"Report contains: {len(report_obj.core_themes)} themes, "
                   f"{len(report_obj.key_events)} events, "
                   f"{len(report_obj.sentiment_risk)} risk items, "
                   f"{len(report_obj.trends)} trends")
        logger.info(f"Overview stats: total={report_obj.overview.total_news}, "
                   f"text={report_obj.overview.text_news}, "
                   f"video={report_obj.overview.video_news}")

        # Save HTML report (pass JSON string)
        html_path = _save_report_to_html(
            date_str=state.get("date", datetime.now().strftime("%Y-%m-%d")),
            task=state.get("task", ""),
            final_report=report_json,
            text_news=state.get("text_news", []),
            video_news=state.get("video_news", []),
            timeline_analysis=timeline,
            trend_analysis=trend,
            trending_keywords=state.get("trending_keywords"),
            relationship_graph=relationship,
            news_images=state.get("news_images", {}),
            daily_papers=state.get("daily_papers", []),
        )

        # Send email if enabled (use executive summary as preview)
        _send_report_email(
            date_str=state.get("date", datetime.now().strftime("%Y-%m-%d")),
            task=state.get("task", ""),
            html_path=html_path,
            summary_text=report_obj.executive_summary,
        )

        _persist_step_snapshot(
            state.get("date", datetime.now().strftime("%Y-%m-%d")),
            state.get("iteration", 0),
            "summary_writer",
            {
                "report_length": len(report_json),
                "themes_count": len(report_obj.core_themes),
                "events_count": len(report_obj.key_events),
            }
        )

        return {
            "final_report": report_json,
            "completed_at": datetime.now(),
            "last_agent": "summary_writer",
        }

    except Exception as exc:
        error_str = str(exc)

        # 检查是否是内容审核错误
        if "data_inspection_failed" in error_str or "inappropriate content" in error_str:
            logger.warning(f"⚠️ Content moderation triggered in summary writer, using fallback report")

            # Fallback: 生成基础的文本报告（不使用 LLM 结构化输出）
            fallback_report = f"""# 新闻日报

**日期**: {state.get("date", datetime.now().strftime("%Y-%m-%d"))}
**任务**: {state.get("task", "")}

## 数据概览

- 文字新闻: {text_count} 条
- 视频新闻: {video_count} 条

## 分析内容

### 时间线分析
{timeline}

### 趋势分析
{trend}

### 关系图谱
{relationship}

### 情感分析
{sentiment}

---
**注**: 由于内容审核限制，本报告采用基础格式。详细的结构化分析暂时无法生成。
"""
            logger.info(f"Using fallback text report: {len(fallback_report)} chars")

        else:
            # 其他错误
            logger.error(f"Summary writer failed: {exc}", exc_info=True)
            fallback_report = f"""# 报告生成失败

**错误信息**: {str(exc)}

**收集数据**:
- 文字新闻: {text_count} 条
- 视频新闻: {video_count} 条

请检查日志获取详细信息。
"""

        # 保存 fallback 报告到 HTML
        try:
            _save_report_to_html(
                date_str=state.get("date", datetime.now().strftime("%Y-%m-%d")),
                task=state.get("task", ""),
                final_report=fallback_report,
                text_news=state.get("text_news", []),
                video_news=state.get("video_news", []),
                timeline_analysis=state.get("timeline_analysis"),
                trend_analysis=state.get("trend_analysis"),
                trending_keywords=state.get("trending_keywords"),
                relationship_graph=state.get("text_relationship_graph"),
                news_images=state.get("news_images", {}),
                daily_papers=state.get("daily_papers", []),
            )
        except Exception:
            logger.warning("Fallback HTML save also failed", exc_info=True)

        return {
            "final_report": fallback_report,
            "completed_at": datetime.now(),
            "last_agent": "summary_writer",
        }
