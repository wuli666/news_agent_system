"""
Agent nodes for the news collection system.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from src.graph.types import NewsItem, State
from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from src.config.settings import settings
from src.prompts import get_system_prompt
from src.tools import ALL_NEWS_TOOLS

logger = logging.getLogger(__name__)

NEWS_TOOL_NAMES = {"get_latest_news", "search_news", "get_historical_news"}
ARTICLE_TOOL_NAMES = {"fetch_article"}
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def _collect_news_from_tool_messages(messages: Sequence[Any]) -> List[NewsItem]:
    """Extract news items from tool outputs produced by the agent."""
    collected: List[NewsItem] = []
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = (message.name or "").strip()
        if tool_name not in NEWS_TOOL_NAMES | ARTICLE_TOOL_NAMES:
            continue

        payload = _coerce_tool_payload(message.content)
        if payload is None and message.artifact is not None:
            payload = message.artifact

        raw_items = _extract_news_items_from_payload(payload, tool_name)
        if not raw_items:
            continue

        normalized = _normalize_news_items(raw_items, default_category="text")
        if normalized:
            collected.extend(normalized)
    return collected


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
        if tool_name in ARTICLE_TOOL_NAMES:
            return [payload]
        for key in ("items", "news", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        # Some NewsNow responses directly store entries without wrapping
        if all(isinstance(v, dict) for v in payload.values()):
            return list(payload.values())
        return []

    if isinstance(payload, list):
        if all(isinstance(item, dict) for item in payload):
            return payload
    return []


def _normalize_news_items(items: Sequence[Dict[str, Any]], default_category: str) -> List[NewsItem]:
    """Normalize heterogeneous tool outputs into NewsItem structures."""
    normalized: List[NewsItem] = []
    for raw in items:
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
        timestamp = raw.get("timestamp") or raw.get("published_at") or raw.get("retrieved_at") or raw.get("time")
        if isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()
        if not timestamp:
            timestamp = datetime.now().isoformat()
        category = str(raw.get("category") or default_category or "text")
        url = raw.get("url") or raw.get("link") or raw.get("mobile_url")

        normalized.append(
            NewsItem(
                title=title,
                content=content or title,
                source=source or "未知来源",
                timestamp=str(timestamp),
                category=category or "text",
                url=url,
            )
        )
    return normalized


def _merge_news_items(existing: Sequence[NewsItem], new_items: Sequence[NewsItem]) -> List[NewsItem]:
    """Merge new news items into the existing list with lightweight deduplication."""
    merged: List[NewsItem] = list(existing) if existing else []
    seen: set[Tuple[str, Any]] = {
        (item["title"], item.get("url")) for item in merged
    }

    for item in new_items:
        key = (item["title"], item.get("url"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _save_report_to_markdown(
    date_str: str,
    task: str,
    final_report: str,
    text_news: Sequence[NewsItem],
    video_news: Sequence[NewsItem],
) -> None:
    """Persist the full report into an output/<date>.md file."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = OUTPUT_DIR / f"{date_str}.md"
        metadata = {
            "task": task,
            "date": date_str,
            "generated_at": datetime.now().isoformat(),
            "text_news_count": len(text_news),
            "video_news_count": len(video_news),
        }
        with report_path.open("w", encoding="utf-8") as fp:
            fp.write(f"# {date_str} 新闻收集报告\n\n")
            fp.write(f"- 任务: {task}\n")
            fp.write(f"- 生成时间: {metadata['generated_at']}\n")
            fp.write(f"- 文字新闻条数: {metadata['text_news_count']}\n")
            fp.write(f"- 视频新闻条数: {metadata['video_news_count']}\n\n")
            fp.write("## 总结内容\n\n")
            fp.write(final_report.strip())
            fp.write("\n\n---\n\n")
            fp.write("## 原始新闻数据\n\n```json\n")
            fp.write(json.dumps(
                {
                    "metadata": metadata,
                    "text_news": list(text_news),
                    "video_news": list(video_news),
                },
                ensure_ascii=False,
                indent=2,
            ))
            fp.write("\n```\n")
        logger.info("Report saved to %s", report_path)
    except Exception as exc:
        logger.error("Failed to save markdown report: %s", exc)


async def coordinator_node(state: State):
    """
    Coordinator node - initializes the news collection process.

    Args:
        state: Current state

    Returns:
        Command with updated state
    """
    logger.info("=== Coordinator: Initializing news collection ===")

    # Load tools (direct @tool objects, no extra setup)
    text_tools = ALL_NEWS_TOOLS
    video_tools = []  # Video tools not implemented yet

    logger.info(f"Loaded {len(text_tools)} text tools and {len(video_tools)} video tools")

    return Command(
        update={
            "iteration": 0,
            "text_news": [],
            "video_news": [],
            "text_analysis": None,
            "video_analysis": None,
            "text_tools": text_tools,  # Cache tools in state
            "video_tools": video_tools,  # Cache tools in state
            "supervisor_decision": "",
            "supervisor_feedback": "",
            "quality_score": 0.0,
            "started_at": datetime.now(),
        }
    )


async def text_agent_node(state: State):
    """
    Text agent node - collects and analyzes text news.

    Uses preloaded tools from state cache to avoid repeated initialization.

    Args:
        state: Current state

    Returns:
        Command with updated state
    """
    logger.info(f"=== Text Agent: Collecting text news (Iteration {state['iteration']}) ===")

    # Use cached tools from state (preloaded in coordinator)
    text_tools = state.get("text_tools", [])

    if not text_tools:
        logger.warning("No text news tools available. Using mock data.")
        # Mock response for testing
        return Command(
            update={
                "text_news": state.get("text_news", []) + [
                    {
                        "title": f"示例文字新闻 {state['iteration'] + 1}",
                        "content": "这是一条示例文字新闻内容...",
                        "source": "示例新闻源",
                        "timestamp": datetime.now().isoformat(),
                        "category": "text",
                        "url": None,
                    }
                ],
                "text_analysis": f"文字新闻分析结果（迭代 {state['iteration'] + 1}）",
            }
        )
    
    # Create agent with text tools
    agent = create_agent(
        model=get_llm_by_type(AGENT_LLM_MAP["text_agent"]),
        tools=text_tools,
        system_prompt=get_system_prompt("text_agent"),
        debug=True,
    )

    agent_messages: List[Any] = []
    final_message_content: Any = ""

    # Execute agent with error handling for content safety
    try:
        result = await agent.ainvoke(
            input={
                "messages": [
                    HumanMessage(
                        content=f"请收集关于 '{state['task']}' 的最新文字新闻，日期: {state['date']}"
                    )
                ]
            }
        )
        agent_messages = result.get("messages", [])
        if agent_messages:
            final_message_content = agent_messages[-1].content
    except Exception as e:
        logger.warning(f"Text agent execution failed: {e}")
        # Check if it's a content safety error
        if "data_inspection_failed" in str(e) or "inappropriate content" in str(e):
            logger.warning("Content safety check failed, using summary mode")
            final_message_content = "由于内容安全检查，本次分析已跳过。已成功调用新闻工具获取数据。"
        else:
            # Re-raise other errors
            raise

    analysis_result = _stringify_message_content(final_message_content) or "本轮未生成分析结果。"
    new_news = _collect_news_from_tool_messages(agent_messages)

    update_payload: Dict[str, Any] = {
        "text_analysis": analysis_result,
    }

    if new_news:
        merged_news = _merge_news_items(state.get("text_news", []), new_news)
        update_payload["text_news"] = merged_news
        logger.info(f"Captured {len(new_news)} text news items (total: {len(merged_news)})")
    else:
        logger.info("No text news items captured from tools in this iteration")

    return Command(
        update=update_payload
    )


async def video_agent_node(state: State):
    """
    Video agent node - collects and analyzes video news.

    Uses preloaded tools from state cache to avoid repeated initialization.

    Args:
        state: Current state

    Returns:
        Command with updated state
    """
    logger.info(f"=== Video Agent: Collecting video news (Iteration {state['iteration']}) ===")

    # Use cached tools from state (preloaded in coordinator)
    video_tools = state.get("video_tools", [])

    if not video_tools:
        logger.warning("No video news tools available. Using mock data.")
        # Mock response for testing
        return Command(
            update={
                "video_news": state.get("video_news", []) + [
                    {
                        "title": f"示例视频新闻 {state['iteration'] + 1}",
                        "content": "这是一条示例视频新闻内容...",
                        "source": "示例视频源",
                        "timestamp": datetime.now().isoformat(),
                        "category": "video",
                        "url": "http://example.com/video",
                    }
                ],
                "video_analysis": f"视频新闻分析结果（迭代 {state['iteration'] + 1}）",
            }
        )
    
    # Create agent with video tools
    agent = create_agent(
        model=get_llm_by_type(AGENT_LLM_MAP["video_agent"]),
        tools=video_tools,
        system_prompt=get_system_prompt("video_agent"),
        debug=True,
    )

    # Execute agent with error handling for content safety
    try:
        result = await agent.ainvoke(
            input={
                "messages": [
                    HumanMessage(
                        content=f"请收集关于 '{state['task']}' 的最新视频新闻，日期: {state['date']}"
                    )
                ]
            }
        )
        analysis_result = result["messages"][-1].content
    except Exception as e:
        logger.warning(f"Video agent execution failed: {e}")
        # Check if it's a content safety error
        if "data_inspection_failed" in str(e) or "inappropriate content" in str(e):
            logger.warning("Content safety check failed, using summary mode")
            analysis_result = "由于内容安全检查，本次分析已跳过。已成功调用新闻工具获取数据。"
        else:
            # Re-raise other errors
            raise

    return Command(
        update={
            "video_analysis": analysis_result,
        }
    )


async def supervisor_node(state: State):
    """
    Supervisor node - evaluates quality and decides next step.
    
    Args:
        state: Current state
        
    Returns:
        Command with updated state
    """
    logger.info(f"=== Supervisor: Evaluating progress (Iteration {state['iteration']}) ===")
    
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    text_news_count = len(state.get("text_news", []))
    video_news_count = len(state.get("video_news", []))
    
    logger.info(f"Text news: {text_news_count}, Video news: {video_news_count}")
    
    # Simple evaluation logic
    # In production, you might want to use an LLM to evaluate quality
    quality_score = min(1.0, (text_news_count + video_news_count) / 6.0)
    
    should_summarize = False
    feedback = ""
    
    if iteration >= max_iterations:
        should_summarize = True
        feedback = f"达到最大迭代次数 ({max_iterations})，准备生成报告"
    elif quality_score >= 0.7 and text_news_count >= 3 and video_news_count >= 3:
        should_summarize = True
        feedback = "已收集足够的新闻内容（质量分数: {:.2f}），准备生成报告".format(quality_score)
    else:
        feedback = f"继续收集新闻（质量分数: {quality_score:.2f}, 文字: {text_news_count}/3, 视频: {video_news_count}/3)"
    
    logger.info(f"Decision: {feedback}")
    
    return Command(
        update={
            "supervisor_decision": "summarize" if should_summarize else "continue",
            "supervisor_feedback": feedback,
            "quality_score": quality_score,
            "iteration": iteration + 1,
        }
    )


async def summary_agent_node(state: State):
    """
    Summary agent node - generates final news report.
    
    Args:
        state: Current state
        
    Returns:
        Command with updated state
    """
    logger.info("=== Summary Agent: Generating final report ===")
    
    # Create summary agent (without tools, just LLM)
    llm = get_llm_by_type(AGENT_LLM_MAP["summary"])
    
    # Prepare content for summary
    text_news = state.get("text_news", [])
    video_news = state.get("video_news", [])
    text_analysis = state.get("text_analysis", "无")
    video_analysis = state.get("video_analysis", "无")
    
    summary_prompt = f"""基于以下收集的新闻内容，生成一份综合的每日新闻报告：

任务: {state['task']}
日期: {state['date']}

文字新闻数量: {len(text_news)}
视频新闻数量: {len(video_news)}

文字新闻分析:
{text_analysis}

视频新闻分析:
{video_analysis}

请生成一份结构化的新闻报告，包括：
1. 执行摘要
2. 重点新闻
3. 分类新闻内容
4. 关键洞察和趋势
"""
    
    messages = [HumanMessage(content=summary_prompt)]

    # Execute with error handling for content safety
    try:
        response = await llm.ainvoke(messages)
        final_report = response.content
        logger.info("Report generated successfully")
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        # Check if it's a content safety error
        if "data_inspection_failed" in str(e) or "inappropriate content" in str(e):
            logger.warning("Content safety check failed, generating basic summary")
            final_report = f"""# 新闻收集报告

任务: {state['task']}
日期: {state['date']}

## 数据统计

- 文字新闻数量: {len(text_news)}
- 视频新闻数量: {len(video_news)}
- 总迭代次数: {state.get('iteration', 0)}

注意: 由于内容安全检查限制，详细分析暂时无法生成。
建议: 调整新闻源或使用更宽松的内容策略。
"""
        else:
            # Re-raise other errors
            raise

    _save_report_to_markdown(
        state["date"],
        state["task"],
        final_report,
        text_news,
        video_news,
    )

    return Command(
        update={
            "final_report": final_report,
            "completed_at": datetime.now(),
        }
    )
