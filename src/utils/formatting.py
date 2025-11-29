"""
格式化相关工具函数

包含：
- 消息内容转字符串
- 工具负载转换
- 安全新闻列表构建
"""
import json
import logging
from typing import Any, Dict, List, Sequence

from src.utils.security import mask_sensitive_text

logger = logging.getLogger(__name__)


def stringify_message_content(content: Any) -> str:
    """
    将 LangChain 消息内容转换为纯字符串。

    Args:
        content: LangChain message content (可能是 str, list, dict, None)

    Returns:
        纯字符串内容

    Examples:
        >>> stringify_message_content("Hello")
        'Hello'
        >>> stringify_message_content([{"text": "Part 1"}, {"text": "Part 2"}])
        'Part 1\\nPart 2'
    """
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


def coerce_tool_payload(content: Any) -> Any:
    """
    将工具消息内容转换回 Python 对象。

    Args:
        content: 工具消息内容（可能是 JSON 字符串、list、dict）

    Returns:
        Python 对象（dict, list, 或 None）

    Examples:
        >>> coerce_tool_payload('{"key": "value"}')
        {'key': 'value'}
        >>> coerce_tool_payload([{"text": '{"a": 1}'}])
        {'a': 1}
    """
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


def build_safe_news_list(news_items: Sequence[Dict[str, Any]]) -> str:
    """
    渲染新闻标题列表，带敏感内容屏蔽。

    Args:
        news_items: 新闻项列表

    Returns:
        格式化的新闻列表字符串

    Examples:
        >>> items = [{"title": "新闻1"}, {"title": "新闻2"}]
        >>> build_safe_news_list(items)
        '1. 新闻1\\n2. 新闻2'
    """
    safe_lines = []
    for idx, item in enumerate(news_items):
        title = mask_sensitive_text(item.get("title", ""))[:200]
        safe_lines.append(f"{idx + 1}. {title}")
    return "\n".join(safe_lines)
