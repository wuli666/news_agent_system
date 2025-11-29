"""
安全相关工具函数

包含：
- 敏感内容屏蔽
"""
import re
from typing import List

# 敏感关键词模式
SENSITIVE_PATTERNS: List[str] = [
    r"自杀", r"枪击", r"爆炸", r"袭击", r"恐怖", r"极端", r"性侵", r"色情",
    r"赌博", r"毒品", r"暴力", r"谋杀", r"分裂", r"煽动", r"恐吓", r"ISIS",
]


def mask_sensitive_text(text: str) -> str:
    """
    屏蔽敏感内容，避免触发 DashScope 内容审核。

    Args:
        text: 原始文本

    Returns:
        屏蔽后的文本

    Examples:
        >>> mask_sensitive_text("新闻包含暴力内容")
        '新闻包含[敏感内容]内容'
    """
    masked = text or ""
    for pattern in SENSITIVE_PATTERNS:
        masked = re.sub(pattern, "[敏感内容]", masked, flags=re.IGNORECASE)
    # 清理多余空白
    return re.sub(r"\s+", " ", masked).strip()


def add_sensitive_pattern(pattern: str) -> None:
    """
    添加自定义敏感关键词模式。

    Args:
        pattern: 正则表达式模式
    """
    if pattern not in SENSITIVE_PATTERNS:
        SENSITIVE_PATTERNS.append(pattern)


def get_sensitive_patterns() -> List[str]:
    """
    获取当前的敏感关键词模式列表。

    Returns:
        敏感模式列表
    """
    return SENSITIVE_PATTERNS.copy()
