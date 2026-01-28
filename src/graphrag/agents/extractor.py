"""
Extractor Agent

Extracts entities and relationships from news content using LLM.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""
    name: str
    type: str  # Person, Organization, Location, Event, Topic, Product
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ExtractedRelation:
    """Represents an extracted relationship."""
    source: str
    target: str
    relation_type: str  # RELATED_TO, WORKS_FOR, LOCATED_IN, etc.
    description: str = ""
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    """Result of entity/relation extraction."""
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    source_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


EXTRACTION_SYSTEM_PROMPT = """你是一个专业的新闻实体和关系抽取助手。你的任务是从新闻文本中提取重要的实体和它们之间的关系。

## 实体类型
- Person: 人物（政治家、企业家、名人等）
- Organization: 组织（公司、政府机构、NGO等）
- Location: 地点（国家、城市、地区）
- Event: 事件（会议、事故、公告等）
- Topic: 话题/主题（AI、经济、政治等）
- Product: 产品/技术（具体产品、服务、技术）

## 关系类型
- RELATED_TO: 一般关联关系
- WORKS_FOR: 人物为组织工作
- LOCATED_IN: 实体位于某地
- CAUSED_BY: 事件由另一事件或实体引起
- PRODUCES: 组织生产产品
- COMPETES_WITH: 组织间竞争关系
- PART_OF: 从属关系

## 输出格式
请以JSON格式输出，包含entities和relations两个数组：
{
    "entities": [
        {"name": "实体名称", "type": "实体类型", "attributes": {"key": "value"}}
    ],
    "relations": [
        {"source": "源实体名称", "target": "目标实体名称", "relation_type": "关系类型", "description": "关系描述"}
    ]
}

注意：
1. 只提取明确在文本中提到的实体和关系
2. 实体名称要规范化（如"阿里巴巴集团"统一为"阿里巴巴"）
3. 避免重复的实体和关系
4. 关系描述要简洁明了
"""


class ExtractorAgent:
    """
    Entity and Relationship Extractor Agent.

    Uses LLM to extract structured information from news text.
    """

    def __init__(self, llm=None):
        """
        Initialize extractor agent.

        Args:
            llm: LangChain LLM instance (will use default if not provided)
        """
        self.llm = llm
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM if not provided."""
        if self.llm is None:
            try:
                from src.llms.llm import get_llm_by_type
                self.llm = get_llm_by_type("qwen")
            except ImportError:
                logger.warning("Could not import LLM from src.llms, extraction will fail")

    async def extract(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ExtractionResult:
        """
        Extract entities and relations from text.

        Args:
            text: Input text to extract from
            metadata: Optional metadata about the source

        Returns:
            ExtractionResult with entities and relations
        """
        if not text or not text.strip():
            return ExtractionResult(
                entities=[],
                relations=[],
                source_text=text,
                metadata=metadata or {}
            )

        if self.llm is None:
            logger.error("LLM not initialized")
            return ExtractionResult(
                entities=[],
                relations=[],
                source_text=text,
                metadata=metadata or {}
            )

        try:
            # Build prompt
            user_prompt = f"""请从以下新闻文本中提取实体和关系：

{text}

请以JSON格式输出结果。"""

            # Call LLM
            messages = [
                SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse response
            return self._parse_response(response_text, text, metadata)

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(
                entities=[],
                relations=[],
                source_text=text,
                metadata=metadata or {}
            )

    def _parse_response(
        self,
        response: str,
        source_text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> ExtractionResult:
        """Parse LLM response into structured result."""
        entities = []
        relations = []

        try:
            # Try to extract JSON from response
            json_str = response

            # Handle markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            data = json.loads(json_str.strip())

            # Parse entities
            for e in data.get("entities", []):
                entities.append(ExtractedEntity(
                    name=e.get("name", ""),
                    type=e.get("type", "Topic"),
                    attributes=e.get("attributes", {}),
                    confidence=e.get("confidence", 1.0)
                ))

            # Parse relations
            for r in data.get("relations", []):
                relations.append(ExtractedRelation(
                    source=r.get("source", ""),
                    target=r.get("target", ""),
                    relation_type=r.get("relation_type", "RELATED_TO"),
                    description=r.get("description", ""),
                    confidence=r.get("confidence", 1.0)
                ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction response as JSON: {e}")
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")

        return ExtractionResult(
            entities=entities,
            relations=relations,
            source_text=source_text,
            metadata=metadata or {}
        )

    async def extract_from_news_items(
        self,
        news_items: List[Dict[str, Any]]
    ) -> List[ExtractionResult]:
        """
        Extract from multiple news items.

        Args:
            news_items: List of news item dictionaries

        Returns:
            List of ExtractionResult objects
        """
        results = []

        for item in news_items:
            # Build text from news item
            text_parts = []

            title = item.get("title", "")
            if title:
                text_parts.append(f"标题：{title}")

            content = item.get("content", item.get("description", ""))
            if content:
                text_parts.append(f"内容：{content}")

            source = item.get("source", item.get("platform", ""))
            if source:
                text_parts.append(f"来源：{source}")

            full_text = "\n".join(text_parts)

            metadata = {
                "title": title,
                "source": source,
                "url": item.get("url", ""),
                "hot": item.get("hot", ""),
            }

            result = await self.extract(full_text, metadata)
            results.append(result)

        return results


async def extract_entities(
    text: str,
    llm=None,
    metadata: Optional[Dict[str, Any]] = None
) -> ExtractionResult:
    """
    Convenience function to extract entities from text.

    Args:
        text: Input text
        llm: Optional LLM instance
        metadata: Optional metadata

    Returns:
        ExtractionResult
    """
    agent = ExtractorAgent(llm=llm)
    return await agent.extract(text, metadata)
