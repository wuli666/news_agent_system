"""
Reasoner Agent

Synthesizes retrieved information and performs causal inference.
Inspired by Archon's Strategist/Synthesizer pattern.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from .retriever import RetrievalResult, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result of reasoning/synthesis."""
    query: str
    answer: str
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


REASONING_SYSTEM_PROMPT = """你是专业的新闻知识图谱分析师，擅长从多源信息中进行深度分析和因果推理。

## 分析框架
1. **信息整合**：综合多条检索结果，提炼核心事实
2. **关联分析**：发现不同信息之间的联系和因果关系
3. **多角度解读**：从不同维度分析问题（时间线、影响、各方立场等）
4. **洞察提炼**：基于事实推导出有价值的见解

## 回答要求
- **问候类**（如"你好"）：简短友好回应，介绍自己能做什么
- **事实查询**：直接给出答案，附上关键细节
- **分析类问题**：
  1. 先总结核心事实
  2. 分析事件的来龙去脉、影响或意义
  3. 如有多方观点，客观呈现
  4. 最后给出你的分析洞察（用【洞察】标注）

## 注意事项
- 只基于检索到的信息回答，不编造
- 如果信息不足，明确说明并给出已知信息
- 如果检索信息与问题无关，直接说"知识库中暂无相关信息"
- 推测性结论要用"根据现有信息推测..."等表述
"""


class ReasonerAgent:
    """
    Reasoner Agent for synthesis and causal inference.

    Takes retrieval results and generates comprehensive answers
    with analytical insights.
    """

    def __init__(self, llm=None):
        """
        Initialize reasoner agent.

        Args:
            llm: LangChain LLM instance
        """
        self.llm = llm
        self._init_llm()

    def _init_llm(self):
        """Initialize LLM if not provided."""
        if self.llm is None:
            from src.llms.llm import get_llm_by_type
            self.llm = get_llm_by_type("qwen")

    def _format_context(self, retrieval: RetrievalResult) -> str:
        """Format retrieval results as context for reasoning."""
        context_parts = []

        # Add search results
        for i, result in enumerate(retrieval.results[:15], 1):
            content = result.content.strip()
            if content:
                source_info = f"[来源{i}]"
                if result.source:
                    source_info = f"[{result.source}]"
                context_parts.append(f"{source_info} {content}")

        # Add insights if available
        if retrieval.insights:
            if retrieval.insights.facts:
                context_parts.append("\n相关事实：")
                for fact in retrieval.insights.facts[:10]:
                    context_parts.append(f"- {fact}")

        return "\n\n".join(context_parts)

    async def reason(
        self,
        query: str,
        retrieval: RetrievalResult,
        additional_context: Optional[str] = None
    ) -> ReasoningResult:
        """
        Synthesize answer from retrieval results.

        Args:
            query: User query
            retrieval: Retrieval results to reason over
            additional_context: Optional extra context

        Returns:
            ReasoningResult
        """
        if self.llm is None:
            logger.error("LLM not initialized")
            return ReasoningResult(
                query=query,
                answer="无法生成回答：LLM未初始化",
                confidence=0.0
            )

        # Check if we have any results
        if not retrieval.results:
            return ReasoningResult(
                query=query,
                answer="未找到相关信息，无法回答该问题。",
                confidence=0.0
            )

        try:
            # Build context
            context = self._format_context(retrieval)

            if additional_context:
                context = f"{additional_context}\n\n{context}"

            # Build prompt
            user_prompt = f"""## 用户问题
{query}

## 检索到的相关信息
{context}

请基于以上信息回答用户的问题。"""

            # Call LLM
            messages = [
                SystemMessage(content=REASONING_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm.ainvoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)

            # Extract insights (marked with 【洞察】)
            insights = []
            if "【洞察】" in answer:
                parts = answer.split("【洞察】")
                if len(parts) > 1:
                    for part in parts[1:]:
                        insight = part.strip().split("\n")[0]
                        if insight:
                            insights.append(insight)

            # Collect sources
            sources = [
                r.content[:100] for r in retrieval.results[:5]
                if r.content
            ]

            # Estimate confidence based on result count and scores
            avg_score = sum(r.score for r in retrieval.results) / len(retrieval.results) if retrieval.results else 0
            confidence = min(0.9, avg_score + 0.1 * min(len(retrieval.results), 10) / 10)

            return ReasoningResult(
                query=query,
                answer=answer,
                confidence=confidence,
                sources=sources,
                insights=insights,
                metadata={
                    "retrieval_mode": retrieval.mode,
                    "result_count": len(retrieval.results),
                }
            )

        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return ReasoningResult(
                query=query,
                answer=f"生成回答时出错：{str(e)}",
                confidence=0.0
            )

    async def answer_question(
        self,
        query: str,
        retrieval_results: List[SearchResult],
        mode: str = "quick"
    ) -> ReasoningResult:
        """
        Answer a question given search results.

        Convenience method that wraps results in RetrievalResult.

        Args:
            query: User question
            retrieval_results: List of SearchResult
            mode: Retrieval mode used

        Returns:
            ReasoningResult
        """
        retrieval = RetrievalResult(
            query=query,
            results=retrieval_results,
            mode=mode
        )
        return await self.reason(query, retrieval)


async def reason_answer(
    query: str,
    retrieval: RetrievalResult,
    llm=None
) -> ReasoningResult:
    """
    Convenience function to reason over retrieval results.

    Args:
        query: User query
        retrieval: Retrieval results
        llm: Optional LLM instance

    Returns:
        ReasoningResult
    """
    agent = ReasonerAgent(llm=llm)
    return await agent.reason(query, retrieval)
