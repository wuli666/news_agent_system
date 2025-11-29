"""
Graph state types for the news multi-agent system.
"""
import operator
from typing import TypedDict, List, Optional, Any, Dict, Annotated
from datetime import datetime


class NewsItem(TypedDict):
    """Individual news item structure."""
    title: str
    content: str
    source: str
    timestamp: str
    category: str  # "text" or "video"
    url: Optional[str]


# ===== Team-Specific States (for Subgraphs) =====

class CollectionTeamState(TypedDict):
    """State for Collection Team Subgraph - supports parallel execution."""
    # Input from main workflow
    input_batch: List[NewsItem]
    task: str
    date: str

    # Internal processing
    text_batch: List[NewsItem]
    video_batch: List[NewsItem]
    assigned_news: List[NewsItem]  # Current agent's assigned news

    # Output to main workflow - using Annotated for concurrent updates
    text_news: Annotated[List[NewsItem], operator.add]
    video_news: Annotated[List[NewsItem], operator.add]
    text_analysis: Optional[str]
    video_analysis: Optional[str]


class AnalysisTeamState(TypedDict):
    """State for Analysis Team Subgraph with intelligent loop control - supports parallel execution."""
    # Input from main workflow
    text_news: List[NewsItem]
    date: str
    iteration: int

    # Internal loop control
    cycle_count: int
    information_gaps: List[Dict[str, Any]]
    skip_sentiment_relationship: bool
    analysis_gap_pending: bool
    gap_fill_mode: bool

    # Internal analysis results - using Annotated for concurrent updates
    research_notes: Annotated[List[Dict[str, Any]], operator.add]
    news_images: Annotated[Dict[str, List[str]], operator.or_]

    # Output to main workflow
    text_sentiment: Optional[str]
    text_relationship_graph: Optional[str]
    text_reflection: Optional[str]


class SummaryTeamState(TypedDict):
    """State for Summary Team Subgraph."""
    # Input from main workflow
    task: str
    date: str
    text_news: List[NewsItem]
    video_news: List[NewsItem]
    research_notes: List[Dict[str, Any]]
    text_sentiment: Optional[str]
    text_relationship_graph: Optional[str]
    trending_keywords: Optional[List[Dict[str, Any]]]
    news_images: Dict[str, List[str]]
    daily_papers: List[Dict[str, Any]]

    # Internal results
    timeline_analysis: Optional[str]
    trend_analysis: Optional[str]
    timeline_chart_path: Optional[str]

    # Output to main workflow
    final_report: Optional[str]


class State(TypedDict):
    """
    Simplified state for the news multi-agent system.
    This state is passed between all agents in the graph.
    """
    # ===== Input Parameters =====
    task: str  # Task description, e.g., "获取今日科技新闻"
    date: str  # Target date for news collection

    # ===== Iteration Control =====
    iteration: int  # Current iteration count
    max_iterations: int  # Maximum allowed iterations

    # ===== News Data Pool =====
    news_pool: List[NewsItem]  # Full deduped news pool from collector
    news_pool_cursor: int  # Offset for batch distribution
    latest_news_batch: List[NewsItem]  # Current batch for agents to process

    # ===== Team Working Data (Temporary) =====
    assigned_news: List[NewsItem]  # News items assigned to current agent
    text_batch: List[NewsItem]  # Text news batch for collection team
    video_batch: List[NewsItem]  # Video news batch for collection team

    # ===== Collected News by Category =====
    text_news: List[NewsItem]  # Text news collected by text collector
    video_news: List[NewsItem]  # Video news collected by video collector

    # ===== Analysis Results =====
    # Collection team results
    text_analysis: Optional[str]  # Text collector's analysis
    video_analysis: Optional[str]  # Video collector's analysis

    # Analysis team results
    research_notes: List[Dict[str, Any]]  # Research agent's background info
    text_relationship_graph: Optional[str]  # Relationship/causal graph
    text_sentiment: Optional[str]  # Sentiment and risk analysis
    text_reflection: Optional[str]  # Reflection on supervisor questions

    news_images: Dict[str, List[str]]  # Mapping title -> related image URLs

    # Daily research papers (from Airiv or similar service)
    daily_papers: List[Dict[str, Any]]
    gap_fill_mode: Optional[bool]  # 内层补洞时是否跳过 sentiment/relationship

    # Summary team results
    timeline_analysis: Optional[str]  # Timeline extraction
    trend_analysis: Optional[str]  # Trend and cluster analysis
    timeline_chart_path: Optional[str]  # Generated chart path
    final_report: Optional[str]  # Final news report

    # ===== Cross-Agent Signals =====
    trending_keywords: Optional[List[Dict[str, Any]]]  # Keywords with counts
    optimized_query: Optional[str]  # Optimized search query
    supervisor_questions: List[str]  # Questions for reflection agent

    # Information gaps detection (for Extract循环)
    information_gaps: List[Dict[str, Any]]  # Reflect发现的信息缺口列表
    text_team_cycle_count: int  # Text team内层循环计数

    # ===== Cached Tools =====
    text_tools: Optional[List[Any]]  # Preloaded text news tools
    video_tools: Optional[List[Any]]  # Preloaded video news tools
    research_tools: Optional[List[Any]]  # Preloaded research tools

    # ===== Control and Decision =====
    supervisor_decision: str  # "collect", "analyze", or "summarize"
    supervisor_feedback: str  # Feedback message
    quality_score: float  # Overall quality score (0-1)
    last_agent: str  # Last executed agent name
    analysis_gap_pending: Optional[bool]  # 是否还有信息缺口未填
    gap_fill_mode: bool  # 是否处于gap填充模式
    skip_sentiment_relationship: bool  # 内层循环优化：是否跳过sentiment/relationship

    # ===== Metadata =====
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


# ===== Pydantic Schemas for Structured Report Output =====

from pydantic import BaseModel, Field


class InformationGap(BaseModel):
    """信息缺口检测结果（用于Reflect Agent的结构化输出）"""
    news_title: str = Field(description="缺少信息的新闻标题")
    missing_type: str = Field(description="缺失信息类型：时间信息/背景信息/人物信息/事件细节/数据证据")
    specific_question: str = Field(description="具体缺少什么信息，如：缺少事件发生的准确时间")
    priority: int = Field(description="优先级，1-5，5最高", ge=1, le=5)
    needs_extract: bool = Field(description="是否需要使用Tavily Extract深度提取", default=True)


class ReflectionResult(BaseModel):
    """Reflect Agent的完整输出结构"""
    overall_assessment: str = Field(description="总体评估，信息完整度如何")
    information_gaps: List[InformationGap] = Field(description="发现的信息缺口列表", default_factory=list)
    needs_continue: bool = Field(description="是否需要继续内层循环补充信息")
    suggestion: str = Field(description="下一步建议")


class SentimentDistribution(BaseModel):
    """情感分布"""
    positive: int = Field(description="正面情感百分比", ge=0, le=100)
    neutral: int = Field(description="中性情感百分比", ge=0, le=100)
    negative: int = Field(description="负面情感百分比", ge=0, le=100)


class Overview(BaseModel):
    """新闻总体概览"""
    total_news: int = Field(description="新闻总数")
    text_news: int = Field(description="文字新闻数量")
    video_news: int = Field(description="视频新闻数量")
    sentiment_distribution: SentimentDistribution = Field(description="情感分布")
    top_keywords: List[str] = Field(description="热门关键词列表，5-10个")
    summary: str = Field(description="一句话概括今日新闻重点，30字以内")


class CoreTheme(BaseModel):
    """核心主题分类"""
    theme: str = Field(description="主题名称，如：国家安全、外交关系等")
    heat_level: int = Field(description="热度等级，1-5星", ge=1, le=5)
    examples: List[str] = Field(description="典型新闻标题示例，1-3条")
    attention: str = Field(description="关注度等级：极高/高/中高/中等/低")


class KeyEvent(BaseModel):
    """重点事件解析"""
    title: str = Field(description="事件标题")
    time: str = Field(description="事件时间")
    summary: str = Field(description="事件摘要，100字左右")
    impact: str = Field(description="影响评估")
    risk_level: str = Field(description="风险等级：高/中/低")
    category: str = Field(description="事件分类：国家安全/外交关系/社会热点/经济动态/文化娱乐")


class SentimentRiskItem(BaseModel):
    """情感与风险分析项"""
    title: str = Field(description="新闻标题")
    sentiment: str = Field(description="情感倾向：正面/中性/负面")
    risk: str = Field(description="关注点/潜在风险描述（简洁）")
    severity: str = Field(description="关注度：高/中/低")


class TrendPrediction(BaseModel):
    """趋势预测"""
    dimension: str = Field(description="趋势维度，如：中日关系、国内安全等")
    short_term: str = Field(description="短期预测（1周内）")
    long_term: str = Field(description="中长期展望（1个月内）")


class StockAnalysis(BaseModel):
    """股票涨跌分析"""
    stock_name: str = Field(description="股票名称，如：茅台、腾讯")
    stock_code: Optional[str] = Field(None, description="股票代码，如有")
    change_percent: Optional[str] = Field(None, description="涨跌幅，如：+5.2%、-3.1%")
    trend: str = Field(description="趋势：上涨/下跌/横盘")
    heat_score: Optional[float] = Field(None, description="热度分数", ge=0, le=10)
    news_summary: str = Field(description="相关新闻摘要，50字以内")


class NewsReport(BaseModel):
    """完整的结构化新闻报告"""
    overview: Overview = Field(description="总体概览")
    core_themes: List[CoreTheme] = Field(description="核心主题分类，3-6个主题")
    key_events: List[KeyEvent] = Field(description="重点事件解析，5-8个事件")
    sentiment_risk: List[SentimentRiskItem] = Field(description="情感与风险分析，8-12个")
    trends: List[TrendPrediction] = Field(description="趋势预测，4-6个维度")
    stocks: List[StockAnalysis] = Field(default=[], description="股票涨跌分析（如有雪球数据），8-15只热门股票")
    executive_summary: str = Field(description="综合报告摘要，200-300字，包含今日焦点、主要动向、关注建议")
    timeline_analysis: Optional[str] = Field(
        None, description="时间线分析概要，复用时间线 agent 的结果"
    )
    trend_analysis: Optional[str] = Field(
        None, description="趋势聚类分析概要，复用趋势 agent 的结果"
    )
    relationship_graph: Optional[str] = Field(
        None, description="关系图谱文本版，复用关系图谱 agent 的结果"
    )
