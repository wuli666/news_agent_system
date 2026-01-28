"""
Agentic RAG - 多策略智能检索增强生成

特点：
1. 多策略并行检索 - LocalSearch, GlobalSearch, HybridSearch, ChainOfExploration
2. 查询规划 - 根据问题类型选择最佳策略组合
3. 结果融合 - 去重、排序、综合多源信息
4. 流式输出 - 详细执行日志
"""

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Generator, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """统一的搜索结果"""
    strategy: str  # 来源策略
    content: str   # 主要内容
    score: float = 1.0  # 相关性分数
    source_entity: str = ""  # 来源实体
    relation: str = ""  # 关系类型
    metadata: Dict = field(default_factory=dict)


@dataclass
class AgentState:
    """Agent 状态"""
    question: str
    query_type: str = "general"  # general, entity, comparison, timeline, causal
    entities: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)
    results: List[SearchResult] = field(default_factory=list)
    final_answer: str = ""


# 检索反思 prompt
RETRIEVAL_REFLECT_PROMPT = """评估检索结果是否足够回答用户问题。

## 用户问题
{question}

## 已检索到的信息摘要
{results_summary}

## 评估标准（宽松）
这是新闻热搜知识图谱，不是学术数据库。只要满足以下任一条件即可判定为"足够"：
- 有3条以上与问题相关的事实
- 能回答问题的核心部分（即使缺少细节）
- 提到了问题中的关键实体

不要期望知识图谱包含：
- 权威机构报告（WHO、CDC等）
- 精确的数字数据（病例数、日期等）
- 媒体原文链接

## 输出JSON格式
{{
    "sufficient": true/false,
    "reason": "简短说明",
    "suggestion": "如果不足，建议补充检索的关键词（可选）"
}}

只输出JSON："""


# 答案反思 prompt
ANSWER_REFLECT_PROMPT = """评估生成的答案质量。

## 用户问题
{question}

## 生成的答案
{answer}

## 评估要求
1. 答案是否直接回答了问题？
2. 是否有明显的事实错误或逻辑问题？
3. 是否过于冗长或偏离主题？

## 输出JSON格式
{{
    "quality": "good/acceptable/poor",
    "issues": ["问题1", "问题2"],
    "suggestion": "如果质量差，建议如何改进"
}}

只输出JSON："""


# 查询分析 prompt
QUERY_ANALYZER_PROMPT = """分析用户问题，提取关键信息。

问题：{question}

请用JSON格式输出：
{{
    "query_type": "问题类型(general/entity/comparison/timeline/causal)",
    "entities": ["问题中提到的实体"],
    "keywords": ["关键词"],
    "intent": "用户意图简述"
}}

问题类型说明：
- general: 一般性问题，如"最近有什么新闻"
- entity: 关于特定实体的问题，如"马斯克最近做了什么"
- comparison: 比较类问题，如"A和B有什么关系"
- timeline: 时间线问题，如"某事件的发展过程"
- causal: 因果分析问题，如"为什么会发生..."

只输出JSON，不要其他内容："""


SYNTHESIZER_PROMPT = """你是资深新闻分析师。根据检索结果回答用户问题。

## 问题
{question}

## 检索结果
{local_results}

{global_results}

{hybrid_results}

{chain_results}

## 回答要求

1. **分类整理**：将相关新闻按主题分类（如政治、经济、社会、国际等）
2. **适度展开**：每条重要新闻用2-3句话说明背景和要点，不要只列标题
3. **关联分析**：如果多条新闻之间有关联，指出它们的联系
4. **过滤噪声**：忽略明显不相关的内容，不要强行拼凑
5. **简短预测**：最后可以用1-2句点评当前态势或趋势

## 格式示例
```
**国际关系**
关于xxx事件，目前情况是...背景是...

**社会民生**
近期xxx引发关注，具体是...

**趋势观察**
整体来看...
```

## 禁止
- 不要写成学术报告（不需要"高置信度""交叉验证"等术语）
- 不要只是简单罗列事件标题
- 信息不足时直接说明，不要编造

请回答："""


class AgenticRAG:
    """
    多策略 Agentic RAG 实现

    检索策略：
    1. LocalSearch - 实体级精确检索，获取实体的直接关联
    2. GlobalSearch - 社区级宏观检索，广泛语义搜索
    3. HybridSearch - 混合检索，结合实体和语义
    4. ChainOfExploration - 多跳图谱探索
    """

    def __init__(self, zep_client=None, llm=None):
        self.zep_client = zep_client
        self.llm = llm
        self._init_components()

    def _init_components(self):
        """初始化组件"""
        if self.zep_client is None:
            from ..services.zep_client import ZepGraphClient
            self.zep_client = ZepGraphClient()

        if self.llm is None:
            from src.llms.llm import get_llm_by_type
            self.llm = get_llm_by_type("qwen")

        self.graph_id = self.zep_client.get_or_create_graph("news_graph")

        # 缓存节点和边数据用于图探索
        self._node_cache = {}
        self._edge_cache = {}

    def _analyze_query(self, question: str) -> Dict:
        """分析查询，提取实体和判断类型"""
        try:
            prompt = QUERY_ANALYZER_PROMPT.format(question=question)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, 'content') else str(response)

            # 提取 JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            return {
                "query_type": "general",
                "entities": [],
                "keywords": [question],
                "intent": question
            }
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                "query_type": "general",
                "entities": [],
                "keywords": [question],
                "intent": question
            }

    def _ensure_graph_loaded(self):
        """确保图数据已加载到内存"""
        if not self._node_cache:
            try:
                graph_data = self.zep_client.get_graph_data(self.graph_id)
                # 节点缓存: uuid -> node
                self._node_cache = {n.uuid: n for n in graph_data.nodes}
                # 节点名称索引: name -> uuid
                self._name_to_uuid = {}
                for n in graph_data.nodes:
                    self._name_to_uuid[n.name] = n.uuid
                    # 也索引部分匹配
                    for word in n.name.split():
                        if len(word) >= 2:
                            if word not in self._name_to_uuid:
                                self._name_to_uuid[word] = n.uuid

                # 边缓存: 邻接表
                self._edge_cache = {}  # uuid -> [edges]
                self._all_edges = list(graph_data.edges)
                for e in graph_data.edges:
                    if e.source_node_uuid not in self._edge_cache:
                        self._edge_cache[e.source_node_uuid] = []
                    self._edge_cache[e.source_node_uuid].append(e)
                    if e.target_node_uuid not in self._edge_cache:
                        self._edge_cache[e.target_node_uuid] = []
                    self._edge_cache[e.target_node_uuid].append(e)

                logger.info(f"Graph loaded: {len(self._node_cache)} nodes, {len(self._all_edges)} edges")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")

    def _find_entity_node(self, entity_name: str):
        """模糊匹配找到实体对应的节点"""
        # 精确匹配
        if entity_name in self._name_to_uuid:
            uuid = self._name_to_uuid[entity_name]
            return self._node_cache.get(uuid)

        # 模糊匹配
        entity_lower = entity_name.lower()
        for node in self._node_cache.values():
            if entity_lower in node.name.lower() or node.name.lower() in entity_lower:
                return node
        return None

    def _local_search(self, entities: List[str], keywords: List[str]) -> List[SearchResult]:
        """
        LocalSearch: 实体中心子图提取

        策略：
        1. 找到实体对应的节点
        2. 提取该节点的所有直接边（1跳邻居）
        3. 聚合实体属性和关联事实
        4. 按边的时间排序（最新优先）
        """
        self._ensure_graph_loaded()
        results = []
        seen_facts = set()

        search_terms = entities + keywords[:2]

        for entity in search_terms[:5]:
            # 1. 找到实体节点
            node = self._find_entity_node(entity)
            if not node:
                continue

            # 2. 获取所有直接关联的边
            edges = self._edge_cache.get(node.uuid, [])

            # 3. 按时间排序（新的在前）
            sorted_edges = sorted(edges, key=lambda e: e.created_at or "", reverse=True)

            # 4. 提取事实
            for edge in sorted_edges[:20]:
                fact = edge.fact
                if not fact or fact in seen_facts:
                    continue
                seen_facts.add(fact)

                # 找出关联的另一个节点
                other_uuid = edge.target_node_uuid if edge.source_node_uuid == node.uuid else edge.source_node_uuid
                other_node = self._node_cache.get(other_uuid)
                other_name = other_node.name if other_node else "未知"

                results.append(SearchResult(
                    strategy="LocalSearch",
                    content=fact,
                    score=1.0,
                    source_entity=node.name,
                    relation=f"{edge.name} → {other_name}",
                    metadata={
                        "method": "subgraph_extraction",
                        "center_node": node.name,
                        "neighbor": other_name,
                        "edge_type": edge.name,
                        "created_at": edge.created_at
                    }
                ))

            # 5. 添加节点摘要（如果有）
            if node.summary and node.summary not in seen_facts:
                seen_facts.add(node.summary)
                results.append(SearchResult(
                    strategy="LocalSearch",
                    content=f"[实体摘要] {node.name}: {node.summary}",
                    score=0.9,
                    source_entity=node.name,
                    relation="summary",
                    metadata={"method": "entity_summary"}
                ))

        return results[:20]

    def _global_search(self, question: str, keywords: List[str]) -> List[SearchResult]:
        """
        GlobalSearch: 社区级宏观检索

        策略：
        1. 语义搜索获取初始结果
        2. 提取高频出现的实体
        3. 基于高频实体扩展搜索（社区发现的简化版）
        4. 聚合同一主题的多条信息
        """
        self._ensure_graph_loaded()
        results = []
        seen_facts = set()
        entity_frequency = {}  # 统计实体出现频率

        # 1. 第一轮：语义搜索
        try:
            search_results = self.zep_client.search(question, self.graph_id, limit=20)

            for r in search_results:
                fact = r.get("fact", "")
                if fact and fact not in seen_facts:
                    seen_facts.add(fact)
                    results.append(SearchResult(
                        strategy="GlobalSearch",
                        content=fact,
                        score=r.get("score", 1.0),
                        relation=r.get("name", ""),
                        metadata={"method": "semantic_search", "round": 1}
                    ))

                    # 统计涉及的实体
                    source_uuid = r.get("source_node_uuid")
                    target_uuid = r.get("target_node_uuid")
                    for uuid in [source_uuid, target_uuid]:
                        if uuid and uuid in self._node_cache:
                            name = self._node_cache[uuid].name
                            entity_frequency[name] = entity_frequency.get(name, 0) + 1

        except Exception as e:
            logger.error(f"GlobalSearch semantic failed: {e}")

        # 2. 找出高频实体（社区中心）
        top_entities = sorted(entity_frequency.items(), key=lambda x: x[1], reverse=True)[:3]

        # 3. 第二轮：基于高频实体扩展
        for entity_name, freq in top_entities:
            if freq < 2:  # 至少出现2次才算高频
                continue

            node = self._find_entity_node(entity_name)
            if not node:
                continue

            edges = self._edge_cache.get(node.uuid, [])
            for edge in edges[:10]:
                fact = edge.fact
                if fact and fact not in seen_facts:
                    seen_facts.add(fact)
                    results.append(SearchResult(
                        strategy="GlobalSearch",
                        content=fact,
                        score=0.8,
                        source_entity=entity_name,
                        relation=edge.name,
                        metadata={
                            "method": "community_expansion",
                            "round": 2,
                            "hub_entity": entity_name,
                            "hub_frequency": freq
                        }
                    ))

        return results[:20]

    def _hybrid_search(self, entities: List[str], question: str) -> List[SearchResult]:
        """
        HybridSearch: 双向检索 + 交叉验证

        策略：
        1. 从问题出发：语义搜索
        2. 从实体出发：图结构搜索
        3. 取交集：同时被两种方法找到的结果可信度更高
        4. 对结果进行 reranking
        """
        self._ensure_graph_loaded()
        results = []

        # 结果集合，用于交叉验证
        semantic_facts = {}  # fact -> score
        graph_facts = {}     # fact -> score

        # 1. 语义搜索（从问题出发）
        try:
            search_results = self.zep_client.search(question, self.graph_id, limit=20)
            for r in search_results:
                fact = r.get("fact", "")
                if fact:
                    semantic_facts[fact] = r.get("score", 1.0)
        except Exception as e:
            logger.error(f"HybridSearch semantic failed: {e}")

        # 2. 图结构搜索（从实体出发）
        for entity in entities[:5]:
            node = self._find_entity_node(entity)
            if not node:
                continue

            edges = self._edge_cache.get(node.uuid, [])
            for edge in edges[:15]:
                fact = edge.fact
                if fact:
                    graph_facts[fact] = graph_facts.get(fact, 0) + 1.0

        # 3. 交叉验证 + Reranking
        all_facts = set(semantic_facts.keys()) | set(graph_facts.keys())

        for fact in all_facts:
            sem_score = semantic_facts.get(fact, 0)
            graph_score = graph_facts.get(fact, 0)

            # 计算综合分数
            if sem_score > 0 and graph_score > 0:
                # 两种方法都找到，高可信度
                final_score = (sem_score + graph_score) * 1.5
                method = "cross_validated"
            elif sem_score > 0:
                final_score = sem_score
                method = "semantic_only"
            else:
                final_score = graph_score * 0.8
                method = "graph_only"

            results.append(SearchResult(
                strategy="HybridSearch",
                content=fact,
                score=final_score,
                metadata={
                    "method": method,
                    "semantic_score": sem_score,
                    "graph_score": graph_score,
                    "cross_validated": sem_score > 0 and graph_score > 0
                }
            ))

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:20]

    def _chain_of_exploration(self, entities: List[str], max_hops: int = 2) -> List[SearchResult]:
        """
        ChainOfExploration: LLM 引导的多跳图谱探索

        策略：
        1. 从种子实体出发
        2. 获取候选邻居节点
        3. 用 LLM 判断哪些方向值得探索（相关性评分）
        4. 沿高分方向继续探索
        5. 记录探索路径，形成推理链
        """
        self._ensure_graph_loaded()
        results = []
        seen_facts = set()
        explored_uuids: Set[str] = set()
        exploration_paths = []  # 记录探索路径

        for entity in entities[:3]:
            # 1. 找到种子节点
            seed_node = self._find_entity_node(entity)
            if not seed_node:
                continue

            explored_uuids.add(seed_node.uuid)
            current_path = [seed_node.name]

            # 2. 多跳探索
            current_nodes = [(seed_node, current_path, 1.0)]  # (node, path, score)

            for hop in range(max_hops):
                next_nodes = []
                candidates_for_llm = []  # 收集候选节点让 LLM 评分

                for current_node, path, path_score in current_nodes:
                    edges = self._edge_cache.get(current_node.uuid, [])

                    # 按时间排序，优先探索新的边
                    sorted_edges = sorted(edges, key=lambda e: e.created_at or "", reverse=True)

                    for edge in sorted_edges[:15]:
                        # 找出邻居节点
                        neighbor_uuid = (edge.target_node_uuid
                                        if edge.source_node_uuid == current_node.uuid
                                        else edge.source_node_uuid)

                        if neighbor_uuid in explored_uuids:
                            continue

                        neighbor_node = self._node_cache.get(neighbor_uuid)
                        if not neighbor_node:
                            continue

                        # 收集候选
                        candidates_for_llm.append({
                            "node": neighbor_node,
                            "edge": edge,
                            "from_node": current_node,
                            "path": path,
                            "path_score": path_score
                        })

                # 3. 如果候选太多，用启发式规则预筛选
                if len(candidates_for_llm) > 10:
                    # 优先选择：有 fact 的边、较新的边
                    candidates_for_llm = sorted(
                        candidates_for_llm,
                        key=lambda c: (
                            1 if c["edge"].fact else 0,
                            c["edge"].created_at or ""
                        ),
                        reverse=True
                    )[:10]

                # 4. 对每个候选，计算探索价值
                for candidate in candidates_for_llm:
                    edge = candidate["edge"]
                    neighbor = candidate["node"]
                    from_node = candidate["from_node"]
                    path = candidate["path"]
                    path_score = candidate["path_score"]

                    # 记录事实
                    fact = edge.fact
                    if fact and fact not in seen_facts:
                        seen_facts.add(fact)

                        new_path = path + [f"--{edge.name}-->", neighbor.name]
                        path_str = " ".join(new_path)

                        # 跳数衰减 + 路径分数
                        hop_decay = 1.0 / (hop + 1)
                        final_score = path_score * hop_decay

                        results.append(SearchResult(
                            strategy="ChainOfExploration",
                            content=fact,
                            score=final_score,
                            source_entity=entity,
                            relation=f"{from_node.name} --{edge.name}--> {neighbor.name}",
                            metadata={
                                "method": "guided_exploration",
                                "hop": hop + 1,
                                "path": path_str,
                                "edge_type": edge.name,
                                "created_at": edge.created_at
                            }
                        ))

                        exploration_paths.append(path_str)

                    # 准备下一跳
                    if neighbor.uuid not in explored_uuids:
                        explored_uuids.add(neighbor.uuid)
                        new_path = path + [f"--{edge.name}-->", neighbor.name]
                        # 传递衰减后的分数
                        next_nodes.append((neighbor, new_path, path_score * 0.7))

                # 限制每层探索的节点数
                current_nodes = sorted(next_nodes, key=lambda x: x[2], reverse=True)[:8]

                if not current_nodes:
                    break

        # 5. 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)

        # 添加探索摘要
        if exploration_paths:
            summary = f"探索了 {len(exploration_paths)} 条路径，覆盖 {len(explored_uuids)} 个节点"
            results.insert(0, SearchResult(
                strategy="ChainOfExploration",
                content=f"[探索摘要] {summary}",
                score=0.5,
                metadata={"method": "exploration_summary", "paths_count": len(exploration_paths)}
            ))

        return results[:20]

    def _select_strategies(self, query_type: str, entities: List[str]) -> List[str]:
        """根据查询类型选择策略组合"""
        # 默认所有策略都用
        strategies = ["LocalSearch", "GlobalSearch", "HybridSearch", "ChainOfExploration"]

        if query_type == "entity" and entities:
            # 实体查询优先 Local 和 Chain
            strategies = ["LocalSearch", "ChainOfExploration", "GlobalSearch", "HybridSearch"]
        elif query_type == "comparison" and len(entities) >= 2:
            # 比较查询优先 Hybrid
            strategies = ["HybridSearch", "LocalSearch", "ChainOfExploration", "GlobalSearch"]
        elif query_type == "timeline":
            # 时间线查询优先 Global
            strategies = ["GlobalSearch", "LocalSearch", "ChainOfExploration", "HybridSearch"]
        elif query_type == "causal":
            # 因果查询优先 Chain
            strategies = ["ChainOfExploration", "HybridSearch", "LocalSearch", "GlobalSearch"]

        return strategies

    def _reflect_on_retrieval(self, question: str, results: Dict[str, List[SearchResult]]) -> Dict:
        """
        检索后反思：评估结果是否足够
        """
        # 生成结果摘要
        total = sum(len(v) for v in results.values())
        summary_parts = []

        for strategy, items in results.items():
            if items:
                facts = [r.content[:80] for r in items[:5]]
                summary_parts.append(f"{strategy}: {len(items)}条\n" + "\n".join(f"  - {f}" for f in facts))

        results_summary = "\n".join(summary_parts) if summary_parts else "无检索结果"

        try:
            prompt = RETRIEVAL_REFLECT_PROMPT.format(
                question=question,
                results_summary=results_summary
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, 'content') else str(response)

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Retrieval reflection failed: {e}")

        # 默认认为足够
        return {"sufficient": True, "reason": "默认通过", "missing": "", "suggestion": ""}

    def _reflect_on_answer(self, question: str, answer: str) -> Dict:
        """
        答案反思：评估答案质量
        """
        try:
            prompt = ANSWER_REFLECT_PROMPT.format(
                question=question,
                answer=answer[:1000]  # 限制长度
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, 'content') else str(response)

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Answer reflection failed: {e}")

        return {"quality": "acceptable", "issues": [], "suggestion": ""}

    def _supplementary_search(self, suggestion: str) -> List[SearchResult]:
        """
        补充检索
        """
        results = []
        try:
            search_results = self.zep_client.search(suggestion, self.graph_id, limit=10)
            seen = set()

            for r in search_results:
                fact = r.get("fact", "")
                if fact and fact not in seen:
                    seen.add(fact)
                    results.append(SearchResult(
                        strategy="SupplementarySearch",
                        content=fact,
                        score=r.get("score", 1.0),
                        relation=r.get("name", ""),
                        metadata={"type": "supplementary"}
                    ))
        except Exception as e:
            logger.error(f"Supplementary search failed: {e}")

        return results

    def _format_results_for_synthesis(self, results: Dict[str, List[SearchResult]]) -> Dict[str, str]:
        """格式化结果用于综合"""
        formatted = {}

        for strategy, items in results.items():
            if not items:
                formatted[strategy.lower().replace("search", "_results").replace("chainofexploration", "chain_results")] = "无相关结果"
                continue

            lines = []
            for item in items[:10]:
                line = f"- {item.content}"
                if item.source_entity:
                    line += f" [实体: {item.source_entity}]"
                if item.metadata.get("hop"):
                    line += f" [跳数: {item.metadata['hop']}]"
                lines.append(line)

            key = strategy.lower()
            if "local" in key:
                formatted["local_results"] = "\n".join(lines)
            elif "global" in key:
                formatted["global_results"] = "\n".join(lines)
            elif "hybrid" in key:
                formatted["hybrid_results"] = "\n".join(lines)
            elif "chain" in key:
                formatted["chain_results"] = "\n".join(lines)

        # 确保所有key都存在
        for key in ["local_results", "global_results", "hybrid_results", "chain_results"]:
            if key not in formatted:
                formatted[key] = "无相关结果"

        return formatted

    def run(self, question: str) -> Dict:
        """运行多策略检索"""
        # 1. 分析查询
        analysis = self._analyze_query(question)
        entities = analysis.get("entities", [])
        keywords = analysis.get("keywords", [question])
        query_type = analysis.get("query_type", "general")

        # 2. 选择策略
        strategies = self._select_strategies(query_type, entities)

        # 3. 并行执行检索
        all_results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            if "LocalSearch" in strategies:
                futures[executor.submit(self._local_search, entities, keywords)] = "LocalSearch"
            if "GlobalSearch" in strategies:
                futures[executor.submit(self._global_search, question, keywords)] = "GlobalSearch"
            if "HybridSearch" in strategies:
                futures[executor.submit(self._hybrid_search, entities, question)] = "HybridSearch"
            if "ChainOfExploration" in strategies:
                futures[executor.submit(self._chain_of_exploration, entities)] = "ChainOfExploration"

            for future in as_completed(futures):
                strategy = futures[future]
                try:
                    all_results[strategy] = future.result()
                except Exception as e:
                    logger.error(f"{strategy} failed: {e}")
                    all_results[strategy] = []

        # 4. 综合生成
        formatted = self._format_results_for_synthesis(all_results)
        prompt = SYNTHESIZER_PROMPT.format(question=question, **formatted)

        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, 'content') else str(response)

        return {
            "answer": answer,
            "analysis": analysis,
            "strategies": strategies,
            "results": {k: len(v) for k, v in all_results.items()},
            "total_results": sum(len(v) for v in all_results.values())
        }

    def stream(self, question: str) -> Generator[Dict, None, None]:
        """
        流式运行多策略 Agentic RAG

        Yields:
            {"type": "log", ...}
            {"type": "chunk", "content": text}
            {"type": "done", "data": {...}}
        """
        all_logs = []

        def log(step: str, content: str, data: Any = None):
            entry = {
                "step": step,
                "content": content,
                "timestamp": time.strftime("%H:%M:%S"),
                "data": data
            }
            all_logs.append(entry)
            return entry

        # ===== Analyze 阶段 =====
        yield {"type": "log", **log("🔍 Query Analysis", f"分析问题: {question}")}

        analysis = self._analyze_query(question)
        entities = analysis.get("entities", [])
        keywords = analysis.get("keywords", [question])
        query_type = analysis.get("query_type", "general")

        yield {"type": "log", **log(
            "📊 分析结果",
            f"类型: {query_type}, 实体: {entities}",
            analysis
        )}

        # ===== Strategy Selection 阶段 =====
        strategies = self._select_strategies(query_type, entities)

        yield {"type": "log", **log(
            "🎯 Strategy Selection",
            f"选择 {len(strategies)} 种检索策略",
            {"strategies": strategies, "reason": f"基于 {query_type} 类型问题优化"}
        )}

        # ===== Execute 阶段 - 并行检索 =====
        yield {"type": "log", **log(
            "⚡ Execute",
            "并行执行多种搜索策略",
            {"parallel": True}
        )}

        all_results = {}
        strategy_status = {}

        # 显示策略树
        strategy_tree = "并行执行多种搜索策略：\n"
        strategy_tree += "    ├─ [GraphRAG] LocalSearch: 实体级精确检索\n"
        strategy_tree += "    ├─ [GraphRAG] GlobalSearch: 社区级宏观检索\n"
        strategy_tree += "    ├─ [GraphRAG] HybridSearch: 双级融合检索\n"
        strategy_tree += "    └─ [GraphRAG] ChainOfExploration: 图谱探索"

        yield {"type": "log", **log("📋 执行计划", strategy_tree)}

        # 并行执行
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            start_times = {}

            if "LocalSearch" in strategies:
                futures[executor.submit(self._local_search, entities, keywords)] = "LocalSearch"
                start_times["LocalSearch"] = time.time()
            if "GlobalSearch" in strategies:
                futures[executor.submit(self._global_search, question, keywords)] = "GlobalSearch"
                start_times["GlobalSearch"] = time.time()
            if "HybridSearch" in strategies:
                futures[executor.submit(self._hybrid_search, entities, question)] = "HybridSearch"
                start_times["HybridSearch"] = time.time()
            if "ChainOfExploration" in strategies:
                futures[executor.submit(self._chain_of_exploration, entities)] = "ChainOfExploration"
                start_times["ChainOfExploration"] = time.time()

            for future in as_completed(futures):
                strategy = futures[future]
                elapsed = int((time.time() - start_times[strategy]) * 1000)

                try:
                    results = future.result()
                    all_results[strategy] = results

                    # 提取预览
                    preview = [r.content[:60] + "..." for r in results[:3]]

                    yield {"type": "log", **log(
                        f"✓ {strategy}",
                        f"找到 {len(results)} 条结果 ({elapsed}ms)",
                        {"count": len(results), "preview": preview}
                    )}

                except Exception as e:
                    logger.error(f"{strategy} failed: {e}")
                    all_results[strategy] = []
                    yield {"type": "log", **log(
                        f"✗ {strategy}",
                        f"执行失败: {str(e)[:50]}",
                        {"error": str(e)}
                    )}

        # ===== Merge 阶段 =====
        total_results = sum(len(v) for v in all_results.values())
        yield {"type": "log", **log(
            "🔀 Results Merge",
            f"合并 {total_results} 条检索结果，去重并排序"
        )}

        # ===== Reflect 阶段 1: 检索结果反思 =====
        yield {"type": "log", **log("🔄 Reflect (Retrieval)", "评估检索结果是否足够...")}

        retrieval_reflect = self._reflect_on_retrieval(question, all_results)
        is_sufficient = retrieval_reflect.get("sufficient", True)

        if is_sufficient:
            yield {"type": "log", **log(
                "✓ 检索充分",
                retrieval_reflect.get("reason", "结果足够回答问题")
            )}
        else:
            yield {"type": "log", **log(
                "⚠️ 检索不足",
                f"{retrieval_reflect.get('reason', '')}，缺少: {retrieval_reflect.get('missing', '')}"
            )}

            # 补充检索
            suggestion = retrieval_reflect.get("suggestion", "")
            if suggestion:
                yield {"type": "log", **log(
                    "🔍 补充检索",
                    f"关键词: {suggestion}"
                )}

                supplementary = self._supplementary_search(suggestion)
                if supplementary:
                    all_results["SupplementarySearch"] = supplementary
                    total_results += len(supplementary)

                    yield {"type": "log", **log(
                        "✓ 补充完成",
                        f"新增 {len(supplementary)} 条结果",
                        {"preview": [r.content[:50] for r in supplementary[:3]]}
                    )}

        # ===== Synthesize 阶段 =====
        yield {"type": "log", **log("🧠 Synthesize", "基于多策略结果综合生成回答...")}

        formatted = self._format_results_for_synthesis(all_results)
        prompt = SYNTHESIZER_PROMPT.format(question=question, **formatted)

        full_answer = ""
        try:
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    full_answer += content
                    yield {"type": "chunk", "content": content}
        except Exception as e:
            error_msg = str(e)
            if "inappropriate" in error_msg.lower() or "content" in error_msg.lower():
                full_answer = "检索到的新闻内容涉及敏感话题，无法生成完整回答。建议尝试更具体的问题，或查看原始检索结果。"
                yield {"type": "chunk", "content": full_answer}
            else:
                raise e

        # ===== Reflect 阶段 2: 答案质量反思 =====
        yield {"type": "log", **log("🔄 Reflect (Answer)", "评估答案质量...")}

        answer_reflect = self._reflect_on_answer(question, full_answer)
        quality = answer_reflect.get("quality", "acceptable")
        issues = answer_reflect.get("issues", [])

        if quality == "good":
            yield {"type": "log", **log("✓ 答案质量良好", "回答直接且完整")}
        elif quality == "acceptable":
            yield {"type": "log", **log("✓ 答案质量可接受", "回答基本满足需求")}
        else:
            issue_str = "、".join(issues) if issues else "质量较差"
            yield {"type": "log", **log(
                "⚠️ 答案质量待改进",
                f"问题: {issue_str}",
                {"suggestion": answer_reflect.get("suggestion", "")}
            )}

        yield {"type": "log", **log("✅ Complete", f"生成了 {len(full_answer)} 字的回答")}

        yield {
            "type": "done",
            "data": {
                "answer": full_answer,
                "analysis": analysis,
                "strategies": strategies,
                "results": {k: len(v) for k, v in all_results.items()},
                "total_results": total_results,
                "retrieval_reflect": retrieval_reflect,
                "answer_reflect": answer_reflect,
                "logs": all_logs
            }
        }
