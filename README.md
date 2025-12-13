# Multi-Agent News Collection System

基于 LangGraph 的多智能体新闻收集系统（双层循环 + 子图架构）

## 概述

这是一个企业级多智能体新闻分析系统，采用**分层子图架构**和**双层循环控制**，通过 9 个专业化 Agent 协作完成新闻收集、深度分析和智能报告生成。系统直接调用 NewsNow 聚合服务抓取多平台热点，使用 Qwen 系列模型进行智能分析。

### 核心特性

- **🏗️ 分层子图架构**: 3 个独立团队子图（Collection/Analysis/Summary），模块化设计，易于测试和扩展
- **🔁 双层循环控制**: 外层质量迭代 + 内层信息补全，确保数据完整性和分析深度
- **🤖 9 个专业 Agent**: 文字/视频收集、研究增强、情感分析、关系图谱、反思、时间线、趋势、图表、报告撰写
- **📊 智能报告生成**: 自动生成 Markdown 报告 + 可视化图表（时间线、趋势分析）
- **⚡ 性能优化**: 并行执行 + 智能跳过策略（第 2 轮起仅执行必要分析）
- **🔌 多数据源支持**: NewsNow 直连（今日头条/百度/微博/抖音/知乎/B站）+ Tavily 深度搜索

## 系统架构

### 主工作流（外层循环）

```
┌─────────────────┐
│   Coordinator   │  ← 初始化：预加载工具、设置参数
└────────┬────────┘
         │
┌────────▼────────┐
│ Main Supervisor │ ◄─┐  外层循环
└────────┬────────┘   │  (基于质量评分)
         │            │  max_iterations 控制
    ┌────┴────┐       │
    │         │       │
┌───▼───┐ ┌──▼──┐    │
│ News  │ │ Collection  │ → 分派新闻批次
│Collector │ Team (子图) │
└───┬───┘ └──┬──┘    │
    │       └─────────┤
    │                 │
┌───▼─────────────┐   │
│ Analysis Team   │───┤  内层循环见下图
│   (子图)        │   │
└───┬─────────────┘   │
    └─────────────────┤
                      │
┌─────────────────┐   │
│  Summary Team   │   │
│    (子图)       │   │
└────────┬────────┘   │
         │            │
         ▼            │
       [END]          │
```

### 分析团队子图（内层循环）

```
Coordinator
     │
     ▼
 ┌──────────────────────────┐
 │   Parallel Analyzers     │
 │  ┌────────────────────┐  │
 │  │ Research Agent     │  │ ← Tavily 搜索增强
 │  │ Sentiment Agent    │  │ ← 情感 + 风险分析
 │  │ Relationship Agent │  │ ← 因果关系图谱
 │  └────────────────────┘  │
 └──────────┬───────────────┘
            │
     ┌──────▼──────┐
     │ Aggregator  │  ← 汇总并行结果
     └──────┬──────┘
            │
     ┌──────▼──────┐
     │   Reflect   │  ← 识别信息缺口
     └──────┬──────┘
            │
     ┌──────▼──────┐
     │   Merger    │ ◄─┐  内层循环
     └──────┬──────┘   │  (基于 information_gaps)
            │          │  max_cycles 控制
      [有缺口?]       │
         yes ─────────┘
          no
         │
         ▼
       [END]
```

### 三大团队职责

| 团队 | 组成 | 职责 | 执行模式 |
|------|------|------|----------|
| **Collection Team** | text_collector, video_collector | 从新闻池分派并收集文字/视频新闻 | 并行执行 |
| **Analysis Team** | research, sentiment, relationship, reflect | 深度分析、增强、反思 | 并行 + 循环 |
| **Summary Team** | timeline, trend, chart, writer | 生成时间线、趋势、图表、最终报告 | 顺序执行 |

## 双层循环机制

### 🔄 外层循环（质量迭代）

**控制节点**: `main_supervisor_node`
**循环变量**: `iteration` (限制: `max_iterations`, 默认 3)
**循环条件**:
- 新闻数量不足（< MIN_TEXT_NEWS 或 < MIN_VIDEO_NEWS）
- 质量评分未达标（< QUALITY_THRESHOLD）
- 未达到最大迭代次数

**流程**:
```
supervisor → news_collector → supervisor → collection_team
    ↑                                           ↓
    └─────── [质量不足] ←── analysis_team ──────┘
                                ↓
                          [质量达标]
                                ↓
                          summary_team → END
```

### 🔁 内层循环（信息补全）

**控制节点**: `analysis_team/merger`
**循环变量**: `cycle_count` (限制: `max_cycles`, 默认 3)
**循环条件**:
- `reflect` 节点识别出信息缺口（information_gaps）
- 未达到最大循环次数

**智能优化策略**:
- **第 1 轮**: 完整分析（research + sentiment + relationship）
- **第 2+ 轮**: 仅执行 research（`skip_sentiment_relationship=True`）

## 9 个专业 Agent 详解

### Collection Team (收集团队)

| Agent | 模型 | 职责 | 工具 |
|-------|------|------|------|
| **text_collector** | qwen-plus | 收集并初步分析文字新闻 | get_latest_news, search_news |
| **video_collector** | qwen-vl-max | 收集并分析视频新闻（多模态） | get_latest_news, search_news |

### Analysis Team (分析团队)

| Agent | 模型 | 职责 | 工具 |
|-------|------|------|------|
| **research** | qwen-plus | 使用 Tavily 搜索深度调研，填补信息缺口 | tavily_search |
| **sentiment** | qwen-plus | 情感分析 + 风险评估 | - |
| **relationship** | qwen-plus | 构建因果关系和实体关系图谱 | - |
| **reflect** | qwen-plus | 反思分析质量，识别信息缺口 | - |

### Summary Team (总结团队)

| Agent | 模型 | 职责 | 输出 |
|-------|------|------|------|
| **timeline** | qwen-plus | 提取事件时间线 | timeline_analysis |
| **trend** | qwen-plus | 识别趋势和热词 | trend_analysis, trending_keywords |
| **chart** | qwen-plus | 生成可视化图表（matplotlib） | timeline_chart_path |
| **writer** | qwen-plus | 撰写最终 Markdown 报告 | final_report |

## 快速开始

### 1. 环境要求

- Python 3.8+
- （可选）Ollama（本地部署模式）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境

```bash
cp .env.example .env
# 编辑 .env，至少配置以下内容：
# - DASHSCOPE_API_KEY (Qwen 模型)
# - TAVILY_API_KEY (可选，用于深度搜索)
```

### 4. 运行系统

```bash
# 基础运行
python main.py

# 自定义任务和日期
python main.py --task "分析科技行业动态" --date "2025-11-25"

# 控制迭代次数
python main.py --max-iterations 5
```

## 项目结构

```
news_agent_system/
├── src/
│   ├── graph/
│   │   ├── types.py              # State 类型定义（State, CollectionTeamState, AnalysisTeamState, SummaryTeamState）
│   │   ├── workflow.py           # 主工作流（8 节点 + 3 子图）
│   │   └── teams/                # 🆕 团队子图
│   │       ├── collection_team.py   # 收集团队子图
│   │       ├── analysis_team.py     # 分析团队子图（含内层循环）
│   │       └── summary_team.py      # 总结团队子图
│   ├── agents/
│   │   └── nodes.py              # 所有 Agent 节点实现
│   ├── services/
│   │   └── newsnow_service.py    # NewsNow HTTP 聚合服务
│   ├── tools/
│   │   ├── news_tools.py         # 新闻收集工具
│   │   └── research_tools.py     # Tavily 研究工具
│   ├── llms/
│   │   └── llm.py                # LLM 提供商管理（Qwen/Ollama）
│   ├── config/
│   │   ├── settings.py           # 系统配置
│   │   └── agents.py             # Agent → LLM 映射
│   ├── prompts/
│   │   └── __init__.py           # 所有 Agent 提示词
│   └── utils/
│       ├── news_dedup.py         # 新闻去重工具
│       └── output_formatter.py   # 报告格式化
├── output/                       # 🆕 输出目录
│   ├── {date}.md                 # 每日 Markdown 报告
│   ├── charts/                   # 图表文件（PNG）
│   └── steps/{date}/             # 各 Agent 的中间输出
├── tests/                        # 测试文件
├── main.py                       # 主程序入口
├── requirements.txt              # Python 依赖
├── .env.example                  # 环境变量模板
├── CLAUDE.md                     # 开发者指南
└── README.md                     # 本文件
```

## 配置说明

### 核心环境变量

```bash
# ===== LLM 配置 =====
LLM_PROVIDER=qwen                          # qwen (云端) 或 ollama (本地)
DASHSCOPE_API_KEY=your-key-here            # Qwen 云端 API 密钥
QWEN_MODEL=qwen-plus                       # 文字模型
QWEN_VL_MODEL=qwen-vl-max-latest           # 视频模型

# ===== 工作流控制 =====
MAX_ITERATIONS=3                           # 外层循环最大迭代次数
QUALITY_THRESHOLD=0.7                      # 质量阈值
MIN_TEXT_NEWS=5                            # 最少文字新闻
MIN_VIDEO_NEWS=3                           # 最少视频新闻

# ===== NewsNow 配置 =====
USE_DIRECT_MODE=true                       # 推荐：直接导入（无子进程开销）
NEWSNOW_BASE_URL=https://newsnow.busiyi.world
NEWSNOW_PER_PLATFORM_LIMIT=30              # 单平台限制
NEWSNOW_MAX_ITEMS=200                      # 总条数上限
NEWSNOW_PLATFORMS=toutiao:今日头条,baidu:百度热搜,...
ENABLE_INTERACTIVE_CRAWLER=false            # 控制是否使用爬虫直接爬取到文章（开启：true 需要手动处理网页安全检测，关闭：false 直接返回整体热搜url）

# ===== 研究工具 =====
TAVILY_API_KEY=your-tavily-key             # 可选：深度搜索增强

# ===== 邮件通知 =====
EMAIL_ENABLED=false                        # 启用后会发送 HTML 报告
EMAIL_SMTP_HOST=smtp.example.com
EMAIL_TO=you@example.com

# ===== 调试 =====
LOG_LEVEL=INFO                             # DEBUG 可查看详细日志
LANGCHAIN_TRACING_V2=true                  # LangSmith 追踪（推荐）
```

### 高级配置

```bash
# Analysis Team 内层循环控制
ANALYSIS_MAX_CYCLES=3                      # 最大分析循环次数

# Supervisor 批次分派
SUPERVISOR_BATCH_SIZE=10                   # 每轮分派给 agents 的新闻数

# 新闻去重
DEDUP_SIMILARITY_THRESHOLD=0.85            # 标题相似度阈值（difflib）
```

## 输出示例

### 报告结构

系统会生成以下文件：

```
output/
├── 2025-11-25.md                         # 📄 主报告
├── charts/
│   └── timeline_2025-11-25.png           # 📊 时间线图表
└── steps/2025-11-25/
    ├── 01_text_collector.json            # 文字收集结果
    ├── 02_research.json                  # 研究增强结果
    ├── 03_sentiment.json                 # 情感分析
    ├── 04_relationship.json              # 关系图谱
    ├── 05_reflect.json                   # 反思结果
    ├── 06_timeline.json                  # 时间线提取
    ├── 07_trend.json                     # 趋势分析
    └── 08_final_report.md                # 最终报告
```

### 报告内容示例

```markdown
# 2025-11-25 新闻收集报告

- 任务: 获取今日最热新闻
- 生成时间: 2025-11-25T17:45:23
- 文字新闻条数: 40
- 视频新闻条数: 0

## 一、执行摘要
今日舆情呈现"科技高光主导、社会情绪撕裂、国际互动升温"的三重格局...

## 二、重点新闻清单
| 标题 | 来源 | 时间 | 链接 |
|------|------|------|------|
| 神舟二十二号飞船发射圆满成功 | 央视新闻 | 2025-11-25 | [链接](...) |
...

## 三、新闻要点分类汇总
### 📄 文字新闻
[40 条新闻详情]

### 🎥 视频新闻
[视频分析]

## 四、关键洞察与趋势分析
### 🔍 关键洞察
1. 国家战略叙事强势回归...
2. 代际冲突与育儿焦虑升级...

### 📈 趋势脉络
| 主题 | 当前状态 | 潜在走向 |
|------|----------|-----------|
| 航天任务进展 | 发射成功 | 连续报道链，科普热潮 |
...

## 五、情感分析与风险评估
[情感极性分布、风险等级]

## 六、关系图谱
[实体关系、因果链]

## 七、时间线可视化
![时间线图表](./charts/timeline_2025-11-25.png)
```

### 控制台输出示例

```
============================================================
Multi-Agent News Collection System (Subgraph Architecture)
============================================================

=== Coordinator: Initializing ===
✓ Preloaded text_tools: 2 tools
✓ Preloaded research_tools: 1 tool

=== Main Supervisor: Evaluating (Iteration 0) ===
→ Decision: news_collector (fetch initial pool)

=== News Collector: Fetching ===
✓ Fetched 150 news items from 6 platforms
✓ Deduplicated: 150 → 122 unique

=== Main Supervisor: Evaluating (Iteration 1) ===
→ Decision: collection_team (batch 0-10)

=== Invoking Collection Team Subgraph ===
  ├─ text_collector: Collected 8 text news
  └─ video_collector: Collected 0 video news

=== Main Supervisor: Evaluating (Iteration 2) ===
→ Quality: 0.45 (below threshold 0.7)
→ Decision: analysis_team

=== Invoking Analysis Team Subgraph ===
  [Cycle 1] Full analysis mode
  ├─ research (parallel): Added 5 research notes
  ├─ sentiment (parallel): Analyzed 8 items
  └─ relationship (parallel): Built 3 entity chains
  ├─ reflect: Identified 2 information gaps
  └─ merger: Looping back (gaps found)

  [Cycle 2] Research-only mode (optimized)
  ├─ research: Filled 2 gaps
  ├─ reflect: No further gaps
  └─ merger: Loop complete ✓

=== Main Supervisor: Evaluating (Iteration 3) ===
→ Quality: 0.82 (达标)
→ Decision: summarize

=== Invoking Summary Team Subgraph ===
  ├─ timeline: Extracted 12 events
  ├─ trend: Identified 5 trending keywords
  ├─ chart: Generated timeline_2025-11-25.png
  └─ writer: Generated final report

============================================================
Execution Completed!
============================================================
Report: output/2025-11-25.md
Chart: output/charts/timeline_2025-11-25.png
Duration: 87.3s
Iterations: 3 (外层) + 2 (内层分析循环)
============================================================
```

## 工作流程详解

### 阶段 1: 初始化 (Coordinator)

- 预加载所有工具（避免重复初始化）
- 设置系统参数（task, date, max_iterations）
- 初始化状态字段

### 阶段 2: 外层循环 (Main Supervisor)

```
WHILE iteration < max_iterations AND quality < threshold:
    1. 评估当前状态（新闻数、质量分）
    2. 决策下一步：
       - news_collector: 获取新闻池
       - collection_team: 分派并收集
       - analysis_team: 深度分析
       - summarize: 生成报告
    3. iteration += 1
```

### 阶段 3: 收集团队 (Collection Team)

- **Coordinator**: 从新闻池按 batch_size 分派
- **text_collector** ∥ **video_collector**: 并行收集
- **Merger**: 合并结果回主状态

### 阶段 4: 分析团队 (Analysis Team + 内层循环)

```
WHILE has_information_gaps AND cycle_count < max_cycles:
    IF cycle_count == 0:
        # 第一轮：全面分析
        [research ∥ sentiment ∥ relationship] → reflect
    ELSE:
        # 第 2+ 轮：仅研究（优化）
        [research] → reflect

    cycle_count += 1
```

### 阶段 5: 总结团队 (Summary Team)

- **Timeline**: 提取事件时间线
- **Trend**: 分析趋势和热词
- **Chart**: 生成 matplotlib 可视化
- **Writer**: 撰写 Markdown 报告

## 开发指南

### 添加新 Agent

1. **定义节点函数**（`src/agents/nodes.py`）:
```python
async def my_agent_node(state: State):
    llm = get_llm_by_type(AGENT_LLM_MAP["my_agent"])
    agent = create_agent(llm, tools, get_system_prompt("my_agent"))
    result = await agent.ainvoke({...})
    return Command(update={...}, goto="supervisor")
```

2. **添加系统提示词**（`src/prompts/__init__.py`）:
```python
PROMPTS["my_agent"] = "You are a specialized agent for..."
```

3. **注册到团队子图**（`src/graph/teams/*.py`）:
```python
team.add_node("my_agent", my_agent_node)
team.add_edge("coordinator", "my_agent")
```

4. **映射 LLM 类型**（`src/config/agents.py`）:
```python
AGENT_LLM_MAP["my_agent"] = "qwen"
```

### 修改子图循环逻辑

编辑 `src/graph/teams/analysis_team.py` 的 `merger` 函数：

```python
def merger(state: AnalysisTeamState):
    # 自定义循环条件
    if custom_condition and cycle_count < max_cycles:
        return Command(update={...}, goto="coordinator")
    return Command(update={...}, goto=END)
```

### 调试技巧

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
export DEBUG=True

# 启用 LangSmith 追踪（推荐）
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key

# 运行单个团队测试
python -c "from src.graph.teams.analysis_team import create_analysis_team; create_analysis_team()"
```

## 性能优化

### 当前优化

1. **工具预加载**: Coordinator 预加载所有工具，避免重复初始化
2. **智能跳过**: 分析第 2 轮起仅执行 research（跳过 sentiment/relationship）
3. **并行执行**: 使用 Send API 并行执行 research/sentiment/relationship
4. **直连模式**: `USE_DIRECT_MODE=true` 避免 MCP 子进程开销
5. **新闻去重**: 自动去重，减少重复分析

### 进一步优化建议

- 添加 Redis 缓存（新闻、分析结果）
- 使用 LangGraph 的 checkpoint 持久化状态
- 配置 PostgreSQL 存储历史报告
- 启用 LangSmith 监控性能瓶颈

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| **工作流编排** | LangGraph | 0.2.0+ |
| **Agent 框架** | LangChain | 0.3.0+ |
| **LLM 提供商** | Qwen (DashScope) | - |
| **多模态模型** | Qwen-VL | - |
| **深度搜索** | Tavily | - |
| **数据聚合** | NewsNow HTTP Service | - |
| **可视化** | Matplotlib | 3.8.0+ |
| **HTTP 客户端** | httpx | 0.27.0+ |

## 常见问题

### Q: 如何减少 API 调用成本？

A:
1. 降低 `MAX_ITERATIONS` 和 `ANALYSIS_MAX_CYCLES`
2. 减少 `SUPERVISOR_BATCH_SIZE`
3. 设置更低的 `MIN_TEXT_NEWS` 和 `MIN_VIDEO_NEWS`
4. 使用 Ollama 本地部署（`LLM_PROVIDER=ollama`）

### Q: 分析团队的内层循环何时终止？

A: 满足以下任一条件：
- `reflect` 未识别出信息缺口（`information_gaps` 为空）
- 达到最大循环次数（`ANALYSIS_MAX_CYCLES`，默认 3）

### Q: 如何禁用某个 Agent？

A: 修改对应团队子图的路由逻辑，例如在 `analysis_team.py` 中跳过 sentiment：
```python
return [Send("research", state)]  # 不发送 sentiment 和 relationship
```

### Q: 支持哪些新闻平台？

A: 默认支持（通过 NewsNow）:
- 今日头条、百度热搜、微博热搜、抖音热点、知乎热榜、Bilibili 热搜

可通过 `NEWSNOW_PLATFORMS` 环境变量自定义。

## 更新日志

### v2.0.0 (2025-11-25) - 双层循环架构

- ✨ 重构为分层子图架构（3 个团队子图）
- ✨ 新增内层分析循环（基于 information_gaps）
- ✨ 新增 6 个专业 Agent（research, sentiment, relationship, reflect, timeline, trend）
- ✨ 支持并行执行（Send API）
- ✨ 智能优化策略（第 2 轮起仅执行 research）
- 🐛 修复子图递归限制问题（使用 conditional_edges + END）
- 📊 增强报告生成（时间线图表、趋势分析）

### v1.0.0 (2025-11-19) - 初始版本

- 基础扁平化架构
- Text/Video Agent
- NewsNow 直连服务

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- 项目文档: [CLAUDE.md](./CLAUDE.md)
- 问题反馈: GitHub Issues

---

**最后更新**: 2025-11-30
**架构版本**: v2.0 (Subgraph + Dual-Loop)
