# Multi-Agent News Collection System (v2)

基于 LangGraph 的多智能体新闻收集系统 - 改进版

## 概述

这是一个基于需求文档设计的新闻收集系统，采用多智能体协作架构，直接调用 NewsNow 聚合服务抓取新闻数据，使用 Qwen 模型进行分析和理解。

### 核心特性

- **多智能体架构**: Coordinator → Supervisor → Text/Video Agents → Summary
- **NewsNow 直连**: 通过内置抓取服务获取多平台热点新闻
- **智能决策**: Supervisor 动态评估质量并控制迭代
- **双模态分析**: 文字新闻(Qwen3) + 视频新闻(Qwen3-VL)
- **迭代优化**: 自动调整收集策略直到达到质量要求

## 系统架构

```
┌─────────────┐
│ Coordinator │  # 初始化系统
└──────┬──────┘
       │
┌──────▼──────┐
│ Supervisor  │  # 评估质量、控制迭代
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
┌──▼──┐ ┌──▼──┐
│Text │ │Video│  # 专业 Agent + 新闻工具
│Agent│ │Agent│
└──┬──┘ └──┬──┘
   │       │
   └───┬───┘
       │
┌──────▼──────┐
│  Supervisor │  # 再次评估
└──────┬──────┘
       │
   [循环或结束]
       │
┌──────▼──────┐
│   Summary   │  # 生成报告
└─────────────┘
```

## 项目结构

```
news_multi_agent/
├── src/
│   ├── graph/
│   │   ├── types.py          # 状态类型定义
│   │   └── workflow.py       # LangGraph 工作流
│   ├── agents/
│   │   └── nodes.py          # 所有 Agent 节点
│   ├── services/
│   │   └── newsnow_service.py  # NewsNow 聚合服务
│   ├── llms/
│   │   └── llm.py            # LLM 提供商管理
│   ├── config/
│   │   ├── settings.py       # 系统配置
│   │   └── agents.py         # Agent 配置
│   └── prompts/
│       └── __init__.py       # Agent 提示词
├── tests/                    # 测试文件
├── main.py                   # 主程序入口
├── requirements.txt          # Python 依赖
├── .env.example              # 环境变量模板
└── README.md                 # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

```bash
cp .env.example .env
# 编辑 .env 文件，配置 API 密钥和服务器
```

### 3. 运行系统

```bash
python main.py
```

### 自定义参数

```bash
python main.py --task "获取科技新闻" --date "2025-11-19" --max-iterations 5
```

## 配置说明

### LLM 配置

```bash
DASHSCOPE_API_KEY=your-api-key             # DashScope API 密钥
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1  # 可切换为 intl 域名
QWEN_MODEL=qwen-plus                       # 文字模型
QWEN_VL_MODEL=qwen-vl-max-latest           # 视频模型
```

### Agent 配置

```bash
MAX_ITERATIONS=3               # 最大迭代次数
QUALITY_THRESHOLD=0.7          # 质量阈值
MIN_TEXT_NEWS=3                # 最少文字新闻
MIN_VIDEO_NEWS=3               # 最少视频新闻
```

### NewsNow 抓取配置

```bash
NEWSNOW_BASE_URL=https://newsnow.busiyi.world  # 数据源地址
NEWSNOW_TIMEOUT_SECONDS=15                     # HTTP 请求超时
NEWSNOW_PER_PLATFORM_LIMIT=30                  # 单个平台最大条数
NEWSNOW_MAX_ITEMS=200                          # 全平台总上限
NEWSNOW_PLATFORMS="weibo:微博热搜,baidu:百度热搜,..."  # 可自定义平台与展示名
NEWSNOW_USER_AGENT="..."                       # 可覆盖默认 UA
NEWSNOW_COOKIE=""                              # 如目标站点需要可设置
```

## Agent 说明

### Coordinator (协调器)
- 初始化系统状态
- 设置工作流参数

### Supervisor (监督器)
- 评估收集质量
- 控制迭代次数
- 决定下一步操作
- 路由到合适的 Agent

### Text Agent (文字新闻)
- 使用 Qwen3 模型
- 调用内置新闻工具收集文字新闻
- 分析文本内容
- 提取关键信息

### Video Agent (视频新闻)
- 使用 Qwen3-VL 模型
- 调用内置新闻工具收集视频新闻
- 分析视频内容
- 提取视觉洞察

### Summary Agent (总结)
- 综合所有收集的新闻
- 生成结构化报告
- 组织分类内容

## NewsNow 服务

系统内置 `NewsNowService`（`src/services/newsnow_service.py`），直接通过 HTTP
聚合今日头条 / 百度热搜 / 微博热搜 / 抖音热点 / 知乎热榜 / B 站热搜等平台的
热门新闻，替代早期的 MCP 部署模式。所有 LangChain 工具（如 `get_latest_news`,
`search_news` 等）都基于该服务封装，无需额外启动 MCP 服务器即可运行。

- 如需扩展或更换数据源，可修改 `NewsNowService` 里的平台配置或 env 变量
- 工具层（`src/tools/news_tools.py`）已经默认启用去重、关键词统计等增强逻辑
- 其他 Agent（Text/Video/Summary）通过 LangGraph 的状态共享这些工具

## 工作流程

1. **初始化**: Coordinator 设置初始状态
2. **迭代循环**:
   - Supervisor 评估当前状态
   - 决定调用 Text Agent 或 Video Agent
   - Agent 通过新闻工具收集数据
   - Agent 使用 Qwen 模型分析
   - 返回 Supervisor 评估
3. **终止条件**:
   - 达到质量阈值 + 足够新闻数量
   - 或达到最大迭代次数
4. **生成报告**: Summary Agent 创建最终报告

## 示例输出

```
============================================================
Multi-Agent News Collection System
============================================================

=== System Configuration ===
Qwen Model: qwen3
Qwen-VL Model: qwen3-vl
...

=== Coordinator: Initializing news collection ===
=== Supervisor: Evaluating progress (Iteration 0) ===
=== Text Agent: Collecting text news (Iteration 0) ===
=== Supervisor: Evaluating progress (Iteration 1) ===
=== Video Agent: Collecting video news (Iteration 1) ===
...
=== Summary Agent: Generating final report ===

============================================================
Execution Completed!
============================================================

[生成的新闻报告]

============================================================
Statistics
============================================================
Total Iterations: 3
Text News Collected: 5
Video News Collected: 4
Final Quality Score: 0.75
Duration: 45.23 seconds
```

## 开发指南

### 添加新的 Agent

1. 在 `src/agents/nodes.py` 中定义节点函数
2. 在 `src/graph/workflow.py` 中注册节点
3. 更新路由逻辑
4. 添加提示词到 `src/prompts/__init__.py`

### 调试模式

```bash
# 设置环境变量
export DEBUG=True
export LOG_LEVEL=DEBUG

# 运行系统
python main.py
```

## 技术栈

- **LangGraph**: Multi-agent 工作流编排
- **NewsNowService**: 内置新闻聚合服务
- **LangChain**: Agent 和工具管理
- **Qwen3**: 文字理解模型
- **Qwen3-VL**: 视频理解模型
- **Python 3.8+**

## 与 v1 的区别

| 特性 | v1 (news_agent_system) | v2 (news_multi_agent) |
|------|----------------------|---------------------|
| 架构 | 基础框架 | 完整实现 |
| 数据源 | 手动请求 | 内置 NewsNowService 统一采集 |
| Agent 创建 | 手动实现 | 使用 create_agent |
| 状态更新 | 字典返回 | Command 模式 |
| 工具使用 | 未实现 | LangChain 工具集成 |
| 代码风格 | 简单 | 参考金融系统 |

## 后续优化

- [ ] 扩展更多数据源/平台
- [ ] 添加更多评估指标
- [ ] 加强新闻去重与打分
- [ ] 添加缓存机制
- [ ] 数据库持久化
- [ ] Web 界面
- [ ] 单元测试

## 许可证

MIT License

**最后更新**: 2025-11-19
