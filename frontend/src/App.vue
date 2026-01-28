<template>
  <div class="app">
    <!-- Full-screen Graph -->
    <div class="graph-fullscreen">
      <GraphPanel
        ref="graphPanel"
        :nodes="filteredNodes"
        :edges="filteredEdges"
        @node-click="handleNodeClick"
      />
    </div>

    <!-- Date Filter (floating top-left) -->
    <div class="date-filter">
      <div class="filter-label">日期筛选</div>
      <div class="filter-options">
        <button
          v-for="opt in dateOptions"
          :key="opt.value"
          :class="['filter-btn', { active: selectedDateRange === opt.value }]"
          @click="selectedDateRange = opt.value"
        >
          {{ opt.label }}
        </button>
      </div>
      <div class="filter-custom" v-if="selectedDateRange === 'custom'">
        <input type="date" v-model="customDate" />
      </div>
      <div class="filter-stats">
        {{ filteredNodes.length }} 节点 / {{ filteredEdges.length }} 关系
      </div>
    </div>

    <!-- Header Stats (floating top-right) -->
    <div class="stats-badge" v-if="stats">
      <span>{{ stats.node_count }} 节点</span>
      <span class="divider">|</span>
      <span>{{ stats.edge_count }} 关系</span>
      <button class="refresh-btn" @click="refreshGraph" :disabled="loading">
        <span v-if="loading">⏳</span>
        <span v-else>🔄</span>
      </button>
    </div>

    <!-- Bottom Search Bar -->
    <div class="search-container">
      <div class="search-box">
        <input
          v-model="searchQuery"
          type="text"
          placeholder="向知识图谱提问..."
          @keyup.enter="handleSearch"
          :disabled="isStreaming"
        />
        <select v-model="searchMode" class="mode-select">
          <option value="quick">快速</option>
          <option value="agentic">智能 Agent</option>
        </select>
        <button class="search-btn" @click="handleSearch" :disabled="isStreaming || !searchQuery.trim()">
          <span v-if="isStreaming">思考中...</span>
          <span v-else>🔍 搜索</span>
        </button>
      </div>
      <div class="quick-suggestions" v-if="!chatResult">
        <span
          v-for="(suggestion, idx) in suggestions"
          :key="idx"
          class="suggestion-chip"
          @click="searchQuery = suggestion"
        >
          {{ suggestion }}
        </span>
      </div>
    </div>

    <!-- Left Slide Panel (Results) -->
    <transition name="slide">
      <div class="result-panel" v-if="showResultPanel">
        <div class="panel-header">
          <h3>搜索结果</h3>
          <button class="close-btn" @click="closeResultPanel">×</button>
        </div>

        <div class="panel-content">
          <!-- Question -->
          <div class="question-block">
            <div class="question-label">问题</div>
            <div class="question-text">{{ lastQuery }}</div>
          </div>

          <!-- Agent Logs -->
          <div class="agent-logs" v-if="agentLogs.length">
            <div class="logs-header" @click="logsExpanded = !logsExpanded">
              <span class="logs-title">
                🤖 Agent 执行日志
                <span class="logs-count" v-if="!logsExpanded && chatResult.status === 'done'">
                  ({{ agentLogs.length }} 条记录)
                </span>
              </span>
              <span class="logs-toggle">{{ logsExpanded ? '▼ 收起' : '▶ 展开' }}</span>
            </div>
            <div class="logs-content" v-show="logsExpanded || chatResult.status === 'thinking' || chatResult.status === 'streaming'">
              <div class="log-item" v-for="(log, idx) in agentLogs" :key="idx">
                <div class="log-header">
                  <span class="log-step">{{ log.step }}</span>
                  <span class="log-time">{{ log.timestamp }}</span>
                </div>
                <div class="log-content">{{ log.content }}</div>
                <div class="log-data" v-if="log.data">
                  <!-- 显示工具参数 -->
                  <template v-if="log.data.tool">
                    <span class="log-tag">{{ log.data.tool }}</span>
                    <span class="log-args" v-if="log.data.args">
                      {{ JSON.stringify(log.data.args) }}
                    </span>
                  </template>
                  <!-- 显示结果预览 -->
                  <template v-if="log.data.preview && log.data.preview.length">
                    <div class="log-preview">
                      <div v-for="(item, i) in log.data.preview" :key="i" class="preview-item">
                        • {{ item.slice(0, 60) }}{{ item.length > 60 ? '...' : '' }}
                      </div>
                    </div>
                  </template>
                  <!-- 显示计划步骤 -->
                  <template v-if="Array.isArray(log.data) && log.data[0]?.purpose">
                    <div class="log-plan">
                      <div v-for="(p, i) in log.data" :key="i" class="plan-item">
                        {{ i + 1 }}. {{ p.purpose }}
                      </div>
                    </div>
                  </template>
                </div>
              </div>
              <!-- 加载指示器 -->
              <div class="log-item loading" v-if="chatResult.status === 'thinking'">
                <span class="thinking-dots"></span>
              </div>
            </div>
          </div>

          <!-- Answer -->
          <div class="answer-block" v-if="chatResult && (chatResult.answer || chatResult.status === 'streaming')">
            <div class="answer-label">回答</div>
            <div class="answer-text">
              <span v-html="formatAnswer(chatResult.answer)"></span>
              <span class="streaming-cursor" v-if="chatResult.status === 'streaming'">▌</span>
            </div>
          </div>

          <!-- Insights -->
          <div class="insights-block" v-if="chatResult?.insights?.length">
            <div class="insights-label">💡 洞察</div>
            <ul class="insights-list">
              <li v-for="(insight, idx) in chatResult.insights" :key="idx">
                {{ insight }}
              </li>
            </ul>
          </div>

          <!-- Sources -->
          <div class="sources-block" v-if="chatResult?.sources?.length">
            <div class="sources-label">📚 信息来源</div>
            <div class="sources-list">
              <div
                v-for="(source, idx) in chatResult.sources"
                :key="idx"
                class="source-item"
              >
                {{ source }}
              </div>
            </div>
          </div>

          <!-- Metadata -->
          <div class="meta-block" v-if="chatResult">
            <span>搜索模式: {{ getModeLabel(chatResult.retrieval_mode) }}</span>
            <span>检索结果: {{ chatResult.result_count }} 条</span>
          </div>
        </div>
      </div>
    </transition>

    <!-- Node Detail Modal -->
    <div class="node-modal" v-if="selectedNode" @click.self="selectedNode = null">
      <div class="modal-content">
        <button class="close-btn" @click="selectedNode = null">×</button>
        <h3>{{ selectedNode.name }}</h3>
        <div class="node-labels">
          <span class="label-tag" v-for="label in selectedNode.labels" :key="label">
            {{ label }}
          </span>
        </div>
        <p class="node-summary" v-if="selectedNode.summary">{{ selectedNode.summary }}</p>
        <button class="ask-btn" @click="askAboutNode(selectedNode)">
          🔍 询问关于此实体
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'
import GraphPanel from './components/GraphPanel.vue'
import { getGraphData, getGraphStats, streamQuestion, getSuggestions } from './api/graphrag'

export default {
  name: 'App',
  components: { GraphPanel },
  setup() {
    const nodes = ref([])
    const edges = ref([])
    const stats = ref(null)
    const loading = ref(false)
    const chatLoading = ref(false)
    const selectedNode = ref(null)
    const searchQuery = ref('')
    const searchMode = ref('agentic')
    const showResultPanel = ref(false)
    const chatResult = ref(null)
    const lastQuery = ref('')
    const suggestions = ref([
      '最近有哪些热点新闻？',
      '今天科技领域发生了什么？',
      '有哪些重要事件？'
    ])

    // Date filtering
    const selectedDateRange = ref('all')
    const customDate = ref('')
    const dateOptions = [
      { label: '全部', value: 'all' },
      { label: '今天', value: 'today' },
      { label: '昨天', value: 'yesterday' },
      { label: '近3天', value: '3days' },
      { label: '近7天', value: '7days' },
      { label: '自定义', value: 'custom' }
    ]

    // Get local date string (YYYY-MM-DD) from Date object
    const toLocalDateStr = (date) => {
      const year = date.getFullYear()
      const month = String(date.getMonth() + 1).padStart(2, '0')
      const day = String(date.getDate()).padStart(2, '0')
      return `${year}-${month}-${day}`
    }

    // Get list of valid date strings for current filter
    const getValidDateStrings = () => {
      const now = new Date()
      const dates = []

      const addDays = (d, days) => {
        const result = new Date(d)
        result.setDate(result.getDate() + days)
        return result
      }

      switch (selectedDateRange.value) {
        case 'today':
          dates.push(toLocalDateStr(now))
          break
        case 'yesterday':
          dates.push(toLocalDateStr(addDays(now, -1)))
          break
        case '3days':
          for (let i = 0; i < 3; i++) {
            dates.push(toLocalDateStr(addDays(now, -i)))
          }
          break
        case '7days':
          for (let i = 0; i < 7; i++) {
            dates.push(toLocalDateStr(addDays(now, -i)))
          }
          break
        case 'custom':
          if (customDate.value) {
            dates.push(customDate.value) // Already in YYYY-MM-DD format
          }
          break
        default:
          return null // 'all' - no filter
      }
      return new Set(dates)
    }

    // Parse created_at to local date string
    const parseToLocalDateStr = (dateStr) => {
      if (!dateStr) return null
      const date = new Date(dateStr)
      if (isNaN(date.getTime())) return null
      return toLocalDateStr(date)
    }

    // Filtered nodes based on date
    const filteredNodes = computed(() => {
      const validDates = getValidDateStrings()
      if (!validDates) return nodes.value // 'all' - no filter

      return nodes.value.filter(node => {
        const dateStr = parseToLocalDateStr(node.created_at)
        if (!dateStr) return false
        return validDates.has(dateStr)
      })
    })

    // Get node UUIDs for edge filtering
    const filteredNodeUuids = computed(() => {
      return new Set(filteredNodes.value.map(n => n.uuid))
    })

    // Filtered edges - only edges connecting filtered nodes
    const filteredEdges = computed(() => {
      const nodeSet = filteredNodeUuids.value
      return edges.value.filter(edge =>
        nodeSet.has(edge.source) && nodeSet.has(edge.target)
      )
    })

    const refreshGraph = async () => {
      loading.value = true
      try {
        const graphRes = await getGraphData()
        if (graphRes.data) {
          nodes.value = graphRes.data.nodes || []
          edges.value = graphRes.data.edges || []
        }

        const statsRes = await getGraphStats()
        if (statsRes.data) {
          stats.value = statsRes.data
        }
      } catch (error) {
        console.error('Failed to load graph:', error)
      } finally {
        loading.value = false
      }
    }

    const streamingAnswer = ref('')
    const isStreaming = ref(false)
    const agentLogs = ref([])  // 记录 agent 的执行日志
    const logsExpanded = ref(true)  // 日志是否展开

    const handleSearch = async () => {
      if (!searchQuery.value.trim() || isStreaming.value) return

      lastQuery.value = searchQuery.value
      showResultPanel.value = true
      streamingAnswer.value = ''
      isStreaming.value = true
      agentLogs.value = []
      logsExpanded.value = true  // 新搜索时展开日志

      // 立即显示思考状态
      chatResult.value = {
        answer: '',
        sources: [],
        status: 'thinking'
      }

      try {
        await streamQuestion(searchQuery.value, {
          mode: searchMode.value === 'quick' ? 'quick' : 'agentic',
          onStart: () => {},
          onLog: (log) => {
            agentLogs.value.push(log)
          },
          onStatus: (message) => {
            agentLogs.value.push({
              step: '📍',
              content: message,
              timestamp: new Date().toLocaleTimeString()
            })
          },
          onChunk: (content) => {
            streamingAnswer.value += content
            chatResult.value = {
              ...chatResult.value,
              answer: streamingAnswer.value,
              status: 'streaming'
            }
          },
          onDone: (data) => {
            chatResult.value = {
              answer: streamingAnswer.value,
              sources: [],
              steps: data.steps || 0,
              status: 'done'
            }
            isStreaming.value = false
            logsExpanded.value = false  // 完成后自动折叠日志
          },
          onError: (error) => {
            console.error('Stream error:', error)
            agentLogs.value.push({
              step: '❌ 错误',
              content: error.message,
              timestamp: new Date().toLocaleTimeString()
            })
            chatResult.value = {
              answer: `出错: ${error.message}`,
              sources: [],
              status: 'error'
            }
            isStreaming.value = false
          }
        })
      } catch (error) {
        console.error('Search failed:', error)
        chatResult.value = {
          answer: `搜索出错: ${error.message}`,
          sources: [],
          status: 'error'
        }
        isStreaming.value = false
      }
    }

    const handleNodeClick = (node) => {
      selectedNode.value = node
    }

    const askAboutNode = (node) => {
      searchQuery.value = `关于"${node.name}"的相关新闻和信息有哪些？`
      selectedNode.value = null
      handleSearch()
    }

    const closeResultPanel = () => {
      showResultPanel.value = false
    }

    const formatAnswer = (text) => {
      if (!text) return ''
      return text
        .replace(/\n/g, '<br>')
        .replace(/【洞察】/g, '<strong class="insight-marker">【洞察】</strong>')
    }

    const getModeLabel = (mode) => {
      const labels = {
        'quick': '快速搜索',
        'panorama': '广度搜索',
        'insight': '深度分析'
      }
      return labels[mode] || mode
    }

    const loadSuggestions = async () => {
      try {
        const res = await getSuggestions()
        if (res.data?.suggestions) {
          suggestions.value = res.data.suggestions.slice(0, 3)
        }
      } catch (e) {
        // Use default suggestions
      }
    }

    onMounted(() => {
      refreshGraph()
      loadSuggestions()
    })

    return {
      nodes,
      edges,
      stats,
      loading,
      selectedNode,
      searchQuery,
      searchMode,
      showResultPanel,
      chatResult,
      lastQuery,
      suggestions,
      isStreaming,
      agentLogs,
      logsExpanded,
      // Date filtering
      selectedDateRange,
      customDate,
      dateOptions,
      filteredNodes,
      filteredEdges,
      // Methods
      refreshGraph,
      handleSearch,
      handleNodeClick,
      askAboutNode,
      closeResultPanel,
      formatAnswer,
      getModeLabel
    }
  }
}
</script>

<style>
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.app {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  position: relative;
}

/* Full-screen Graph */
.graph-fullscreen {
  width: 100%;
  height: 100%;
}

/* Date Filter */
.date-filter {
  position: fixed;
  top: 16px;
  left: 16px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 12px 16px;
  border-radius: 16px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  z-index: 100;
  min-width: 200px;
}

.filter-label {
  font-size: 11px;
  font-weight: 600;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
}

.filter-options {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.filter-btn {
  padding: 6px 12px;
  border: 1px solid #e0e0e0;
  background: white;
  border-radius: 16px;
  font-size: 12px;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}

.filter-btn:hover {
  border-color: #667eea;
  color: #667eea;
}

.filter-btn.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-color: transparent;
}

.filter-custom {
  margin-top: 10px;
}

.filter-custom input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  font-size: 13px;
  outline: none;
}

.filter-custom input:focus {
  border-color: #667eea;
}

.filter-stats {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid #eee;
  font-size: 12px;
  color: #888;
}

/* Stats Badge */
.stats-badge {
  position: fixed;
  top: 16px;
  right: 16px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 10px 16px;
  border-radius: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #333;
  z-index: 100;
}

.stats-badge .divider {
  color: #ddd;
}

.refresh-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  padding: 4px;
  border-radius: 50%;
  transition: background 0.2s;
}

.refresh-btn:hover:not(:disabled) {
  background: #f0f0f0;
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Bottom Search Bar */
.search-container {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 100;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.search-box {
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.98);
  backdrop-filter: blur(10px);
  border-radius: 28px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
  padding: 6px;
  border: 1px solid rgba(0, 0, 0, 0.08);
}

.search-box input {
  width: 400px;
  padding: 12px 20px;
  border: none;
  outline: none;
  font-size: 15px;
  background: transparent;
}

.search-box input::placeholder {
  color: #999;
}

.mode-select {
  padding: 8px 12px;
  border: none;
  background: #f5f5f5;
  border-radius: 16px;
  font-size: 13px;
  cursor: pointer;
  outline: none;
}

.search-btn {
  padding: 12px 24px;
  margin-left: 8px;
  border: none;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 22px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
}

.search-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.search-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.quick-suggestions {
  display: flex;
  gap: 8px;
}

.suggestion-chip {
  padding: 6px 14px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  font-size: 12px;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.suggestion-chip:hover {
  background: white;
  color: #333;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

/* Left Result Panel - Floating Card Style */
.result-panel {
  position: fixed;
  left: 20px;
  top: 20px;
  bottom: 20px;
  width: 400px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border-radius: 20px;
  box-shadow:
    0 8px 32px rgba(0, 0, 0, 0.12),
    0 0 0 1px rgba(255, 255, 255, 0.5) inset;
  z-index: 200;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.slide-enter-active,
.slide-leave-active {
  transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.slide-enter-from,
.slide-leave-to {
  transform: translateX(-120%) scale(0.95);
  opacity: 0;
}

.panel-header {
  padding: 20px 24px;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
  border-bottom: 1px solid rgba(102, 126, 234, 0.15);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.panel-content {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.panel-content::-webkit-scrollbar {
  width: 6px;
}

.panel-content::-webkit-scrollbar-track {
  background: transparent;
}

.panel-content::-webkit-scrollbar-thumb {
  background: rgba(102, 126, 234, 0.3);
  border-radius: 3px;
}

.panel-content::-webkit-scrollbar-thumb:hover {
  background: rgba(102, 126, 234, 0.5);
}

.question-block {
  margin-bottom: 16px;
  padding: 14px;
  background: rgba(102, 126, 234, 0.06);
  border-radius: 14px;
  border-left: 3px solid #667eea;
}

.question-label,
.answer-label,
.insights-label,
.sources-label {
  font-size: 11px;
  font-weight: 600;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 6px;
}

.question-text {
  font-size: 15px;
  color: #333;
  font-weight: 500;
  line-height: 1.5;
}

.answer-block {
  margin-bottom: 16px;
  padding: 16px;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  border-radius: 14px;
  border: 1px solid rgba(102, 126, 234, 0.1);
}

.answer-label {
  margin-bottom: 8px;
}

.answer-text {
  font-size: 14px;
  line-height: 1.7;
  color: #333;
}

.answer-text :deep(.insight-marker) {
  color: #667eea;
}

.insights-block {
  margin-bottom: 16px;
  padding: 14px;
  background: linear-gradient(135deg, rgba(168, 85, 247, 0.08) 0%, rgba(139, 92, 246, 0.12) 100%);
  border-radius: 14px;
  border: 1px solid rgba(139, 92, 246, 0.15);
}

.insights-list {
  margin: 0;
  padding-left: 18px;
  font-size: 13px;
  color: #6d28d9;
}

.insights-list li {
  margin-bottom: 4px;
}

.sources-block {
  margin-bottom: 16px;
}

.sources-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.source-item {
  padding: 10px 12px;
  background: rgba(0, 0, 0, 0.03);
  border-radius: 10px;
  font-size: 12px;
  color: #555;
  line-height: 1.5;
  border: 1px solid rgba(0, 0, 0, 0.04);
}

.meta-block {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: #888;
  padding-top: 14px;
  border-top: 1px solid rgba(0, 0, 0, 0.06);
}

/* Loading Animation */
.loading-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
}

.loading-dots {
  display: flex;
  gap: 6px;
}

.loading-dots span {
  width: 10px;
  height: 10px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 50%;
  animation: bounce 1.4s ease-in-out infinite;
}

.loading-dots span:nth-child(1) { animation-delay: 0s; }
.loading-dots span:nth-child(2) { animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 80%, 100% {
    transform: scale(0.6);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

.loading-text {
  margin-top: 16px;
  font-size: 13px;
  color: #888;
}

/* Agent Logs */
.agent-logs {
  margin-bottom: 16px;
  background: #1a1a2e;
  border-radius: 12px;
  overflow: hidden;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
}

.logs-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 14px;
  background: #16213e;
  cursor: pointer;
  user-select: none;
}

.logs-title {
  font-size: 12px;
  font-weight: 600;
  color: #4ade80;
}

.logs-count {
  font-weight: normal;
  color: #666;
  margin-left: 8px;
}

.logs-toggle {
  font-size: 11px;
  color: #888;
  transition: color 0.2s;
}

.logs-header:hover .logs-toggle {
  color: #4ade80;
}

.logs-content {
  max-height: 300px;
  overflow-y: auto;
  padding: 8px;
}

.logs-content::-webkit-scrollbar {
  width: 6px;
}

.logs-content::-webkit-scrollbar-thumb {
  background: #333;
  border-radius: 3px;
}

.log-item {
  padding: 8px 10px;
  border-radius: 6px;
  margin-bottom: 6px;
  background: rgba(255,255,255,0.03);
}

.log-item.loading {
  display: flex;
  justify-content: center;
  padding: 12px;
}

.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}

.log-step {
  font-size: 12px;
  font-weight: 600;
  color: #60a5fa;
}

.log-time {
  font-size: 10px;
  color: #555;
}

.log-content {
  font-size: 12px;
  color: #e2e8f0;
  line-height: 1.5;
}

.log-data {
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px solid rgba(255,255,255,0.05);
}

.log-tag {
  display: inline-block;
  padding: 2px 8px;
  background: #7c3aed;
  color: white;
  border-radius: 4px;
  font-size: 10px;
  margin-right: 8px;
}

.log-args {
  font-size: 11px;
  color: #a78bfa;
}

.log-preview {
  margin-top: 6px;
}

.preview-item {
  font-size: 11px;
  color: #94a3b8;
  padding: 2px 0;
}

.log-plan {
  margin-top: 4px;
}

.plan-item {
  font-size: 11px;
  color: #fbbf24;
  padding: 2px 0;
}

/* Thinking inline */
.thinking-inline {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #888;
  font-size: 14px;
}

.thinking-dots {
  display: inline-block;
  width: 20px;
  height: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 50%;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(0.8); opacity: 0.5; }
  50% { transform: scale(1); opacity: 1; }
}

/* Streaming cursor */
.streaming-cursor {
  display: inline-block;
  color: #667eea;
  animation: blink 1s step-end infinite;
  margin-left: 2px;
  font-weight: bold;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

/* Close Button */
.close-btn {
  width: 32px;
  height: 32px;
  border: none;
  background: rgba(102, 126, 234, 0.1);
  border-radius: 10px;
  font-size: 18px;
  color: #667eea;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.close-btn:hover {
  background: rgba(102, 126, 234, 0.2);
  transform: scale(1.05);
}

/* Node Modal */
.node-modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 300;
}

.modal-content {
  background: white;
  border-radius: 16px;
  padding: 28px;
  max-width: 400px;
  width: 90%;
  position: relative;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
}

.modal-content h3 {
  margin: 0 0 12px 0;
  font-size: 20px;
}

.node-labels {
  display: flex;
  gap: 6px;
  margin-bottom: 16px;
}

.label-tag {
  padding: 4px 12px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
}

.node-summary {
  font-size: 14px;
  color: #666;
  line-height: 1.6;
  margin-bottom: 20px;
}

.ask-btn {
  width: 100%;
  padding: 12px;
  border: none;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 10px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: transform 0.2s;
}

.ask-btn:hover {
  transform: translateY(-1px);
}
</style>
