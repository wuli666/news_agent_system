/**
 * GraphRAG API Client
 */

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5002'

const service = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for long operations
  headers: {
    'Content-Type': 'application/json'
  }
})

// Response interceptor
service.interceptors.response.use(
  response => {
    const res = response.data
    if (res.success === false) {
      return Promise.reject(new Error(res.error || 'Unknown error'))
    }
    return res
  },
  error => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

// Retry helper
export async function requestWithRetry(requestFn, maxRetries = 3, delay = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await requestFn()
    } catch (error) {
      if (i === maxRetries - 1) throw error
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)))
    }
  }
}

// ============================================================
// Graph API
// ============================================================

export function getGraphData(graphId = null) {
  const params = graphId ? { graph_id: graphId } : {}
  return service.get('/api/graph/data', { params })
}

export function getEntity(entityId, graphId = null) {
  const params = graphId ? { graph_id: graphId } : {}
  return service.get(`/api/graph/entity/${entityId}`, { params })
}

export function getTimeline(graphId = null, limit = 50) {
  return service.get('/api/graph/timeline', {
    params: { graph_id: graphId, limit }
  })
}

export function searchGraph(query, graphId = null, limit = 10) {
  return service.get('/api/graph/search', {
    params: { q: query, graph_id: graphId, limit }
  })
}

export function getGraphStats(graphId = null) {
  const params = graphId ? { graph_id: graphId } : {}
  return service.get('/api/graph/stats', { params })
}

export function ingestNews(newsItems, options = {}) {
  return service.post('/api/graph/ingest', {
    news: newsItems,
    graph_id: options.graphId,
    extract: options.extract !== false,
    async: options.async || false
  })
}

export function getTaskStatus(taskId) {
  return service.get(`/api/graph/task/${taskId}`)
}

// ============================================================
// Chat API
// ============================================================

export function sendQuestion(question, options = {}) {
  return service.post('/api/chat', {
    question,
    session_id: options.sessionId || 'default',
    graph_id: options.graphId,
    mode: options.mode || 'quick'
  })
}

/**
 * Stream chat response using SSE
 * @param {string} question - User question
 * @param {object} options - Options { mode, graphId, onChunk, onDone, onError }
 */
export async function streamQuestion(question, options = {}) {
  const { mode = 'agentic', graphId, onChunk, onDone, onError, onStart, onStatus, onPlan, onTool } = options

  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        graph_id: graphId,
        mode
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            switch (data.type) {
              case 'start':
                if (onStart) onStart()
                break
              case 'log':
                if (options.onLog) options.onLog(data)
                break
              case 'status':
                if (onStatus) onStatus(data.message)
                break
              case 'plan':
                if (onPlan) onPlan(data.data, data.message)
                break
              case 'tool':
                if (onTool) onTool(data.data)
                break
              case 'chunk':
                if (onChunk) onChunk(data.content)
                break
              case 'done':
                if (onDone) onDone(data.data || data)
                break
              case 'error':
                if (onError) onError(new Error(data.content))
                break
            }
          } catch (e) {
            console.warn('Failed to parse SSE data:', line)
          }
        }
      }
    }
  } catch (error) {
    if (onError) onError(error)
    throw error
  }
}

export function getChatHistory(sessionId = 'default', limit = 50) {
  return service.get('/api/chat/history', {
    params: { session_id: sessionId, limit }
  })
}

export function clearChatHistory(sessionId = 'default') {
  return service.delete('/api/chat/history', {
    params: { session_id: sessionId }
  })
}

export function getSuggestions(graphId = null) {
  const params = graphId ? { graph_id: graphId } : {}
  return service.get('/api/chat/suggest', { params })
}

// ============================================================
// Health Check
// ============================================================

export function healthCheck() {
  return service.get('/health')
}

export default {
  getGraphData,
  getEntity,
  getTimeline,
  searchGraph,
  getGraphStats,
  ingestNews,
  getTaskStatus,
  sendQuestion,
  streamQuestion,
  getChatHistory,
  clearChatHistory,
  getSuggestions,
  healthCheck
}
