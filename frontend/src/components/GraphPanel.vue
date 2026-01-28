<template>
  <div class="graph-container" ref="container">
    <svg ref="svg" class="graph-svg"></svg>
    <div class="graph-controls">
      <button @click="zoomIn" title="放大">+</button>
      <button @click="zoomOut" title="缩小">−</button>
      <button @click="resetZoom" title="重置">⟲</button>
    </div>
    <div class="no-data" v-if="!nodes.length">
      <div class="no-data-icon">📊</div>
      <p class="no-data-title">暂无图谱数据</p>
      <p class="no-data-hint">请先运行新闻收集任务来构建知识图谱</p>
    </div>
  </div>
</template>

<script>
import { ref, watch, onMounted, onUnmounted } from 'vue'
import * as d3 from 'd3'

export default {
  name: 'GraphPanel',
  props: {
    nodes: {
      type: Array,
      default: () => []
    },
    edges: {
      type: Array,
      default: () => []
    }
  },
  emits: ['node-click'],
  setup(props, { emit }) {
    const container = ref(null)
    const svg = ref(null)

    let simulation = null
    let svgSelection = null
    let g = null
    let zoom = null

    // Color palette for clusters
    const clusterColors = [
      '#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a',
      '#fee140', '#30cfd0', '#a8edea', '#ff9a9e', '#fbc2eb',
      '#a18cd1', '#fad0c4', '#ffecd2', '#fcb69f', '#ff8177',
      '#b721ff', '#21d4fd', '#08aeea', '#2af598', '#ff6b6b'
    ]

    const nodeClusterMap = ref(new Map())

    // Simple clustering based on connected components
    const computeClusters = (nodes, edges) => {
      const parent = new Map()
      nodes.forEach(n => parent.set(n.uuid, n.uuid))

      const find = (x) => {
        if (parent.get(x) !== x) {
          parent.set(x, find(parent.get(x)))
        }
        return parent.get(x)
      }

      const union = (a, b) => {
        const pa = find(a), pb = find(b)
        if (pa !== pb) parent.set(pa, pb)
      }

      edges.forEach(e => {
        if (parent.has(e.source) && parent.has(e.target)) {
          union(e.source, e.target)
        }
      })

      // Assign cluster IDs
      const clusterIds = new Map()
      let clusterId = 0
      nodes.forEach(n => {
        const root = find(n.uuid)
        if (!clusterIds.has(root)) {
          clusterIds.set(root, clusterId++)
        }
        nodeClusterMap.value.set(n.uuid, clusterIds.get(root))
      })

      return clusterId // number of clusters
    }

    const getNodeColor = (node) => {
      const clusterId = nodeClusterMap.value.get(node.uuid) || 0
      return clusterColors[clusterId % clusterColors.length]
    }

    const initGraph = () => {
      if (!svg.value || !container.value) return

      const width = container.value.clientWidth
      const height = container.value.clientHeight || window.innerHeight

      // Clear previous
      d3.select(svg.value).selectAll('*').remove()

      // Setup SVG
      svgSelection = d3.select(svg.value)
        .attr('width', width)
        .attr('height', height)

      // Add zoom behavior
      zoom = d3.zoom()
        .scaleExtent([0.1, 4])
        .on('zoom', (event) => {
          g.attr('transform', event.transform)
        })

      svgSelection.call(zoom)

      // Main group for zoom/pan
      g = svgSelection.append('g')

      // Arrow marker for edges
      svgSelection.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#bbb')
    }

    const updateGraph = () => {
      if (!g || !props.nodes.length) return

      const width = container.value.clientWidth
      const height = container.value.clientHeight || window.innerHeight

      // Prepare data
      const nodeMap = new Map(props.nodes.map(n => [n.uuid, n]))

      const links = props.edges
        .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
        .map(e => ({
          ...e,
          source: e.source,
          target: e.target
        }))

      const nodes = props.nodes.map(n => ({ ...n, id: n.uuid }))

      // Compute clusters for coloring
      computeClusters(props.nodes, props.edges)

      // Create simulation
      simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.uuid).distance(120))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(35))

      // Clear and redraw
      g.selectAll('*').remove()

      // Draw edges
      const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('stroke', '#ddd')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', 1.5)
        .attr('marker-end', 'url(#arrowhead)')


      // Draw nodes
      const node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .call(d3.drag()
          .on('start', dragstarted)
          .on('drag', dragged)
          .on('end', dragended))
        .on('click', (event, d) => {
          event.stopPropagation()
          emit('node-click', d)
        })

      // Node circles with gradient
      node.append('circle')
        .attr('r', 14)
        .attr('fill', d => getNodeColor(d))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2.5)
        .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))')

      // Node labels
      node.append('text')
        .attr('dx', 18)
        .attr('dy', 4)
        .attr('font-size', 12)
        .attr('font-weight', 500)
        .attr('fill', '#333')
        .text(d => {
          const name = d.name || ''
          return name.length > 8 ? name.slice(0, 8) + '...' : name
        })

      // Simulation tick
      simulation.on('tick', () => {
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y)

        node
          .attr('transform', d => `translate(${d.x},${d.y})`)
      })

      // Drag functions
      function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart()
        d.fx = d.x
        d.fy = d.y
      }

      function dragged(event, d) {
        d.fx = event.x
        d.fy = event.y
      }

      function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0)
        d.fx = null
        d.fy = null
      }
    }

    const zoomIn = () => {
      svgSelection.transition().call(zoom.scaleBy, 1.3)
    }

    const zoomOut = () => {
      svgSelection.transition().call(zoom.scaleBy, 0.7)
    }

    const resetZoom = () => {
      svgSelection.transition().call(zoom.transform, d3.zoomIdentity)
    }

    // Watch for data changes
    watch(() => [props.nodes, props.edges], () => {
      updateGraph()
    }, { deep: true })

    // Resize handler
    const handleResize = () => {
      initGraph()
      updateGraph()
    }

    onMounted(() => {
      initGraph()
      updateGraph()
      window.addEventListener('resize', handleResize)
    })

    onUnmounted(() => {
      if (simulation) simulation.stop()
      window.removeEventListener('resize', handleResize)
    })

    return {
      container,
      svg,
      zoomIn,
      zoomOut,
      resetZoom
    }
  }
}
</script>

<style scoped>
.graph-container {
  width: 100%;
  height: 100%;
  position: relative;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
}

.graph-svg {
  width: 100%;
  height: 100%;
}

.graph-controls {
  position: absolute;
  top: 16px;
  left: 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  z-index: 10;
}

.graph-controls button {
  width: 36px;
  height: 36px;
  border: none;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.95);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  cursor: pointer;
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.graph-controls button:hover {
  background: white;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transform: translateY(-1px);
}


.no-data {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  color: #666;
}

.no-data-icon {
  font-size: 64px;
  margin-bottom: 16px;
  opacity: 0.5;
}

.no-data-title {
  font-size: 20px;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #333;
}

.no-data-hint {
  font-size: 14px;
  margin: 0;
  color: #888;
}

:deep(.node) {
  cursor: pointer;
}

:deep(.node:hover circle) {
  stroke: #667eea;
  stroke-width: 3;
  filter: drop-shadow(0 4px 8px rgba(102, 126, 234, 0.4));
}
</style>
