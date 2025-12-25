/**
 * Knowledge Graph Explorer - D3.js Force-Directed Graph
 * 
 * Event-centric visualization with:
 * - Labels on all nodes
 * - Tooltips on hover
 * - Variable link thickness (by strength)
 * - Variable node size (by importance)
 * - Color coding by type
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import * as d3 from 'd3'
import { useStore } from '../store'
import { 
  Network, 
  Maximize2, 
  RefreshCw,
  Filter,
  Calendar,
  MapPin,
  FileText,
  Zap,
  AlertTriangle,
  Info,
  X
} from 'lucide-react'

// Types matching backend GraphNode/GraphLink
interface GraphNode {
  id: string
  type: 'paper' | 'event' | 'pattern' | 'data_source' | 'climate_index'
  label: string
  date?: string
  lat?: number
  lon?: number
  confidence?: number
  cluster?: string
  metadata?: Record<string, any>
  // D3 simulation properties
  x?: number
  y?: number
  fx?: number | null
  fy?: number | null
}

interface GraphLink {
  source: string | GraphNode
  target: string | GraphNode
  type: string
  strength: number
  label?: string
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
  stats: {
    total_nodes: number
    papers: number
    events: number
    patterns: number
    total_links: number
  }
}

// Color scheme by node type
const NODE_COLORS: Record<string, string> = {
  event: '#ef4444',       // red-500 - events are central
  paper: '#3b82f6',       // blue-500 - research papers
  pattern: '#10b981',     // emerald-500 - causal patterns
  data_source: '#8b5cf6', // violet-500
  climate_index: '#f59e0b', // amber-500
}

// Node size by type (events largest, papers smallest)
const NODE_SIZES: Record<string, number> = {
  event: 20,
  pattern: 14,
  paper: 10,
  data_source: 12,
  climate_index: 12,
}

const NODE_ICONS: Record<string, any> = {
  event: AlertTriangle,
  paper: FileText,
  pattern: Zap,
  data_source: Network,
  climate_index: Calendar,
}

// Link color by strength
function getLinkColor(strength: number): string {
  if (strength >= 0.7) return '#059669' // emerald-600 - strong
  if (strength >= 0.5) return '#3b82f6' // blue-500 - moderate  
  if (strength >= 0.3) return '#94a3b8' // slate-400 - weak
  return '#cbd5e1' // slate-300 - very weak
}

function getLinkMarker(strength: number): string {
  if (strength >= 0.7) return 'url(#kg-arrow-strong)'
  if (strength >= 0.5) return 'url(#kg-arrow-moderate)'
  return 'url(#kg-arrow-weak)'
}

export function KnowledgeGraphView() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { backend } = useStore()
  
  // State
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [hoveredLink, setHoveredLink] = useState<GraphLink | null>(null)
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState({
    showPapers: true,
    showEvents: true,
    showPatterns: true,
  })

  // Fetch graph data
  const fetchGraphData = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const params = new URLSearchParams({
        backend,
        include_papers: String(filters.showPapers),
        include_events: String(filters.showEvents),
        include_patterns: String(filters.showPatterns),
        limit_papers: '50',
        limit_events: '20',
      })
      
      const res = await fetch(`/api/v1/knowledge/graph?${params}`)
      if (!res.ok) throw new Error(`Failed to fetch graph: ${res.statusText}`)
      
      const data: GraphData = await res.json()
      setGraphData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [backend, filters])

  // Load on mount and when filters change
  useEffect(() => {
    fetchGraphData()
  }, [fetchGraphData])

  // D3 visualization
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !graphData || graphData.nodes.length === 0) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Clear previous
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])

    // Zoom behavior
    const g = svg.append('g')
    
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })
    
    svg.call(zoom)

    // Arrow markers for links
    const defs = svg.append('defs')
    
    defs.selectAll('marker')
      .data(['strong', 'moderate', 'weak'])
      .join('marker')
      .attr('id', d => `kg-arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('fill', d => {
        if (d === 'strong') return '#059669'
        if (d === 'moderate') return '#3b82f6'
        return '#94a3b8'
      })
      .attr('d', 'M0,-5L10,0L0,5')

    // Prepare data for simulation
    const nodes = graphData.nodes.map(d => ({ ...d }))
    const links = graphData.links.map(d => ({ ...d }))

    // Force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links)
        .id((d: any) => d.id)
        .distance(120))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => (NODE_SIZES[d.type] || 10) + 15))

    // Draw links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', d => getLinkColor(d.strength))
      .attr('stroke-width', d => Math.max(1.5, d.strength * 5))
      .attr('stroke-opacity', 0.7)
      .attr('marker-end', d => getLinkMarker(d.strength))
      .on('mouseover', (_event, d) => setHoveredLink(d))
      .on('mouseout', () => setHoveredLink(null))
      .style('cursor', 'pointer')

    // Link labels (type/strength)
    const linkLabels = g.append('g')
      .attr('class', 'link-labels')
      .selectAll('text')
      .data(links)
      .join('text')
      .attr('font-size', '9px')
      .attr('fill', '#64748b')
      .attr('text-anchor', 'middle')
      .attr('dy', -4)
      .text(d => d.label || d.type.replace(/_/g, ' '))

    // Draw nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag<any, GraphNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))
      .on('click', (_event, d) => setSelectedNode(d))
      .on('mouseover', (_event, d) => setHoveredNode(d))
      .on('mouseout', () => setHoveredNode(null))

    // Node glow/shadow for events
    node.filter(d => d.type === 'event')
      .append('circle')
      .attr('r', d => (NODE_SIZES[d.type] || 10) + 4)
      .attr('fill', 'none')
      .attr('stroke', NODE_COLORS.event)
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.3)

    // Node circles with variable size
    node.append('circle')
      .attr('r', d => NODE_SIZES[d.type] || 10)
      .attr('fill', d => NODE_COLORS[d.type] || '#94a3b8')
      .attr('stroke', '#0f172a')
      .attr('stroke-width', 2)
      .attr('opacity', 0.9)

    // Node labels
    node.append('text')
      .text(d => {
        const label = d.label || d.id
        return label.length > 25 ? label.substring(0, 22) + '...' : label
      })
      .attr('dy', d => -(NODE_SIZES[d.type] || 10) - 8)
      .attr('text-anchor', 'middle')
      .attr('font-size', d => d.type === 'event' ? '12px' : '10px')
      .attr('font-weight', d => d.type === 'event' ? '600' : '400')
      .attr('fill', '#1e293b')
      .attr('paint-order', 'stroke')
      .attr('stroke', '#ffffff')
      .attr('stroke-width', 3)

    // Type emoji on nodes
    node.append('text')
      .text(d => {
        if (d.type === 'event') return '⚠'
        if (d.type === 'pattern') return '⚡'
        if (d.type === 'paper') return '📄'
        return ''
      })
      .attr('dy', 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', d => d.type === 'paper' ? '8px' : '10px')

    // Simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)

      linkLabels
        .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
        .attr('y', (d: any) => (d.source.y + d.target.y) / 2)

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`)
    })

    // Drag functions
    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      event.subject.fx = event.subject.x
      event.subject.fy = event.subject.y
    }

    function dragged(event: any) {
      event.subject.fx = event.x
      event.subject.fy = event.y
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0)
      event.subject.fx = null
      event.subject.fy = null
    }

    // Cleanup
    return () => {
      simulation.stop()
    }
  }, [graphData])

  // Node details panel
  const NodeDetails = ({ node }: { node: GraphNode }) => {
    const Icon = NODE_ICONS[node.type] || Info
    
    return (
      <div className="absolute right-4 top-4 w-80 bg-white/95 backdrop-blur border border-slate-200 rounded-lg shadow-xl overflow-hidden z-10">
        {/* Header */}
        <div 
          className="p-4 border-b border-slate-200"
          style={{ backgroundColor: NODE_COLORS[node.type] + '15' }}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Icon size={18} style={{ color: NODE_COLORS[node.type] }} />
              <span className="text-xs uppercase tracking-wider text-slate-500">
                {node.type}
              </span>
            </div>
            <button 
              onClick={() => setSelectedNode(null)}
              className="p-1 hover:bg-slate-100 rounded"
            >
              <X size={16} />
            </button>
          </div>
          <h3 className="font-semibold mt-2 text-slate-900">{node.label}</h3>
        </div>
        
        {/* Content */}
        <div className="p-4 space-y-3 text-sm">
          {node.date && (
            <div className="flex items-center gap-2 text-slate-600">
              <Calendar size={14} />
              <span>{node.date}</span>
            </div>
          )}
          
          {(node.lat && node.lon) && (
            <div className="flex items-center gap-2 text-slate-600">
              <MapPin size={14} />
              <span>{node.lat.toFixed(2)}°N, {node.lon.toFixed(2)}°E</span>
            </div>
          )}
          
          {node.confidence && (
            <div className="flex items-center gap-2">
              <span className="font-medium text-slate-600">Confidence:</span>
              <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div 
                  className="h-full rounded-full" 
                  style={{ 
                    width: `${node.confidence * 100}%`,
                    backgroundColor: NODE_COLORS[node.type]
                  }}
                />
              </div>
              <span className="text-xs text-slate-500">{(node.confidence * 100).toFixed(0)}%</span>
            </div>
          )}
          
          {node.cluster && (
            <div className="flex items-center gap-2">
              <span className="font-medium text-slate-600">Cluster:</span>
              <span className="px-2 py-0.5 bg-slate-100 rounded text-xs">{node.cluster}</span>
            </div>
          )}
          
          {/* Metadata */}
          {node.metadata && Object.keys(node.metadata).length > 0 && (
            <div className="pt-2 border-t border-slate-100">
              {node.metadata.description && (
                <p className="text-slate-600 text-xs line-clamp-3">
                  {node.metadata.description}
                </p>
              )}
              {node.metadata.authors && node.metadata.authors.length > 0 && (
                <p className="text-xs text-slate-500 mt-1">
                  <span className="font-medium">Authors:</span>{' '}
                  {node.metadata.authors.slice(0, 3).join(', ')}
                  {node.metadata.authors.length > 3 && ' et al.'}
                </p>
              )}
              {node.metadata.abstract && (
                <p className="text-xs text-slate-500 mt-1 line-clamp-2">
                  {node.metadata.abstract}
                </p>
              )}
              {node.metadata.doi && (
                <a 
                  href={`https://doi.org/${node.metadata.doi}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-blue-600 hover:underline mt-1 block"
                >
                  DOI: {node.metadata.doi}
                </a>
              )}
            </div>
          )}
        </div>
      </div>
    )
  }
  
  // Tooltip for hovered elements
  const Tooltip = () => {
    if (hoveredLink) {
      return (
        <div className="absolute left-4 bottom-4 bg-slate-900 text-white px-3 py-2 rounded-lg shadow-lg text-sm z-10 max-w-xs">
          <div className="font-medium">{hoveredLink.type.replace(/_/g, ' ')}</div>
          <div className="text-slate-300 text-xs mt-1">
            Strength: {(hoveredLink.strength * 100).toFixed(0)}%
            {hoveredLink.label && ` • ${hoveredLink.label}`}
          </div>
        </div>
      )
    }
    
    if (hoveredNode && !selectedNode) {
      return (
        <div className="absolute left-4 bottom-4 bg-slate-900 text-white px-3 py-2 rounded-lg shadow-lg text-sm z-10 max-w-xs">
          <div className="font-medium">{hoveredNode.label}</div>
          <div className="text-slate-300 text-xs mt-1">
            {hoveredNode.type} 
            {hoveredNode.cluster && ` • ${hoveredNode.cluster}`}
            {hoveredNode.confidence && ` • ${(hoveredNode.confidence * 100).toFixed(0)}% confidence`}
          </div>
          {hoveredNode.date && (
            <div className="text-slate-400 text-xs">{hoveredNode.date}</div>
          )}
        </div>
      )
    }
    
    return null
  }

  return (
    <div className="h-full flex flex-col bg-slate-50 rounded-lg border border-slate-200 overflow-hidden">
      {/* Header */}
      <div className="p-4 bg-white border-b border-slate-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Network className="text-blue-600" size={20} />
          <h2 className="font-semibold text-slate-900">Knowledge Graph</h2>
          {graphData && (
            <span className="text-sm text-slate-500">
              {graphData.stats.total_nodes} nodes • {graphData.stats.total_links} links
            </span>
          )}
        </div>
        
        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`p-2 rounded transition-colors ${
              showFilters ? 'bg-blue-100 text-blue-600' : 'hover:bg-slate-100'
            }`}
            title="Toggle filters"
          >
            <Filter size={16} />
          </button>
          <button 
            onClick={fetchGraphData} 
            className="p-2 hover:bg-slate-100 rounded transition-colors"
            title="Refresh"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          </button>
          <button 
            className="p-2 hover:bg-slate-100 rounded transition-colors"
            title="Fullscreen"
          >
            <Maximize2 size={16} />
          </button>
        </div>
      </div>
      
      {/* Filters */}
      {showFilters && (
        <div className="p-3 bg-slate-50 border-b border-slate-200 flex items-center gap-6">
          <span className="text-sm text-slate-600 font-medium">Show:</span>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showEvents}
              onChange={e => setFilters({ ...filters, showEvents: e.target.checked })}
              className="rounded border-slate-300"
            />
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS.event }} />
            <span className="text-sm">Events ({graphData?.stats.events || 0})</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showPapers}
              onChange={e => setFilters({ ...filters, showPapers: e.target.checked })}
              className="rounded border-slate-300"
            />
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS.paper }} />
            <span className="text-sm">Papers ({graphData?.stats.papers || 0})</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showPatterns}
              onChange={e => setFilters({ ...filters, showPatterns: e.target.checked })}
              className="rounded border-slate-300"
            />
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS.pattern }} />
            <span className="text-sm">Patterns ({graphData?.stats.patterns || 0})</span>
          </label>
        </div>
      )}
      
      {/* Main Graph Area */}
      <div ref={containerRef} className="flex-1 relative min-h-[400px]">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white/80 z-20">
            <div className="flex items-center gap-3">
              <RefreshCw className="animate-spin text-blue-600" size={24} />
              <span className="text-slate-600">Loading knowledge graph...</span>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center p-6">
              <AlertTriangle className="mx-auto text-red-500 mb-3" size={32} />
              <p className="text-red-600 font-medium">{error}</p>
              <button
                onClick={fetchGraphData}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Retry
              </button>
            </div>
          </div>
        )}
        
        {!loading && !error && graphData && graphData.nodes.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center p-6">
              <Network className="mx-auto text-slate-300 mb-3" size={48} />
              <p className="text-slate-500 font-medium">No data in knowledge graph</p>
              <p className="text-slate-400 text-sm mt-1">
                Run an investigation to populate the knowledge base
              </p>
            </div>
          </div>
        )}
        
        <svg ref={svgRef} className="w-full h-full" />
        
        {/* Selected Node Details */}
        {selectedNode && <NodeDetails node={selectedNode} />}
        
        {/* Hover Tooltip */}
        <Tooltip />
        
        {/* Legend */}
        <div className="absolute left-4 top-4 bg-white/95 backdrop-blur border border-slate-200 rounded-lg p-3 shadow-sm">
          <div className="text-xs text-slate-500 font-medium mb-2">Legend</div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-sm">
              <div className="w-5 h-5 rounded-full border-2 border-slate-800" style={{ backgroundColor: NODE_COLORS.event }} />
              <span>Events</span>
              <span className="text-slate-400 text-xs">({graphData?.stats.events || 0})</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-3 h-3 rounded-full border-2 border-slate-800" style={{ backgroundColor: NODE_COLORS.paper }} />
              <span>Papers</span>
              <span className="text-slate-400 text-xs">({graphData?.stats.papers || 0})</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-4 h-4 rounded-full border-2 border-slate-800" style={{ backgroundColor: NODE_COLORS.pattern }} />
              <span>Patterns</span>
              <span className="text-slate-400 text-xs">({graphData?.stats.patterns || 0})</span>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-100">
            <div className="text-xs text-slate-500 font-medium mb-1">Link Strength</div>
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs">
                <div className="w-8 h-1 rounded" style={{ backgroundColor: '#059669' }} />
                <span>Strong (≥70%)</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-6 h-0.5 rounded" style={{ backgroundColor: '#3b82f6' }} />
                <span>Moderate (≥50%)</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-4 h-0.5 rounded" style={{ backgroundColor: '#94a3b8' }} />
                <span>Weak</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default KnowledgeGraphView
