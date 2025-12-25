/**
 * Knowledge Graph 3D Explorer
 * 
 * Professional 3D force-directed graph using react-force-graph-3d (ThreeJS)
 * 
 * Features:
 * - Full 3D navigation (orbit, zoom, pan)
 * - Labels on all nodes (always visible for events/patterns, hover for papers)
 * - Tooltips with detailed info
 * - Variable link thickness (by strength)
 * - Variable node sizes (by type/importance)
 * - Color coding by node type
 * - Directional particles on links
 * - Click to select, hover for preview
 */

import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
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
  X,
  Box,
  Eye,
  EyeOff
} from 'lucide-react'
import SpriteText from 'three-spritetext'

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
  // Force graph properties
  x?: number
  y?: number
  z?: number
  fx?: number
  fy?: number
  fz?: number
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
  event: 12,
  pattern: 8,
  paper: 5,
  data_source: 6,
  climate_index: 6,
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
  if (strength >= 0.3) return '#64748b' // slate-500 - weak
  return '#94a3b8' // slate-400 - very weak
}

// Link width by strength
function getLinkWidth(strength: number): number {
  return Math.max(0.5, strength * 3)
}

export function KnowledgeGraph3DView() {
  const containerRef = useRef<HTMLDivElement>(null)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null)
  const { backend } = useStore()
  
  // State
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [showFilters, setShowFilters] = useState(false)
  const [showLabels, setShowLabels] = useState(true)
  const [filters, setFilters] = useState({
    showPapers: true,
    showEvents: true,
    showPatterns: true,
  })
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

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

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        })
      }
    }
    
    handleResize()
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Custom node object with 3D sphere + label
  const nodeThreeObject = useCallback((node: GraphNode) => {
    const group = new THREE.Group()
    
    // Main sphere
    const size = NODE_SIZES[node.type] || 5
    const geometry = new THREE.SphereGeometry(size, 16, 16)
    const material = new THREE.MeshLambertMaterial({
      color: NODE_COLORS[node.type] || '#94a3b8',
      transparent: true,
      opacity: 0.9,
    })
    const sphere = new THREE.Mesh(geometry, material)
    group.add(sphere)
    
    // Glow effect for events (important nodes)
    if (node.type === 'event') {
      const glowGeometry = new THREE.SphereGeometry(size * 1.3, 16, 16)
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: NODE_COLORS.event,
        transparent: true,
        opacity: 0.2,
      })
      const glow = new THREE.Mesh(glowGeometry, glowMaterial)
      group.add(glow)
    }
    
    // Label (always visible for events/patterns, configurable for others)
    if (showLabels && (node.type === 'event' || node.type === 'pattern' || !filters.showPapers)) {
      const label = node.label || node.id
      const displayLabel = label.length > 20 ? label.substring(0, 18) + '...' : label
      
      const sprite = new SpriteText(displayLabel)
      sprite.color = '#1e293b'
      sprite.textHeight = node.type === 'event' ? 4 : 3
      sprite.backgroundColor = 'rgba(255,255,255,0.85)'
      sprite.padding = 1.5
      sprite.borderRadius = 2
      sprite.position.y = size + 6
      group.add(sprite)
    }
    
    return group
  }, [showLabels, filters.showPapers])

  // Node color accessor
  const nodeColor = useCallback((node: GraphNode) => {
    return NODE_COLORS[node.type] || '#94a3b8'
  }, [])

  // Node value (size) accessor
  const nodeVal = useCallback((node: GraphNode) => {
    return NODE_SIZES[node.type] || 5
  }, [])

  // Link color accessor
  const linkColor = useCallback((link: GraphLink) => {
    return getLinkColor(link.strength)
  }, [])

  // Link width accessor
  const linkWidth = useCallback((link: GraphLink) => {
    return getLinkWidth(link.strength)
  }, [])

  // Handle node click
  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNode(node)
    
    // Animate camera to focus on node
    if (fgRef.current && node.x !== undefined && node.y !== undefined && node.z !== undefined) {
      const distance = 150
      fgRef.current.cameraPosition(
        { x: node.x, y: node.y, z: node.z + distance },
        { x: node.x, y: node.y, z: node.z },
        1000
      )
    }
  }, [])

  // Handle node hover
  const handleNodeHover = useCallback((node: GraphNode | null) => {
    setHoveredNode(node)
    if (containerRef.current) {
      containerRef.current.style.cursor = node ? 'pointer' : 'default'
    }
  }, [])

  // Fit to view
  const handleFitView = useCallback(() => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(400, 50)
    }
  }, [])

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
  
  // Hover tooltip
  const Tooltip = ({ node }: { node: GraphNode }) => (
    <div className="absolute left-4 bottom-20 bg-slate-900 text-white px-3 py-2 rounded-lg shadow-lg text-sm z-10 max-w-xs pointer-events-none">
      <div className="font-medium">{node.label}</div>
      <div className="text-slate-300 text-xs mt-1">
        {node.type} 
        {node.cluster && ` • ${node.cluster}`}
        {node.confidence && ` • ${(node.confidence * 100).toFixed(0)}% confidence`}
      </div>
      {node.date && (
        <div className="text-slate-400 text-xs">{node.date}</div>
      )}
      <div className="text-slate-400 text-xs mt-1">Click to select</div>
    </div>
  )

  // Prepared graph data for ForceGraph3D
  const graphDataPrepared = useMemo(() => {
    if (!graphData) return { nodes: [], links: [] }
    return {
      nodes: graphData.nodes,
      links: graphData.links.map(link => ({
        ...link,
        // Ensure source/target are IDs for force graph
        source: typeof link.source === 'string' ? link.source : link.source.id,
        target: typeof link.target === 'string' ? link.target : link.target.id,
      }))
    }
  }, [graphData])

  return (
    <div className="h-full flex flex-col bg-slate-900 rounded-lg border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="p-4 bg-slate-800 border-b border-slate-700 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Box className="text-blue-400" size={20} />
          <h2 className="font-semibold text-white">Knowledge Graph 3D</h2>
          {graphData && (
            <span className="text-sm text-slate-400">
              {graphData.stats.total_nodes} nodes • {graphData.stats.total_links} links
            </span>
          )}
        </div>
        
        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowLabels(!showLabels)}
            className={`p-2 rounded transition-colors ${
              showLabels ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
            title={showLabels ? 'Hide labels' : 'Show labels'}
          >
            {showLabels ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`p-2 rounded transition-colors ${
              showFilters ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
            title="Toggle filters"
          >
            <Filter size={16} />
          </button>
          <button 
            onClick={fetchGraphData} 
            className="p-2 bg-slate-700 text-slate-300 hover:bg-slate-600 rounded transition-colors"
            title="Refresh"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          </button>
          <button 
            onClick={handleFitView}
            className="p-2 bg-slate-700 text-slate-300 hover:bg-slate-600 rounded transition-colors"
            title="Fit to view"
          >
            <Maximize2 size={16} />
          </button>
        </div>
      </div>
      
      {/* Filters */}
      {showFilters && (
        <div className="p-3 bg-slate-800/50 border-b border-slate-700 flex items-center gap-6">
          <span className="text-sm text-slate-400 font-medium">Show:</span>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showEvents}
              onChange={e => setFilters({ ...filters, showEvents: e.target.checked })}
              className="rounded border-slate-600 bg-slate-700"
            />
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS.event }} />
            <span className="text-sm text-slate-300">Events ({graphData?.stats.events || 0})</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showPapers}
              onChange={e => setFilters({ ...filters, showPapers: e.target.checked })}
              className="rounded border-slate-600 bg-slate-700"
            />
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS.paper }} />
            <span className="text-sm text-slate-300">Papers ({graphData?.stats.papers || 0})</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showPatterns}
              onChange={e => setFilters({ ...filters, showPatterns: e.target.checked })}
              className="rounded border-slate-600 bg-slate-700"
            />
            <span className="w-3 h-3 rounded-full" style={{ backgroundColor: NODE_COLORS.pattern }} />
            <span className="text-sm text-slate-300">Patterns ({graphData?.stats.patterns || 0})</span>
          </label>
        </div>
      )}
      
      {/* Main Graph Area */}
      <div ref={containerRef} className="flex-1 relative min-h-[500px]">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-20">
            <div className="flex items-center gap-3">
              <RefreshCw className="animate-spin text-blue-400" size={24} />
              <span className="text-slate-300">Loading knowledge graph...</span>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center z-20">
            <div className="text-center p-6">
              <AlertTriangle className="mx-auto text-red-500 mb-3" size={32} />
              <p className="text-red-400 font-medium">{error}</p>
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
          <div className="absolute inset-0 flex items-center justify-center z-20">
            <div className="text-center p-6">
              <Network className="mx-auto text-slate-600 mb-3" size={48} />
              <p className="text-slate-400 font-medium">No data in knowledge graph</p>
              <p className="text-slate-500 text-sm mt-1">
                Run an investigation to populate the knowledge base
              </p>
            </div>
          </div>
        )}
        
        {graphData && graphData.nodes.length > 0 && (
          <ForceGraph3D
            ref={fgRef}
            graphData={graphDataPrepared}
            width={dimensions.width}
            height={dimensions.height}
            backgroundColor="#0f172a"
            
            // Node styling
            nodeThreeObject={nodeThreeObject}
            nodeThreeObjectExtend={false}
            nodeColor={nodeColor}
            nodeVal={nodeVal}
            nodeLabel={() => ''}
            nodeOpacity={0.9}
            
            // Link styling
            linkColor={linkColor}
            linkWidth={linkWidth}
            linkOpacity={0.6}
            linkDirectionalParticles={2}
            linkDirectionalParticleWidth={1.5}
            linkDirectionalParticleSpeed={0.005}
            linkDirectionalParticleColor={linkColor}
            
            // Interaction
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
            enableNodeDrag={true}
            enableNavigationControls={true}
            controlType="orbit"
            
            // Physics
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
            warmupTicks={100}
            cooldownTicks={100}
          />
        )}
        
        {/* Selected Node Details */}
        {selectedNode && <NodeDetails node={selectedNode} />}
        
        {/* Hover Tooltip */}
        {hoveredNode && !selectedNode && <Tooltip node={hoveredNode} />}
        
        {/* Legend */}
        <div className="absolute left-4 top-4 bg-slate-800/95 backdrop-blur border border-slate-700 rounded-lg p-3 shadow-lg z-10">
          <div className="text-xs text-slate-400 font-medium mb-2">Legend</div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2 text-sm">
              <div className="w-4 h-4 rounded-full border-2 border-slate-600" style={{ backgroundColor: NODE_COLORS.event }} />
              <span className="text-slate-300">Events</span>
              <span className="text-slate-500 text-xs">({graphData?.stats.events || 0})</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-3 h-3 rounded-full border-2 border-slate-600" style={{ backgroundColor: NODE_COLORS.paper }} />
              <span className="text-slate-300">Papers</span>
              <span className="text-slate-500 text-xs">({graphData?.stats.papers || 0})</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-3.5 h-3.5 rounded-full border-2 border-slate-600" style={{ backgroundColor: NODE_COLORS.pattern }} />
              <span className="text-slate-300">Patterns</span>
              <span className="text-slate-500 text-xs">({graphData?.stats.patterns || 0})</span>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-700">
            <div className="text-xs text-slate-400 font-medium mb-1">Link Strength</div>
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs">
                <div className="w-8 h-1 rounded" style={{ backgroundColor: '#059669' }} />
                <span className="text-slate-400">Strong</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-6 h-0.5 rounded" style={{ backgroundColor: '#3b82f6' }} />
                <span className="text-slate-400">Moderate</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-4 h-0.5 rounded" style={{ backgroundColor: '#64748b' }} />
                <span className="text-slate-400">Weak</span>
              </div>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-700 text-xs text-slate-500">
            <p>🖱️ Drag to rotate</p>
            <p>⚙️ Scroll to zoom</p>
            <p>🎯 Click node for details</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default KnowledgeGraph3DView
