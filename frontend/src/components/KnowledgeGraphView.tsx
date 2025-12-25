/**
 * Knowledge Graph Explorer - Cosmograph Integration
 * 
 * Event-centric 3D graph visualization for exploring:
 * - Events (central nodes)
 * - Papers (research evidence)
 * - Patterns (causal relationships)
 * 
 * LLM Cockpit commands:
 * - "expand geographically" → show nearby events
 * - "find physical correlations" → highlight causal links
 * - "show precursors" → trace back causal chain
 * - "current risk assessment" → compare with present conditions
 */

import { useEffect, useState, useCallback } from 'react'
import { Cosmograph, prepareCosmographData } from '@cosmograph/react'
import type { CosmographConfig } from '@cosmograph/react'
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
  ChevronRight,
  Send,
  Search
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
}

interface GraphLink {
  source: string
  target: string
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
  event: '#ef4444',      // red-500 - events are central/important
  paper: '#3b82f6',      // blue-500 - research papers
  pattern: '#10b981',    // emerald-500 - causal patterns
  data_source: '#8b5cf6', // violet-500 - data sources
  climate_index: '#f59e0b', // amber-500 - climate indices
}

const NODE_ICONS: Record<string, any> = {
  event: AlertTriangle,
  paper: FileText,
  pattern: Zap,
  data_source: Network,
  climate_index: Calendar,
}

export function KnowledgeGraphView() {
  const { backend } = useStore()
  
  // State
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [cockpitCommand, setCockpitCommand] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState({
    showPapers: true,
    showEvents: true,
    showPatterns: true,
  })
  
  // Cosmograph config state
  const [cosmographConfig, setCosmographConfig] = useState<CosmographConfig>({})
  const [cosmographPoints, setCosmographPoints] = useState<any[]>([])
  const [cosmographLinks, setCosmographLinks] = useState<any[]>([])

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
        limit_papers: '100',
        limit_events: '50',
      })
      
      const res = await fetch(`/api/v1/knowledge/graph?${params}`)
      if (!res.ok) throw new Error(`Failed to fetch graph: ${res.statusText}`)
      
      const data: GraphData = await res.json()
      setGraphData(data)
      
      // Prepare data for Cosmograph
      if (data.nodes.length > 0) {
        await prepareCosmographView(data)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [backend, filters])

  // Prepare Cosmograph data
  const prepareCosmographView = async (data: GraphData) => {
    // Transform nodes to Cosmograph format
    const rawPoints = data.nodes.map(node => ({
      id: node.id,
      label: node.label,
      type: node.type,
      date: node.date,
      confidence: node.confidence || 0.5,
      cluster: node.cluster || node.type,
      // Color based on type
      color: NODE_COLORS[node.type] || '#94a3b8',
      // Size based on confidence/importance
      size: node.type === 'event' ? 15 : (node.confidence || 0.5) * 10 + 5,
    }))

    // Transform links
    const rawLinks = data.links.map(link => ({
      source: link.source,
      target: link.target,
      type: link.type,
      strength: link.strength,
      label: link.label,
    }))

    // Prepare for Cosmograph
    const dataConfig = {
      points: {
        pointIdBy: 'id',
      },
      links: {
        linkSourceBy: 'source',
        linkTargetsBy: ['target'],
      },
    }

    try {
      const result = await prepareCosmographData(dataConfig, rawPoints, rawLinks)
      if (result) {
        // @ts-expect-error - Cosmograph types are complex
        setCosmographPoints(result.points || [])
        // @ts-expect-error - Cosmograph types are complex
        setCosmographLinks(result.links || [])
        setCosmographConfig({
          ...result.cosmographConfig,
          // Custom styling - use string colors
          pointColor: '#3b82f6',
          pointSize: 8,
          linkWidth: 1,
          linkColor: '#475569',
          linkArrows: true,
          // Simulation settings
          simulationGravity: 0.25,
          simulationRepulsion: 1,
          simulationLinkSpring: 0.3,
          // Events
          onClick: (point: any) => {
            if (point) {
              const node = data.nodes.find(n => n.id === point.id)
              setSelectedNode(node || null)
            } else {
              setSelectedNode(null)
            }
          },
        } as CosmographConfig)
      }
    } catch (err) {
      console.error('Failed to prepare Cosmograph data:', err)
    }
  }

  // Load on mount
  useEffect(() => {
    fetchGraphData()
  }, [fetchGraphData])

  // Handle cockpit command
  const handleCockpitCommand = async () => {
    if (!cockpitCommand.trim()) return
    
    const cmd = cockpitCommand.toLowerCase()
    
    // Parse commands
    if (cmd.includes('expand') && selectedNode) {
      // Expand selected node
      try {
        const res = await fetch(`/api/v1/knowledge/graph/expand/${selectedNode.id}?backend=${backend}`)
        const expanded = await res.json()
        console.log('Expanded:', expanded)
        // TODO: Merge expanded nodes into graph
      } catch (err) {
        console.error('Expand failed:', err)
      }
    } else if (cmd.includes('refresh') || cmd.includes('reload')) {
      fetchGraphData()
    } else if (cmd.includes('filter event')) {
      setFilters({ ...filters, showEvents: true, showPapers: false, showPatterns: false })
    } else if (cmd.includes('filter paper')) {
      setFilters({ ...filters, showEvents: false, showPapers: true, showPatterns: false })
    } else if (cmd.includes('show all')) {
      setFilters({ showEvents: true, showPapers: true, showPatterns: true })
    }
    
    setCockpitCommand('')
  }

  // Node details panel
  const NodeDetails = ({ node }: { node: GraphNode }) => {
    const Icon = NODE_ICONS[node.type] || Info
    
    return (
      <div className="absolute right-4 top-4 w-80 bg-phi-base/95 backdrop-blur border border-phi-border rounded-lg shadow-xl overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-phi-border" style={{ backgroundColor: NODE_COLORS[node.type] + '20' }}>
          <div className="flex items-center gap-2">
            <Icon size={20} style={{ color: NODE_COLORS[node.type] }} />
            <span className="text-phi-xs uppercase tracking-wider text-phi-secondary">
              {node.type}
            </span>
          </div>
          <h3 className="font-medium mt-2">{node.label}</h3>
        </div>
        
        {/* Content */}
        <div className="p-4 space-y-3 text-phi-sm">
          {node.date && (
            <div className="flex items-center gap-2 text-phi-secondary">
              <Calendar size={14} />
              <span>{node.date}</span>
            </div>
          )}
          
          {(node.lat && node.lon) && (
            <div className="flex items-center gap-2 text-phi-secondary">
              <MapPin size={14} />
              <span>{node.lat.toFixed(2)}°, {node.lon.toFixed(2)}°</span>
            </div>
          )}
          
          {node.confidence && (
            <div className="flex items-center gap-2">
              <span className="text-phi-secondary">Confidence:</span>
              <div className="flex-1 h-2 bg-phi-subtle rounded-full overflow-hidden">
                <div 
                  className="h-full bg-phi-accent" 
                  style={{ width: `${node.confidence * 100}%` }}
                />
              </div>
              <span className="text-phi-xs">{(node.confidence * 100).toFixed(0)}%</span>
            </div>
          )}
          
          {node.cluster && (
            <div>
              <span className="text-phi-secondary">Cluster:</span>
              <span className="ml-2 px-2 py-0.5 bg-phi-subtle rounded text-phi-xs">{node.cluster}</span>
            </div>
          )}
          
          {/* Metadata */}
          {node.metadata && Object.keys(node.metadata).length > 0 && (
            <div className="pt-2 border-t border-phi-border">
              {node.metadata.description && (
                <p className="text-phi-secondary text-phi-xs line-clamp-3">
                  {node.metadata.description}
                </p>
              )}
              {node.metadata.authors && (
                <p className="text-phi-xs text-phi-muted mt-1">
                  {node.metadata.authors.slice(0, 3).join(', ')}
                  {node.metadata.authors.length > 3 && ' et al.'}
                </p>
              )}
            </div>
          )}
        </div>
        
        {/* Actions */}
        <div className="p-3 border-t border-phi-border bg-phi-subtle/50 flex gap-2">
          <button 
            onClick={() => handleCockpitCommand()}
            className="flex-1 px-3 py-1.5 bg-phi-accent text-white rounded text-phi-sm flex items-center justify-center gap-1"
          >
            <ChevronRight size={14} />
            Expand
          </button>
          <button 
            onClick={() => setSelectedNode(null)}
            className="px-3 py-1.5 border border-phi-border rounded text-phi-sm"
          >
            Close
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-phi-base">
      {/* Header */}
      <div className="p-4 border-b border-phi-border flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Network className="text-phi-accent" size={20} />
          <h2 className="font-semibold">Knowledge Graph Explorer</h2>
          {graphData && (
            <span className="text-phi-sm text-phi-secondary">
              {graphData.stats.total_nodes} nodes • {graphData.stats.total_links} links
            </span>
          )}
        </div>
        
        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`p-2 rounded ${showFilters ? 'bg-phi-accent text-white' : 'hover:bg-phi-subtle'}`}
          >
            <Filter size={16} />
          </button>
          <button onClick={fetchGraphData} className="p-2 hover:bg-phi-subtle rounded">
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          </button>
          <button className="p-2 hover:bg-phi-subtle rounded">
            <Maximize2 size={16} />
          </button>
        </div>
      </div>
      
      {/* Filters */}
      {showFilters && (
        <div className="p-3 border-b border-phi-border bg-phi-subtle/30 flex items-center gap-4">
          <span className="text-phi-sm text-phi-secondary">Show:</span>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showEvents}
              onChange={e => setFilters({ ...filters, showEvents: e.target.checked })}
              className="rounded"
            />
            <AlertTriangle size={14} style={{ color: NODE_COLORS.event }} />
            <span className="text-phi-sm">Events</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showPapers}
              onChange={e => setFilters({ ...filters, showPapers: e.target.checked })}
              className="rounded"
            />
            <FileText size={14} style={{ color: NODE_COLORS.paper }} />
            <span className="text-phi-sm">Papers</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filters.showPatterns}
              onChange={e => setFilters({ ...filters, showPatterns: e.target.checked })}
              className="rounded"
            />
            <Zap size={14} style={{ color: NODE_COLORS.pattern }} />
            <span className="text-phi-sm">Patterns</span>
          </label>
        </div>
      )}
      
      {/* Main Graph Area */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-phi-base/80 z-10">
            <div className="flex items-center gap-3">
              <RefreshCw className="animate-spin text-phi-accent" size={24} />
              <span>Loading knowledge graph...</span>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center p-6">
              <AlertTriangle className="mx-auto text-red-500 mb-3" size={32} />
              <p className="text-red-500">{error}</p>
              <button
                onClick={fetchGraphData}
                className="mt-4 px-4 py-2 bg-phi-accent text-white rounded"
              >
                Retry
              </button>
            </div>
          </div>
        )}
        
        {!loading && !error && graphData && (
          <>
            {graphData.nodes.length > 0 ? (
              <Cosmograph
                points={cosmographPoints}
                links={cosmographLinks}
                {...cosmographConfig}
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center p-6">
                  <Network className="mx-auto text-phi-muted mb-3" size={48} />
                  <p className="text-phi-secondary">No data in knowledge graph</p>
                  <p className="text-phi-muted text-phi-sm mt-1">
                    Run an investigation to populate the knowledge base
                  </p>
                </div>
              </div>
            )}
          </>
        )}
        
        {/* Selected Node Details */}
        {selectedNode && <NodeDetails node={selectedNode} />}
        
        {/* Legend */}
        <div className="absolute left-4 bottom-4 bg-phi-base/90 backdrop-blur border border-phi-border rounded-lg p-3">
          <div className="text-phi-xs text-phi-secondary mb-2">Legend</div>
          <div className="space-y-1.5">
            {Object.entries(NODE_COLORS).slice(0, 3).map(([type, color]) => {
              const Icon = NODE_ICONS[type] || Info
              return (
                <div key={type} className="flex items-center gap-2 text-phi-sm">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                  <Icon size={12} style={{ color }} />
                  <span className="capitalize">{type}s</span>
                  {graphData && (
                    <span className="text-phi-muted">
                      ({graphData.stats[type + 's' as keyof typeof graphData.stats] || 0})
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>
      
      {/* LLM Cockpit */}
      <div className="p-3 border-t border-phi-border bg-phi-subtle/30">
        <div className="flex items-center gap-2">
          <Search size={16} className="text-phi-secondary" />
          <input
            type="text"
            value={cockpitCommand}
            onChange={e => setCockpitCommand(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleCockpitCommand()}
            placeholder="LLM Cockpit: expand geographically, find correlations, show precursors..."
            className="flex-1 bg-transparent border-none outline-none text-phi-sm"
          />
          <button
            onClick={handleCockpitCommand}
            className="p-1.5 hover:bg-phi-subtle rounded"
          >
            <Send size={14} />
          </button>
        </div>
      </div>
    </div>
  )
}

export default KnowledgeGraphView
