/**
 * Cosmograph Knowledge Graph Explorer
 * GPU-accelerated visualization using @cosmograph/react v2.x
 * 
 * Uses prepareCosmographData() API as per docs/COSMOGRAPH_REFERENCE.md
 */

import { useEffect, useState, useRef, useCallback } from 'react'
import { 
  Cosmograph, 
  CosmographProvider,
  prepareCosmographData,
} from '@cosmograph/react'
import type { CosmographRef, CosmographConfig } from '@cosmograph/react'
import { useStore } from '../store'

// Types for backend data
interface KnowledgeNode {
  id: string
  name: string
  type: string
  date?: string
  properties?: Record<string, unknown>
}

interface KnowledgeLink {
  source: string
  target: string
  type: string
  weight?: number
  date?: string
}

interface GraphData {
  nodes: KnowledgeNode[]
  links: KnowledgeLink[]
  stats: {
    total_nodes: number
    total_links: number
    papers: number
    events: number
    patterns: number
  }
}

// Color mapping per tipo di nodo
const NODE_COLORS: Record<string, string> = {
  paper: '#3b82f6',
  event: '#f59e0b',
  pattern: '#10b981',
  variable: '#8b5cf6',
  region: '#ec4899',
  concept: '#06b6d4',
  default: '#64748b'
}

export function CosmographView() {
  const { backend } = useStore()
  const cosmographRef = useRef<CosmographRef>(null)
  
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [cosmographConfig, setCosmographConfig] = useState<CosmographConfig>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<KnowledgeNode | null>(null)

  // Fetch and prepare data using prepareCosmographData
  useEffect(() => {
    const fetchAndPrepare = async () => {
      try {
        setLoading(true)
        
        // Fetch from backend
        const res = await fetch(`/api/v1/knowledge/graph?backend=${backend}`)
        if (!res.ok) throw new Error('Failed to fetch graph data')
        const data: GraphData = await res.json()
        setGraphData(data)
        
        // Create node id set for filtering links
        const nodeIds = new Set(data.nodes.map(n => n.id))
        
        // Transform to Cosmograph format - raw arrays
        const rawPoints = data.nodes.map(node => ({
          id: node.id,
          label: node.name || node.id,
          nodeType: node.type,
          color: NODE_COLORS[node.type] || NODE_COLORS.default,
        }))
        
        const rawLinks = data.links
          .filter(link => nodeIds.has(link.source) && nodeIds.has(link.target))
          .map(link => ({
            source: link.source,
            target: link.target,
          }))
        
        // Data config maps fields to Cosmograph's internal format
        const dataConfig = {
          points: {
            pointIdBy: 'id',
            pointLabelBy: 'label',
            pointColorBy: 'color',
          },
          links: {
            linkSourceBy: 'source',
            linkTargetsBy: ['target'], // Must be array!
          },
        }
        
        // Prepare data using Cosmograph's helper
        const result = await prepareCosmographData(dataConfig, rawPoints, rawLinks)
        
        if (result) {
          const { points, links, cosmographConfig: config } = result
          setCosmographConfig({
            points,
            links,
            ...config,
            // Additional visual config
            backgroundColor: '#0f172a',
            fitViewOnInit: true,
            fitViewDelay: 500,
            pointSize: 6,
            linkWidth: 0.5,
            linkArrows: false,
            spaceSize: 4096,
            showLabels: false, // Disable built-in labels - too long
          })
        }
        
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }
    
    fetchAndPrepare()
  }, [backend])

  // Handle click - receives point index
  const handleClick = useCallback((pointIndex: number | undefined) => {
    if (pointIndex === undefined) {
      setSelectedNode(null)
      return
    }
    // Look up the original node by index
    if (graphData && pointIndex < graphData.nodes.length) {
      setSelectedNode(graphData.nodes[pointIndex])
    }
  }, [graphData])

  // Fit view
  const fitView = useCallback(() => {
    cosmographRef.current?.fitView()
  }, [])

  // Pause/Resume simulation
  const [paused, setPaused] = useState(false)
  const togglePause = useCallback(() => {
    if (paused) {
      cosmographRef.current?.start()
    } else {
      cosmographRef.current?.pause()
    }
    setPaused(!paused)
  }, [paused])

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-900 rounded-lg">
        <div className="text-center">
          <div className="animate-spin w-10 h-10 border-3 border-blue-500 border-t-transparent rounded-full mx-auto mb-3" />
          <p className="text-slate-300">Loading knowledge graph...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-900 rounded-lg">
        <div className="text-center p-6 bg-red-900/30 rounded-lg">
          <p className="text-red-400 text-lg">Error: {error}</p>
          <button 
            onClick={() => window.location.reload()}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-slate-900 rounded-lg border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="p-4 bg-slate-800 border-b border-slate-700">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="font-semibold text-white text-lg">Knowledge Graph Explorer</h2>
            {graphData && (
              <p className="text-sm text-slate-400 mt-1">
                {graphData.stats.total_nodes} nodes • {graphData.stats.total_links} links
              </p>
            )}
          </div>
          
          <div className="flex items-center gap-4">
            {/* Legend */}
            <div className="flex gap-2 text-xs">
              {Object.entries(NODE_COLORS)
                .filter(([k]) => k !== 'default')
                .slice(0, 4)
                .map(([type, color]) => (
                  <div key={type} className="flex items-center gap-1">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                    <span className="text-slate-400 capitalize">{type}</span>
                  </div>
                ))}
            </div>
            
            {/* Controls */}
            <div className="flex gap-2">
              <button
                onClick={togglePause}
                className={`px-3 py-1.5 text-xs rounded ${
                  paused 
                    ? 'bg-green-600 hover:bg-green-700' 
                    : 'bg-amber-600 hover:bg-amber-700'
                } text-white`}
              >
                {paused ? '▶ Play' : '⏸ Pause'}
              </button>
              <button
                onClick={fitView}
                className="px-3 py-1.5 text-xs rounded bg-slate-600 hover:bg-slate-500 text-white"
              >
                ⊡ Fit
              </button>
            </div>
          </div>
        </div>
      </div>
      
      {/* Graph */}
      <div className="flex-1 relative">
        <CosmographProvider>
          <Cosmograph
            ref={cosmographRef}
            {...cosmographConfig}
            onClick={handleClick}
          />
        </CosmographProvider>
        
        {/* Selected Node Panel */}
        {selectedNode && (
          <div className="absolute top-4 left-4 p-4 bg-slate-800/95 rounded-lg border border-slate-600 max-w-sm z-20 shadow-xl">
            <div className="flex justify-between items-start">
              <div>
                <h3 className="font-semibold text-white">{selectedNode.name}</h3>
                <p className="text-xs text-slate-400 mt-1 flex items-center gap-1">
                  <span 
                    className="inline-block w-2 h-2 rounded-full"
                    style={{ backgroundColor: NODE_COLORS[selectedNode.type] || NODE_COLORS.default }}
                  />
                  <span className="capitalize">{selectedNode.type}</span>
                </p>
              </div>
              <button 
                onClick={() => setSelectedNode(null)}
                className="text-slate-400 hover:text-white text-lg"
              >
                ×
              </button>
            </div>
            
            {selectedNode.properties && Object.keys(selectedNode.properties).length > 0 && (
              <div className="mt-3 pt-3 border-t border-slate-700 text-xs text-slate-300">
                {Object.entries(selectedNode.properties).slice(0, 4).map(([k, v]) => (
                  <div key={k} className="truncate mb-1">
                    <span className="text-slate-500">{k}:</span> {String(v).slice(0, 60)}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div className="p-2 bg-slate-800 border-t border-slate-700 text-xs text-slate-500 text-center">
        🖱️ Drag to pan • Scroll to zoom • Click node for details
      </div>
    </div>
  )
}

export default CosmographView
