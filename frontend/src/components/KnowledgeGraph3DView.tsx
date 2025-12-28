/**
 * Knowledge Graph 3D Explorer
 * Interactive 3D visualization using react-force-graph-3d
 */

import { useEffect, useState, useRef, useCallback } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import { useStore } from '../store'
import SpriteText from 'three-spritetext'

interface GraphNode {
  id: string
  name: string
  type: string
  properties?: Record<string, any>
}

interface GraphLink {
  source: string
  target: string
  type: string
  weight?: number
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
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
  paper: '#3b82f6',      // blue
  event: '#f59e0b',      // amber
  pattern: '#10b981',    // emerald
  variable: '#8b5cf6',   // violet
  region: '#ec4899',     // pink
  default: '#64748b'     // slate
}

export function KnowledgeGraph3DView() {
  const { backend } = useStore()
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const fgRef = useRef<any>()
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

  // Resize handler
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        })
      }
    }
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  // Fetch data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const res = await fetch(`/api/v1/knowledge/graph?backend=${backend}`)
        if (!res.ok) throw new Error('Failed to fetch')
        const data = await res.json()
        setGraphData(data)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [backend])

  // Node click handler - zoom to node
  const handleNodeClick = useCallback((node: any) => {
    if (fgRef.current) {
      const distance = 100
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z)
      fgRef.current.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
        node,
        1000
      )
    }
  }, [])

  // Node hover handler
  const handleNodeHover = useCallback((node: any) => {
    setHoveredNode(node || null)
  }, [])

  // Create node label sprite
  const nodeThreeObject = useCallback((node: any) => {
    const sprite = new SpriteText(node.name || node.id)
    sprite.color = '#ffffff'
    sprite.textHeight = 4
    sprite.backgroundColor = 'rgba(0,0,0,0.6)'
    sprite.padding = 2
    sprite.borderRadius = 3
    return sprite
  }, [])

  // Get node color by type
  const getNodeColor = useCallback((node: any) => {
    return NODE_COLORS[node.type] || NODE_COLORS.default
  }, [])

  return (
    <div className="h-full flex flex-col bg-slate-900 rounded-lg border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="p-4 bg-slate-800 border-b border-slate-700 flex justify-between items-center">
        <div>
          <h2 className="font-semibold text-white">Knowledge Graph 3D</h2>
          {graphData && (
            <p className="text-sm text-slate-400 mt-1">
              {graphData.stats.total_nodes} nodes • {graphData.stats.total_links} links
              {graphData.stats.papers > 0 && ` • ${graphData.stats.papers} papers`}
              {graphData.stats.events > 0 && ` • ${graphData.stats.events} events`}
            </p>
          )}
        </div>
        
        {/* Legend */}
        <div className="flex gap-3 text-xs">
          {Object.entries(NODE_COLORS).filter(([k]) => k !== 'default').map(([type, color]) => (
            <div key={type} className="flex items-center gap-1">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-slate-400 capitalize">{type}</span>
            </div>
          ))}
        </div>
      </div>
      
      {/* Graph Container */}
      <div ref={containerRef} className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-10">
            <div className="text-center">
              <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-2" />
              <p className="text-slate-300">Loading graph...</p>
            </div>
          </div>
        )}
        
        {error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center p-4 bg-red-900/30 rounded-lg">
              <p className="text-red-400">Error: {error}</p>
              <button 
                onClick={() => window.location.reload()}
                className="mt-2 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Retry
              </button>
            </div>
          </div>
        )}
        
        {graphData && !loading && !error && (
          <ForceGraph3D
            ref={fgRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={{
              nodes: graphData.nodes.map(n => ({ ...n })),
              links: graphData.links.map(l => ({ ...l }))
            }}
            nodeLabel={(node: any) => `${node.name || node.id}\n(${node.type})`}
            nodeColor={getNodeColor}
            nodeOpacity={0.9}
            nodeResolution={16}
            nodeRelSize={6}
            linkColor={() => 'rgba(255,255,255,0.2)'}
            linkWidth={1}
            linkOpacity={0.4}
            linkDirectionalParticles={2}
            linkDirectionalParticleSpeed={0.005}
            linkDirectionalParticleWidth={2}
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
            backgroundColor="#0f172a"
            showNavInfo={false}
          />
        )}
        
        {/* Hover Info Panel */}
        {hoveredNode && (
          <div className="absolute top-4 left-4 p-3 bg-slate-800/95 rounded-lg border border-slate-600 max-w-xs z-20">
            <h3 className="font-semibold text-white truncate">{hoveredNode.name || hoveredNode.id}</h3>
            <p className="text-xs text-slate-400 capitalize mt-1">Type: {hoveredNode.type}</p>
            {hoveredNode.properties && Object.keys(hoveredNode.properties).length > 0 && (
              <div className="mt-2 text-xs text-slate-300">
                {Object.entries(hoveredNode.properties).slice(0, 3).map(([k, v]) => (
                  <div key={k} className="truncate">
                    <span className="text-slate-500">{k}:</span> {String(v).slice(0, 50)}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Controls hint */}
      <div className="p-2 bg-slate-800 border-t border-slate-700 text-xs text-slate-500 text-center">
        🖱️ Drag to rotate • Scroll to zoom • Click node to focus
      </div>
    </div>
  )
}

export default KnowledgeGraph3DView
