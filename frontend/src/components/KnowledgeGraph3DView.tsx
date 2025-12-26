/**
 * Knowledge Graph 3D Explorer - MINIMAL TEST VERSION
 */

import { useEffect, useState } from 'react'
import { useStore } from '../store'

interface GraphData {
  nodes: any[]
  links: any[]
  stats: {
    total_nodes: number
    total_links: number
    papers: number
    events: number
    patterns: number
  }
}

export function KnowledgeGraph3DView() {
  const { backend } = useStore()
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
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

  return (
    <div className="h-full flex flex-col bg-slate-900 rounded-lg border border-slate-700">
      <div className="p-4 bg-slate-800 border-b border-slate-700">
        <h2 className="font-semibold text-white">Knowledge Graph 3D - TEST</h2>
        {graphData && (
          <p className="text-sm text-slate-400 mt-1">
            {graphData.stats.total_nodes} nodes • {graphData.stats.total_links} links
          </p>
        )}
      </div>
      
      <div className="flex-1 flex items-center justify-center">
        {loading && <p className="text-slate-300">Loading...</p>}
        {error && <p className="text-red-400">Error: {error}</p>}
        {graphData && !loading && !error && (
          <div className="text-center">
            <p className="text-green-400 text-lg">✓ Component Works!</p>
            <p className="text-slate-400 mt-2">
              Data loaded: {graphData.nodes.length} nodes
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default KnowledgeGraph3DView
