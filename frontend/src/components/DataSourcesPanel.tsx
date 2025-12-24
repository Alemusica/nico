/**
 * Data Sources Panel
 * 
 * Manage data source connections, resolution settings, and cache.
 */

import { useState, useEffect } from 'react'
import clsx from 'clsx'
import { 
  Cloud, Settings, Trash2, RefreshCw, 
  CheckCircle, XCircle, ChevronDown, ChevronUp,
  HardDrive, Clock, Globe
} from 'lucide-react'

interface DataSource {
  name: string
  description: string
  enabled: boolean
  connected: boolean
  variables: string[]
  default_resolution: {
    temporal: string
    spatial: string
  }
  min_start_date: string
}

interface CacheStats {
  total_entries: number
  total_size_mb: number
  sources: Record<string, {
    count: number
    size_mb: number
    accesses: number
  }>
}

interface Resolution {
  value: string
  label: string
  description: string
}

const API_BASE = 'http://localhost:8000'

export function DataSourcesPanel() {
  const [sources, setSources] = useState<Record<string, DataSource>>({})
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  const [resolutions, setResolutions] = useState<{
    temporal: Resolution[]
    spatial: Resolution[]
  }>({ temporal: [], spatial: [] })
  const [currentResolution, setCurrentResolution] = useState({ temporal: 'daily', spatial: '0.25' })
  const [loading, setLoading] = useState(true)
  const [expandedSource, setExpandedSource] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      // Load sources
      const sourcesRes = await fetch(`${API_BASE}/data/sources`)
      if (sourcesRes.ok) {
        const data = await sourcesRes.json()
        setSources(data.sources)
        setCurrentResolution(data.default_resolution)
      }
      
      // Load cache stats
      const cacheRes = await fetch(`${API_BASE}/data/cache/stats`)
      if (cacheRes.ok) {
        setCacheStats(await cacheRes.json())
      }
      
      // Load resolution options
      const resRes = await fetch(`${API_BASE}/data/resolutions`)
      if (resRes.ok) {
        setResolutions(await resRes.json())
      }
    } catch (err) {
      setError('Failed to load data sources. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const updateResolution = async (temporal?: string, spatial?: string) => {
    const newRes = {
      temporal: temporal || currentResolution.temporal,
      spatial: spatial || currentResolution.spatial,
    }
    
    try {
      const res = await fetch(`${API_BASE}/data/resolution`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newRes),
      })
      
      if (res.ok) {
        const data = await res.json()
        setCurrentResolution(data.resolution)
      }
    } catch (err) {
      console.error('Failed to update resolution:', err)
    }
  }

  const clearCache = async (source?: string) => {
    if (!confirm(`Clear ${source || 'all'} cache?`)) return
    
    try {
      const url = source 
        ? `${API_BASE}/data/cache?source=${source}`
        : `${API_BASE}/data/cache`
      
      await fetch(url, { method: 'DELETE' })
      loadData()
    } catch (err) {
      console.error('Failed to clear cache:', err)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="spinner" />
        <span className="ml-3 text-slate-500">Loading data sources...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 rounded-lg text-red-700">
        <p>{error}</p>
        <button onClick={loadData} className="mt-2 btn btn-sm">
          <RefreshCw className="w-4 h-4 mr-1" /> Retry
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Resolution Settings */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5 text-blue-500" />
          <h3 className="font-semibold">Default Resolution</h3>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          {/* Temporal Resolution */}
          <div>
            <label className="text-sm text-slate-600 mb-1 block">
              <Clock className="w-4 h-4 inline mr-1" />
              Temporal
            </label>
            <select
              value={currentResolution.temporal}
              onChange={(e) => updateResolution(e.target.value, undefined)}
              className="input w-full"
            >
              {resolutions.temporal.map(r => (
                <option key={r.value} value={r.value} title={r.description}>
                  {r.label}
                </option>
              ))}
            </select>
          </div>
          
          {/* Spatial Resolution */}
          <div>
            <label className="text-sm text-slate-600 mb-1 block">
              <Globe className="w-4 h-4 inline mr-1" />
              Spatial
            </label>
            <select
              value={currentResolution.spatial}
              onChange={(e) => updateResolution(undefined, e.target.value)}
              className="input w-full"
            >
              {resolutions.spatial.map(r => (
                <option key={r.value} value={r.value} title={r.description}>
                  {r.label}
                </option>
              ))}
            </select>
          </div>
        </div>
        
        <p className="text-xs text-slate-500 mt-2">
          Lower resolution = faster downloads. Adjust based on your analysis needs.
        </p>
      </div>

      {/* Data Sources */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Cloud className="w-5 h-5 text-green-500" />
            <h3 className="font-semibold">Data Sources</h3>
          </div>
          <button onClick={loadData} className="btn btn-ghost btn-sm">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
        
        <div className="space-y-2">
          {Object.entries(sources).map(([key, source]) => (
            <div 
              key={key}
              className={clsx(
                "border rounded-lg transition-all",
                source.connected ? "border-green-200 bg-green-50/50" : "border-slate-200"
              )}
            >
              <div 
                className="flex items-center justify-between p-3 cursor-pointer"
                onClick={() => setExpandedSource(expandedSource === key ? null : key)}
              >
                <div className="flex items-center gap-3">
                  {source.connected ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : (
                    <XCircle className="w-5 h-5 text-slate-400" />
                  )}
                  <div>
                    <h4 className="font-medium">{source.name}</h4>
                    <p className="text-xs text-slate-500">{source.description}</p>
                  </div>
                </div>
                {expandedSource === key ? (
                  <ChevronUp className="w-4 h-4 text-slate-400" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-slate-400" />
                )}
              </div>
              
              {expandedSource === key && (
                <div className="px-3 pb-3 border-t border-slate-100">
                  <div className="grid grid-cols-2 gap-4 mt-3 text-sm">
                    <div>
                      <span className="text-slate-500">Variables:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {source.variables.slice(0, 5).map(v => (
                          <span key={v} className="px-2 py-0.5 bg-slate-100 rounded text-xs">
                            {v}
                          </span>
                        ))}
                        {source.variables.length > 5 && (
                          <span className="px-2 py-0.5 text-slate-400 text-xs">
                            +{source.variables.length - 5} more
                          </span>
                        )}
                      </div>
                    </div>
                    <div>
                      <span className="text-slate-500">Data from:</span>
                      <p className="font-mono text-xs mt-1">{source.min_start_date}</p>
                    </div>
                  </div>
                  
                  {cacheStats?.sources[key] && (
                    <div className="mt-3 p-2 bg-slate-50 rounded text-xs">
                      <span className="text-slate-500">Cached: </span>
                      <span className="font-medium">
                        {cacheStats.sources[key].count} files 
                        ({cacheStats.sources[key].size_mb.toFixed(1)} MB)
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Cache Stats */}
      {cacheStats && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <HardDrive className="w-5 h-5 text-purple-500" />
              <h3 className="font-semibold">Cache</h3>
            </div>
            <button 
              onClick={() => clearCache()}
              className="btn btn-ghost btn-sm text-red-500 hover:bg-red-50"
            >
              <Trash2 className="w-4 h-4 mr-1" /> Clear All
            </button>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 bg-slate-50 rounded-lg">
              <p className="text-2xl font-bold">{cacheStats.total_entries}</p>
              <p className="text-xs text-slate-500">Total Files</p>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg">
              <p className="text-2xl font-bold">{cacheStats.total_size_mb.toFixed(1)} MB</p>
              <p className="text-xs text-slate-500">Total Size</p>
            </div>
          </div>
          
          {cacheStats.total_entries > 0 && (
            <div className="mt-4 space-y-2">
              {Object.entries(cacheStats.sources).map(([source, stats]) => (
                <div key={source} className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">{source}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-slate-400">
                      {stats.count} files, {stats.size_mb.toFixed(1)} MB
                    </span>
                    <button
                      onClick={() => clearCache(source)}
                      className="p-1 hover:bg-red-50 rounded text-red-400 hover:text-red-600"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
