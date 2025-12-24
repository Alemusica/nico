/**
 * Data Explorer Component
 * 
 * Interactive UI for downloading data from satellite APIs
 * with resolution, position, and time range configuration.
 */

import { useState } from 'react'
import clsx from 'clsx'
import { 
  Download, MapPin, Calendar, Settings2, 
  Satellite, Cloud, BarChart3,
  Loader2
} from 'lucide-react'

const API_BASE = 'http://localhost:8000'

interface DownloadConfig {
  source: 'cmems' | 'era5' | 'climate_indices'
  variables: string[]
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
  start_date: string
  end_date: string
  temporal_resolution: string
  spatial_resolution: string
}

interface DownloadResult {
  success: boolean
  message: string
  file_path?: string
  size_mb?: number
}

const SOURCES = {
  cmems: {
    name: 'CMEMS Satellite',
    icon: Satellite,
    description: 'Sea level anomaly from satellite altimetry',
    variables: ['sla', 'adt', 'ugos', 'vgos'],
    default_vars: ['sla'],
    min_date: '2022-10-04',
    color: 'blue',
  },
  era5: {
    name: 'ERA5 Reanalysis',
    icon: Cloud,
    description: 'Atmospheric reanalysis (precipitation, temperature, pressure)',
    variables: ['total_precipitation', '2m_temperature', 'mean_sea_level_pressure', 'runoff', '10m_u_component_of_wind', '10m_v_component_of_wind'],
    default_vars: ['total_precipitation', '2m_temperature'],
    min_date: '1950-01-01',
    color: 'orange',
  },
  climate_indices: {
    name: 'Climate Indices',
    icon: BarChart3,
    description: 'NAO, AO, EA, SCAND teleconnection indices',
    variables: ['nao', 'ao', 'ea', 'scand', 'pna', 'amo'],
    default_vars: ['nao', 'ao'],
    min_date: '1950-01-01',
    color: 'purple',
  },
}

const TEMPORAL_OPTIONS = [
  { value: 'hourly', label: 'Hourly' },
  { value: '6-hourly', label: '6-hourly' },
  { value: 'daily', label: 'Daily' },
  { value: 'monthly', label: 'Monthly' },
]

const SPATIAL_OPTIONS = [
  { value: '0.1', label: '0.1°' },
  { value: '0.25', label: '0.25°' },
  { value: '0.5', label: '0.5°' },
  { value: '1.0', label: '1.0°' },
]

// Preset regions
const PRESET_REGIONS = [
  { name: 'Italy', lat: [36, 47], lon: [6, 19] },
  { name: 'Mediterranean', lat: [30, 46], lon: [-6, 36] },
  { name: 'Arctic', lat: [66, 90], lon: [-180, 180] },
  { name: 'Adriatic Sea', lat: [40, 46], lon: [12, 20] },
  { name: 'Custom', lat: [44, 46], lon: [8, 12] },
]

export function DataExplorer() {
  const [selectedSource, setSelectedSource] = useState<keyof typeof SOURCES>('era5')
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadResult, setDownloadResult] = useState<DownloadResult | null>(null)
  
  // Config state
  const [config, setConfig] = useState<DownloadConfig>({
    source: 'era5',
    variables: SOURCES.era5.default_vars,
    lat_min: 44,
    lat_max: 47,
    lon_min: 8,
    lon_max: 12,
    start_date: '2024-01-01',
    end_date: '2024-01-31',
    temporal_resolution: 'daily',
    spatial_resolution: '0.25',
  })

  const updateSource = (source: keyof typeof SOURCES) => {
    setSelectedSource(source)
    setConfig({
      ...config,
      source,
      variables: SOURCES[source].default_vars,
    })
  }

  const toggleVariable = (variable: string) => {
    if (config.variables.includes(variable)) {
      setConfig({ ...config, variables: config.variables.filter(v => v !== variable) })
    } else {
      setConfig({ ...config, variables: [...config.variables, variable] })
    }
  }

  const setPresetRegion = (preset: typeof PRESET_REGIONS[0]) => {
    setConfig({
      ...config,
      lat_min: preset.lat[0],
      lat_max: preset.lat[1],
      lon_min: preset.lon[0],
      lon_max: preset.lon[1],
    })
  }

  const estimateSize = (): string => {
    const latSpan = Math.abs(config.lat_max - config.lat_min)
    const lonSpan = Math.abs(config.lon_max - config.lon_min)
    const days = Math.ceil((new Date(config.end_date).getTime() - new Date(config.start_date).getTime()) / (1000 * 60 * 60 * 24))
    
    const spatialRes = parseFloat(config.spatial_resolution)
    const latPoints = Math.ceil(latSpan / spatialRes)
    const lonPoints = Math.ceil(lonSpan / spatialRes)
    
    let timePoints = days
    if (config.temporal_resolution === 'hourly') timePoints = days * 24
    else if (config.temporal_resolution === '6-hourly') timePoints = days * 4
    
    // Rough estimate: 4 bytes per float, per variable
    const sizeMB = (latPoints * lonPoints * timePoints * config.variables.length * 4) / (1024 * 1024)
    
    if (sizeMB < 1) return `~${Math.round(sizeMB * 1024)} KB`
    return `~${sizeMB.toFixed(1)} MB`
  }

  const downloadData = async () => {
    setIsDownloading(true)
    setDownloadResult(null)
    
    try {
      const res = await fetch(`${API_BASE}/data/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      
      const result = await res.json()
      setDownloadResult({
        success: res.ok,
        message: res.ok ? 'Download complete!' : result.detail || 'Download failed',
        file_path: result.file_path,
        size_mb: result.size_mb,
      })
    } catch (err) {
      setDownloadResult({
        success: false,
        message: `Error: ${err instanceof Error ? err.message : 'Unknown error'}`,
      })
    } finally {
      setIsDownloading(false)
    }
  }

  const sourceInfo = SOURCES[selectedSource]

  return (
    <div className="space-y-4">
      {/* Source Selection */}
      <div className="grid grid-cols-3 gap-2">
        {(Object.entries(SOURCES) as [keyof typeof SOURCES, typeof SOURCES.cmems][]).map(([key, source]) => {
          const Icon = source.icon
          const isSelected = selectedSource === key
          return (
            <button
              key={key}
              onClick={() => updateSource(key)}
              className={clsx(
                "p-3 rounded-lg border-2 transition-all text-left",
                isSelected 
                  ? `border-${source.color}-500 bg-${source.color}-50` 
                  : "border-slate-200 hover:border-slate-300"
              )}
            >
              <div className="flex items-center gap-2 mb-1">
                <Icon className={clsx("w-4 h-4", isSelected ? `text-${source.color}-500` : "text-slate-400")} />
                <span className={clsx("text-sm font-medium", isSelected ? "text-slate-800" : "text-slate-600")}>
                  {source.name}
                </span>
              </div>
              <p className="text-xs text-slate-500 line-clamp-1">{source.description}</p>
            </button>
          )
        })}
      </div>

      {/* Configuration */}
      <div className="card p-4 space-y-4">
        {/* Variables */}
        <div>
          <label className="text-sm font-medium text-slate-700 mb-2 block">
            📊 Variables
          </label>
          <div className="flex flex-wrap gap-2">
            {sourceInfo.variables.map(v => (
              <button
                key={v}
                onClick={() => toggleVariable(v)}
                className={clsx(
                  "px-3 py-1.5 text-sm rounded-full border transition-colors",
                  config.variables.includes(v)
                    ? "bg-blue-500 text-white border-blue-500"
                    : "bg-white text-slate-600 border-slate-200 hover:border-blue-300"
                )}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        {/* Region */}
        <div>
          <label className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-1">
            <MapPin className="w-4 h-4" />
            Region
          </label>
          
          {/* Presets */}
          <div className="flex flex-wrap gap-1 mb-2">
            {PRESET_REGIONS.map(preset => (
              <button
                key={preset.name}
                onClick={() => setPresetRegion(preset)}
                className="px-2 py-1 text-xs bg-slate-100 hover:bg-slate-200 rounded transition-colors"
              >
                {preset.name}
              </button>
            ))}
          </div>
          
          {/* Manual inputs */}
          <div className="grid grid-cols-4 gap-2">
            <div>
              <label className="text-xs text-slate-500">Lat Min</label>
              <input
                type="number"
                value={config.lat_min}
                onChange={(e) => setConfig({ ...config, lat_min: parseFloat(e.target.value) })}
                className="input w-full text-sm"
                step="0.1"
              />
            </div>
            <div>
              <label className="text-xs text-slate-500">Lat Max</label>
              <input
                type="number"
                value={config.lat_max}
                onChange={(e) => setConfig({ ...config, lat_max: parseFloat(e.target.value) })}
                className="input w-full text-sm"
                step="0.1"
              />
            </div>
            <div>
              <label className="text-xs text-slate-500">Lon Min</label>
              <input
                type="number"
                value={config.lon_min}
                onChange={(e) => setConfig({ ...config, lon_min: parseFloat(e.target.value) })}
                className="input w-full text-sm"
                step="0.1"
              />
            </div>
            <div>
              <label className="text-xs text-slate-500">Lon Max</label>
              <input
                type="number"
                value={config.lon_max}
                onChange={(e) => setConfig({ ...config, lon_max: parseFloat(e.target.value) })}
                className="input w-full text-sm"
                step="0.1"
              />
            </div>
          </div>
        </div>

        {/* Time Range */}
        <div>
          <label className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-1">
            <Calendar className="w-4 h-4" />
            Time Range
          </label>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-500">Start</label>
              <input
                type="date"
                value={config.start_date}
                min={sourceInfo.min_date}
                onChange={(e) => setConfig({ ...config, start_date: e.target.value })}
                className="input w-full"
              />
            </div>
            <div>
              <label className="text-xs text-slate-500">End</label>
              <input
                type="date"
                value={config.end_date}
                onChange={(e) => setConfig({ ...config, end_date: e.target.value })}
                className="input w-full"
              />
            </div>
          </div>
        </div>

        {/* Resolution */}
        <div>
          <label className="text-sm font-medium text-slate-700 mb-2 flex items-center gap-1">
            <Settings2 className="w-4 h-4" />
            Resolution
          </label>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-500">Temporal</label>
              <select
                value={config.temporal_resolution}
                onChange={(e) => setConfig({ ...config, temporal_resolution: e.target.value })}
                className="input w-full"
              >
                {TEMPORAL_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-slate-500">Spatial</label>
              <select
                value={config.spatial_resolution}
                onChange={(e) => setConfig({ ...config, spatial_resolution: e.target.value })}
                className="input w-full"
              >
                {SPATIAL_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Estimate & Download */}
        <div className="pt-4 border-t">
          <div className="flex items-center justify-between mb-3">
            <div>
              <span className="text-sm text-slate-500">Estimated size:</span>
              <span className="ml-2 font-semibold text-slate-700">{estimateSize()}</span>
            </div>
            <span className="text-xs text-slate-400">
              {config.variables.length} variable(s)
            </span>
          </div>
          
          <button
            onClick={downloadData}
            disabled={isDownloading || config.variables.length === 0}
            className={clsx(
              "w-full py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors",
              isDownloading || config.variables.length === 0
                ? "bg-slate-200 text-slate-400 cursor-not-allowed"
                : "bg-blue-600 text-white hover:bg-blue-700"
            )}
          >
            {isDownloading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Downloading...
              </>
            ) : (
              <>
                <Download className="w-5 h-5" />
                Download Data
              </>
            )}
          </button>
        </div>

        {/* Result */}
        {downloadResult && (
          <div className={clsx(
            "p-3 rounded-lg text-sm",
            downloadResult.success ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"
          )}>
            {downloadResult.message}
            {downloadResult.file_path && (
              <p className="text-xs mt-1 font-mono opacity-75">{downloadResult.file_path}</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
