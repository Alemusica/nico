/**
 * Investigation Briefing Component
 * 
 * Shows data request plan before download and allows user to confirm/modify.
 */

import { useState } from 'react'
import clsx from 'clsx'
import { 
  MapPin, Calendar, Target, Database, Download,
  CheckCircle, Clock, Edit2, X,
  ChevronDown, ChevronUp
} from 'lucide-react'

export interface DataRequest {
  source: string
  variables: string[]
  lat_range: [number, number]
  lon_range: [number, number]
  time_range: [string, string]
  resolution: {
    temporal: string
    spatial: string
  }
  description: string
  estimated_size_mb: number
  estimated_time_sec: number
}

export interface Briefing {
  query: string
  location_name: string
  location_bbox: [number, number, number, number]
  event_type: string
  time_period: [string, string]
  precursor_period: [string, string]
  data_requests: DataRequest[]
  total_estimated_size_mb: number
  total_estimated_time_sec: number
  cached_sources: string[]
  needs_download: string[]
}

interface InvestigationBriefingProps {
  briefing: Briefing
  briefingId: string
  onConfirm: (briefingId: string, resolution?: { temporal: string; spatial: string }) => void
  onCancel: () => void
  onModify?: (briefingId: string) => void
}

const TEMPORAL_OPTIONS = [
  { value: 'hourly', label: 'Hourly' },
  { value: '6-hourly', label: '6-hourly' },
  { value: 'daily', label: 'Daily' },
]

const SPATIAL_OPTIONS = [
  { value: '0.1', label: '0.1°' },
  { value: '0.25', label: '0.25°' },
  { value: '0.5', label: '0.5°' },
]

export function InvestigationBriefing({
  briefing,
  briefingId,
  onConfirm,
  onCancel,
  onModify: _onModify,
}: InvestigationBriefingProps) {
  // onModify is available for future use
  void _onModify;
  const [expandedRequest, setExpandedRequest] = useState<string | null>(null)
  const [modifiedResolution, setModifiedResolution] = useState<{ temporal: string; spatial: string } | null>(null)
  const [showResolutionEditor, setShowResolutionEditor] = useState(false)

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`
  }

  const formatSize = (mb: number): string => {
    if (mb < 1) return `${Math.round(mb * 1024)} KB`
    return `${mb.toFixed(1)} MB`
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden max-w-2xl">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Database className="w-5 h-5" />
          Investigation Briefing
        </h3>
        <p className="text-blue-100 text-sm mt-1">Review and confirm data request</p>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Location & Event */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-start gap-2">
            <MapPin className="w-4 h-4 text-green-500 mt-0.5" />
            <div>
              <p className="text-xs text-slate-500">Location</p>
              <p className="font-medium">{briefing.location_name}</p>
              <p className="text-xs text-slate-400 font-mono">
                {briefing.location_bbox[0].toFixed(2)}°-{briefing.location_bbox[1].toFixed(2)}°N,{' '}
                {briefing.location_bbox[2].toFixed(2)}°-{briefing.location_bbox[3].toFixed(2)}°E
              </p>
            </div>
          </div>
          
          <div className="flex items-start gap-2">
            <Target className="w-4 h-4 text-orange-500 mt-0.5" />
            <div>
              <p className="text-xs text-slate-500">Event Type</p>
              <p className="font-medium capitalize">{briefing.event_type}</p>
            </div>
          </div>
        </div>

        {/* Time Periods */}
        <div className="flex items-start gap-2">
          <Calendar className="w-4 h-4 text-purple-500 mt-0.5" />
          <div className="flex-1">
            <p className="text-xs text-slate-500">Analysis Period</p>
            <p className="font-medium">
              {briefing.precursor_period[0]} → {briefing.precursor_period[1]}
            </p>
            <p className="text-xs text-slate-400">
              Event: {briefing.time_period[0]} to {briefing.time_period[1]}
            </p>
          </div>
        </div>

        {/* Data Requests */}
        <div className="border-t pt-4">
          <h4 className="text-sm font-semibold text-slate-700 mb-2">Data Requests</h4>
          
          <div className="space-y-2">
            {briefing.data_requests.map((req, idx) => {
              const isCached = briefing.cached_sources.includes(req.source)
              const isExpanded = expandedRequest === req.source
              
              return (
                <div 
                  key={idx}
                  className={clsx(
                    "border rounded-lg overflow-hidden",
                    isCached ? "border-green-200 bg-green-50/30" : "border-slate-200"
                  )}
                >
                  <div 
                    className="flex items-center justify-between p-3 cursor-pointer"
                    onClick={() => setExpandedRequest(isExpanded ? null : req.source)}
                  >
                    <div className="flex items-center gap-2">
                      {isCached ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <Download className="w-4 h-4 text-blue-500" />
                      )}
                      <span className="font-medium">{req.source}</span>
                      <span className="text-xs text-slate-400">
                        {req.variables.length} variables
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-3">
                      {isCached ? (
                        <span className="text-xs text-green-600 font-medium">Cached</span>
                      ) : (
                        <span className="text-xs text-slate-500">
                          {formatSize(req.estimated_size_mb)} • {formatTime(req.estimated_time_sec)}
                        </span>
                      )}
                      {isExpanded ? (
                        <ChevronUp className="w-4 h-4 text-slate-400" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-slate-400" />
                      )}
                    </div>
                  </div>
                  
                  {isExpanded && (
                    <div className="px-3 pb-3 border-t border-slate-100 bg-slate-50/50">
                      <div className="grid grid-cols-2 gap-3 mt-2 text-xs">
                        <div>
                          <p className="text-slate-500">Variables</p>
                          <p className="font-mono">{req.variables.join(', ')}</p>
                        </div>
                        <div>
                          <p className="text-slate-500">Resolution</p>
                          <p className="font-mono">
                            {req.resolution.temporal}, {req.resolution.spatial}°
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {/* Summary */}
        <div className="border-t pt-4">
          {briefing.needs_download.length > 0 ? (
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-blue-500" />
                <span className="text-sm">
                  Total download: <strong>{formatSize(briefing.total_estimated_size_mb)}</strong>,{' '}
                  ~<strong>{formatTime(briefing.total_estimated_time_sec)}</strong>
                </span>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2 p-3 bg-green-50 rounded-lg text-green-700">
              <CheckCircle className="w-4 h-4" />
              <span className="text-sm font-medium">All data cached - instant analysis!</span>
            </div>
          )}
        </div>

        {/* Resolution Modifier */}
        {showResolutionEditor && (
          <div className="border-t pt-4">
            <h4 className="text-sm font-semibold text-slate-700 mb-2">Modify Resolution</h4>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-slate-500">Temporal</label>
                <select
                  className="input w-full mt-1"
                  value={modifiedResolution?.temporal || briefing.data_requests[0]?.resolution.temporal}
                  onChange={(e) => setModifiedResolution({
                    temporal: e.target.value,
                    spatial: modifiedResolution?.spatial || briefing.data_requests[0]?.resolution.spatial || '0.25',
                  })}
                >
                  {TEMPORAL_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-slate-500">Spatial</label>
                <select
                  className="input w-full mt-1"
                  value={modifiedResolution?.spatial || briefing.data_requests[0]?.resolution.spatial}
                  onChange={(e) => setModifiedResolution({
                    temporal: modifiedResolution?.temporal || briefing.data_requests[0]?.resolution.temporal || 'daily',
                    spatial: e.target.value,
                  })}
                >
                  {SPATIAL_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between p-4 bg-slate-50 border-t">
        <button
          onClick={() => setShowResolutionEditor(!showResolutionEditor)}
          className="btn btn-ghost text-slate-600"
        >
          <Edit2 className="w-4 h-4 mr-1" />
          {showResolutionEditor ? 'Hide Options' : 'Modify'}
        </button>
        
        <div className="flex gap-2">
          <button onClick={onCancel} className="btn btn-ghost">
            <X className="w-4 h-4 mr-1" />
            Cancel
          </button>
          <button 
            onClick={() => onConfirm(briefingId, modifiedResolution || undefined)}
            className="btn btn-primary"
          >
            <CheckCircle className="w-4 h-4 mr-1" />
            Confirm & Start
          </button>
        </div>
      </div>
    </div>
  )
}
