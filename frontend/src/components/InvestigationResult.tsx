import { useState } from 'react'
import { Play, Database, FileText, TrendingUp, AlertCircle, Zap } from 'lucide-react'
import clsx from 'clsx'

interface InvestigationResultProps {
  result: {
    location?: string
    time_range?: string
    event_type?: string
    data_sources_count?: number
    papers_found?: number
    confidence?: number
    key_findings?: string[]
    recommendations?: string[]
    correlations?: any[]
    raw_result?: any
  }
  onRunAnalysis?: () => void
  onViewPapers?: () => void
  onViewData?: () => void
}

// Auto-suggest variables based on event type
const EVENT_VARIABLES: Record<string, string[]> = {
  flood: ['precipitation', 'river_discharge', 'soil_moisture', 'temperature', 'snowmelt'],
  drought: ['soil_moisture', 'temperature', 'evapotranspiration', 'precipitation', 'vegetation_index'],
  storm_surge: ['sea_level', 'wind_speed', 'atmospheric_pressure', 'wave_height', 'tides'],
  heatwave: ['temperature', 'humidity', 'solar_radiation', 'wind_speed', 'heat_index'],
  extreme_precipitation: ['precipitation', 'atmospheric_moisture', 'temperature', 'wind_patterns'],
}

// Suggested max lags based on event type (in days)
const EVENT_LAGS: Record<string, number> = {
  flood: 14,  // Floods typically have 1-2 week precursors
  drought: 90,  // Droughts develop over months
  storm_surge: 7,  // Storm surges have short-term precursors
  heatwave: 21,  // Heatwaves build over weeks
  extreme_precipitation: 7,
}

export function InvestigationResult({ 
  result, 
  onRunAnalysis,
  onViewPapers,
  onViewData
}: InvestigationResultProps) {
  const [showDetails, setShowDetails] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(false)
  
  // Get suggested variables and lag for this event type
  const eventType = result.event_type?.toLowerCase() || 'flood'
  const suggestedVariables = EVENT_VARIABLES[eventType] || EVENT_VARIABLES.flood
  const suggestedLag = EVENT_LAGS[eventType] || 14
  
  // Check if we have correlations to preview
  const hasCorrelations = result.correlations && result.correlations.length > 0

  return (
    <div className="space-y-phi-md">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-phi-lg font-semibold text-slate-800 flex items-center gap-phi-sm">
            🕵️ Investigation Complete
          </h3>
          <p className="text-phi-sm text-slate-600 mt-1">
            Analysis ready for next steps
          </p>
        </div>
        {result.confidence && (
          <div className="px-phi-md py-phi-sm bg-green-100 text-green-700 rounded-full text-phi-sm font-medium">
            {(result.confidence * 100).toFixed(0)}% Confidence
          </div>
        )}
      </div>

      {/* Summary Grid */}
      <div className="grid grid-cols-2 gap-phi-md">
        <div className="space-y-1">
          <div className="text-phi-xs text-slate-500">Location</div>
          <div className="text-phi-sm font-medium text-slate-800">
            📍 {result.location || 'Unknown'}
          </div>
        </div>
        
        <div className="space-y-1">
          <div className="text-phi-xs text-slate-500">Event Type</div>
          <div className="text-phi-sm font-medium text-slate-800">
            🎯 {result.event_type || 'Unknown'}
          </div>
        </div>
        
        <div className="space-y-1">
          <div className="text-phi-xs text-slate-500">Time Period</div>
          <div className="text-phi-sm font-medium text-slate-800">
            📅 {result.time_range || 'Unknown'}
          </div>
        </div>
        
        <div className="space-y-1">
          <div className="text-phi-xs text-slate-500">Data Sources</div>
          <div className="text-phi-sm font-medium text-slate-800">
            📊 {result.data_sources_count || 0} sources collected
          </div>
        </div>
      </div>

      {/* Papers Found */}
      {result.papers_found && result.papers_found > 0 && (
        <div className="p-phi-md bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center gap-phi-sm text-phi-sm text-blue-800">
            <FileText className="w-4 h-4" />
            <span className="font-medium">{result.papers_found} scientific papers</span>
            <span className="text-blue-600">saved to knowledge base</span>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="grid grid-cols-3 gap-phi-sm">
        <button
          onClick={onRunAnalysis}
          className="btn btn-primary flex items-center justify-center gap-phi-sm relative group"
        >
          <Play className="w-4 h-4" />
          <span className="text-phi-sm">Run Analysis</span>
          {suggestedVariables.length > 0 && (
            <Zap className="w-3 h-3 text-yellow-300 absolute -top-1 -right-1 animate-pulse" />
          )}
        </button>
        
        <button
          onClick={onViewData}
          className="btn btn-secondary flex items-center justify-center gap-phi-sm"
        >
          <Database className="w-4 h-4" />
          <span className="text-phi-sm">View Data</span>
        </button>
        
        {result.papers_found && result.papers_found > 0 && (
          <button
            onClick={onViewPapers}
            className="btn btn-secondary flex items-center justify-center gap-phi-sm"
          >
            <FileText className="w-4 h-4" />
            <span className="text-phi-sm">Papers</span>
          </button>
        )}
      </div>

      {/* Analysis Suggestions */}
      {suggestedVariables.length > 0 && (
        <div className="border border-blue-200 rounded-lg p-phi-md bg-gradient-to-br from-blue-50 to-slate-50">
          <button
            onClick={() => setShowSuggestions(!showSuggestions)}
            className="flex items-center gap-phi-sm text-phi-sm font-medium text-blue-700 hover:text-blue-900 w-full"
          >
            <Zap className="w-4 h-4" />
            <span>Smart Analysis Suggestions</span>
            <span className="ml-auto text-phi-xs">{showSuggestions ? '▲' : '▼'}</span>
          </button>
          
          {showSuggestions && (
            <div className="mt-phi-md space-y-phi-sm">
              <div>
                <div className="text-phi-xs text-slate-600 mb-1">Suggested Variables for {result.event_type}:</div>
                <div className="flex flex-wrap gap-1">
                  {suggestedVariables.map(v => (
                    <span key={v} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-phi-xs font-medium">
                      {v}
                    </span>
                  ))}
                </div>
              </div>
              
              <div className="pt-phi-sm border-t border-blue-200">
                <div className="text-phi-xs text-slate-600 mb-1">Recommended Max Lag:</div>
                <div className="px-2 py-1 bg-emerald-100 text-emerald-800 rounded text-phi-xs font-medium inline-block">
                  {suggestedLag} days ({Math.round(suggestedLag/7)} weeks)
                </div>
                <p className="text-phi-xs text-slate-500 mt-1">
                  {eventType === 'flood' && 'Floods typically show precursors 1-2 weeks before'}
                  {eventType === 'drought' && 'Droughts develop gradually over months'}
                  {eventType === 'storm_surge' && 'Storm surges have short-term atmospheric precursors'}
                  {eventType === 'heatwave' && 'Heatwaves build up over 2-3 weeks'}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Correlation Preview */}
      {hasCorrelations && (
        <div className="border border-emerald-200 rounded-lg p-phi-md bg-gradient-to-br from-emerald-50 to-slate-50">
          <div className="flex items-center gap-phi-sm text-phi-sm font-medium text-emerald-700 mb-phi-sm">
            <TrendingUp className="w-4 h-4" />
            <span>Preliminary Correlations Found</span>
          </div>
          <div className="space-y-1">
            {result.correlations.slice(0, 3).map((corr: any, i: number) => (
              <div key={i} className="flex items-center justify-between text-phi-xs">
                <span className="text-slate-700">
                  {corr.index || corr.type}: {corr.interpretation || corr.message}
                </span>
                {corr.favorable_for_flood !== undefined && (
                  <span className={clsx(
                    'px-2 py-0.5 rounded text-phi-xs',
                    corr.favorable_for_flood ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                  )}>
                    {corr.favorable_for_flood ? 'High Risk' : 'Low Risk'}
                  </span>
                )}
              </div>
            ))}
          </div>
          <p className="text-phi-xs text-slate-500 mt-phi-sm">
            Run full PCMCI analysis for robust causal inference
          </p>
        </div>
      )}

      {/* Key Findings */}
      {result.key_findings && result.key_findings.length > 0 && (
        <div className="space-y-phi-sm">
          <div className="text-phi-sm font-medium text-slate-700">🔍 Key Findings</div>
          <div className="space-y-phi-xs">
            {result.key_findings.map((finding, i) => (
              <div key={i} className="flex gap-phi-sm text-phi-sm text-slate-600">
                <span className="text-slate-400">{i + 1}.</span>
                <span>{finding}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {result.recommendations && result.recommendations.length > 0 && (
        <div className="space-y-phi-sm">
          <div className="text-phi-sm font-medium text-slate-700">💡 Recommendations</div>
          <div className="space-y-phi-xs">
            {result.recommendations.map((rec, i) => (
              <div key={i} className="flex gap-phi-sm text-phi-sm text-slate-600">
                <span className="text-slate-400">{i + 1}.</span>
                <span>{rec}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Details Toggle */}
      <button
        onClick={() => setShowDetails(!showDetails)}
        className="text-phi-sm text-slate-500 hover:text-slate-700 transition-colors"
      >
        {showDetails ? '▲ Hide' : '▼ Show'} technical details
      </button>

      {showDetails && result.raw_result && (
        <pre className="p-phi-md bg-slate-100 rounded-lg text-phi-xs text-slate-700 overflow-auto max-h-64">
          {JSON.stringify(result.raw_result, null, 2)}
        </pre>
      )}
    </div>
  )
}
