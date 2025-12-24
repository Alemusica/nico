import { useState } from 'react'
import { Play, Database, FileText, TrendingUp, AlertCircle } from 'lucide-react'
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
    raw_result?: any
  }
  onRunAnalysis?: () => void
  onViewPapers?: () => void
  onViewData?: () => void
}

export function InvestigationResult({ 
  result, 
  onRunAnalysis,
  onViewPapers,
  onViewData
}: InvestigationResultProps) {
  const [showDetails, setShowDetails] = useState(false)

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
          className="btn btn-primary flex items-center justify-center gap-phi-sm"
        >
          <Play className="w-4 h-4" />
          <span className="text-phi-sm">Run Analysis</span>
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
