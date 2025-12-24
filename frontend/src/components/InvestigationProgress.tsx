/**
 * Investigation Progress Component
 * Shows real-time progress of investigation with step indicators
 */

import { CheckCircle, Circle, Loader2, AlertCircle, Download, Globe, FileSearch, Brain, Search, FileText } from 'lucide-react'
import clsx from 'clsx'

export interface ProgressStep {
  step: number | string
  substep?: string
  status: 'started' | 'progress' | 'complete' | 'error'
  message: string
  data?: Record<string, unknown>
  progress?: number
  result?: Record<string, unknown>
}

interface InvestigationProgressProps {
  steps: ProgressStep[]
  isComplete: boolean
}

const STEP_CONFIG = [
  { id: 1, name: 'Parse Query', icon: FileSearch, color: 'blue' },
  { id: 2, name: 'Resolve Location', icon: Globe, color: 'green' },
  { id: 3, name: 'Collect Data', icon: Download, color: 'purple' },
  { id: 4, name: 'Correlation Analysis', icon: Brain, color: 'orange' },
  { id: 5, name: 'Expand Search', icon: Search, color: 'cyan' },
  { id: 6, name: 'Generate Findings', icon: FileText, color: 'emerald' },
]

const SUBSTEP_ICONS: Record<string, string> = {
  satellite: '📡',
  era5: '🌤️',
  indices: '🌍',
  papers: '📚',
}

export function InvestigationProgress({ steps, isComplete }: InvestigationProgressProps) {
  // Get current step status
  const getStepStatus = (stepId: number) => {
    const stepEvents = steps.filter(s => s.step === stepId)
    if (stepEvents.length === 0) return 'pending'
    
    const last = stepEvents[stepEvents.length - 1]
    if (last.status === 'complete') return 'complete'
    if (last.status === 'error') return 'error'
    if (last.status === 'started' || last.status === 'progress') return 'active'
    return 'pending'
  }

  // Get substeps for step 3 (data collection)
  const getSubsteps = () => {
    return steps.filter(s => s.step === 3 && s.substep)
  }

  // Get latest message for a step
  const getStepMessage = (stepId: number) => {
    const stepEvents = steps.filter(s => s.step === stepId)
    if (stepEvents.length === 0) return ''
    return stepEvents[stepEvents.length - 1].message
  }

  return (
    <div className="bg-slate-50 rounded-lg p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <div className={clsx(
          "w-2 h-2 rounded-full",
          isComplete ? "bg-green-500" : "bg-blue-500 animate-pulse"
        )} />
        <span className="text-sm font-medium text-slate-700">
          {isComplete ? '✅ Investigation Complete' : '🔍 Investigation in Progress...'}
        </span>
      </div>

      {/* Step Progress */}
      <div className="space-y-2">
        {STEP_CONFIG.map((config) => {
          const status = getStepStatus(config.id)
          const Icon = config.icon
          const message = getStepMessage(config.id)
          
          return (
            <div key={config.id} className="flex items-start gap-3">
              {/* Status Icon */}
              <div className={clsx(
                "w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5",
                status === 'complete' && "bg-green-100 text-green-600",
                status === 'active' && "bg-blue-100 text-blue-600",
                status === 'error' && "bg-red-100 text-red-600",
                status === 'pending' && "bg-slate-100 text-slate-400",
              )}>
                {status === 'complete' && <CheckCircle className="w-4 h-4" />}
                {status === 'active' && <Loader2 className="w-4 h-4 animate-spin" />}
                {status === 'error' && <AlertCircle className="w-4 h-4" />}
                {status === 'pending' && <Circle className="w-4 h-4" />}
              </div>

              {/* Step Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <Icon className={clsx(
                    "w-4 h-4",
                    status === 'pending' ? "text-slate-400" : "text-slate-600"
                  )} />
                  <span className={clsx(
                    "text-sm font-medium",
                    status === 'pending' ? "text-slate-400" : "text-slate-700"
                  )}>
                    {config.name}
                  </span>
                </div>
                
                {/* Message */}
                {message && status !== 'pending' && (
                  <p className="text-xs text-slate-500 mt-0.5 truncate">
                    {message}
                  </p>
                )}

                {/* Substeps for Step 3 */}
                {config.id === 3 && status !== 'pending' && (
                  <div className="mt-2 pl-2 border-l-2 border-slate-200 space-y-1">
                    {getSubsteps().map((sub, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-xs">
                        <span>{SUBSTEP_ICONS[sub.substep!] || '•'}</span>
                        <span className={clsx(
                          sub.status === 'complete' ? "text-green-600" :
                          sub.status === 'error' ? "text-red-600" :
                          sub.status === 'started' ? "text-blue-600" :
                          "text-slate-500"
                        )}>
                          {sub.message}
                        </span>
                        {sub.status === 'started' && (
                          <Loader2 className="w-3 h-3 animate-spin text-blue-500" />
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      {/* Progress Bar */}
      {!isComplete && (
        <div className="mt-4">
          <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-blue-500 to-emerald-500 transition-all duration-500"
              style={{ 
                width: `${Math.min(100, (steps.filter(s => s.status === 'complete').length / 8) * 100)}%` 
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
