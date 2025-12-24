/**
 * Chat Panel Component - Swiss Design
 * AI Assistant for Causal Discovery with WebSocket Streaming
 * Now includes Investigation Briefing before download
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Sparkles, RefreshCw, Globe, Database, MapPin, Clock, Download, X, Check } from 'lucide-react'
import clsx from 'clsx'
import { useStore } from '../store'
import { chat, investigateStreaming, isInvestigationQuery, createInvestigationBriefing } from '../api'
import type { InvestigateProgress, InvestigateResponse, InvestigationBriefingData } from '../api'
import { InvestigationProgress } from './InvestigationProgress'
import { InvestigationResult } from './InvestigationResult'
import type { ProgressStep } from './InvestigationProgress'

interface ChatPanelProps {
  expanded?: boolean
}

const suggestions = [
  'Analizza alluvioni Lago Maggiore 2000',
  'Investigate floods Po Valley 1994',
  'Explain the NAO → SST relationship',
  'Find precursors for Venice acqua alta',
]

const TEMPORAL_OPTIONS = [
  { value: 'hourly', label: 'Hourly', desc: 'High detail, large download' },
  { value: '6-hourly', label: '6-hourly', desc: 'Good balance' },
  { value: 'daily', label: 'Daily', desc: 'Fast download' },
]

const SPATIAL_OPTIONS = [
  { value: '0.1', label: '0.1°', desc: '~11 km, high detail' },
  { value: '0.25', label: '0.25°', desc: '~25 km, default' },
  { value: '0.5', label: '0.5°', desc: '~50 km, fast' },
]

export interface ResolutionConfig {
  temporal: string
  spatial: string
}

// Briefing Card Component with Resolution Picker
function BriefingCard({ 
  briefing, 
  onConfirm, 
  onCancel,
  isLoading 
}: { 
  briefing: InvestigationBriefingData
  onConfirm: (resolution?: ResolutionConfig) => void
  onCancel: () => void
  isLoading: boolean
}) {
  const [showOptions, setShowOptions] = useState(false)
  const [resolution, setResolution] = useState<ResolutionConfig>({ 
    temporal: 'daily', 
    spatial: '0.25' 
  })
  
  const formatTime = (sec: number) => sec < 60 ? `${Math.round(sec)}s` : `${Math.floor(sec/60)}m ${Math.round(sec%60)}s`
  
  return (
    <div className="bg-gradient-to-br from-blue-50 to-slate-50 rounded-xl border border-blue-200 overflow-hidden">
      {/* Header */}
      <div className="bg-blue-600 text-white px-4 py-3">
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5" />
          <span className="font-semibold">Investigation Briefing</span>
        </div>
        <p className="text-blue-100 text-xs mt-1">Review data to be downloaded</p>
      </div>
      
      {/* Content */}
      <div className="p-4 space-y-3">
        {/* Location */}
        {briefing.location && (
          <div className="flex items-start gap-2">
            <MapPin className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-medium text-slate-800">{briefing.location.name}</p>
              <p className="text-xs text-slate-500">
                {briefing.location.region}, {briefing.location.country} • 
                {briefing.location.lat.toFixed(2)}°N, {briefing.location.lon.toFixed(2)}°E
              </p>
            </div>
          </div>
        )}
        
        {/* Period */}
        <div className="flex items-start gap-2">
          <Clock className="w-4 h-4 text-purple-600 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-medium text-slate-800">
              {briefing.event_context.start_date} → {briefing.event_context.end_date}
            </p>
            <p className="text-xs text-slate-500 capitalize">{briefing.event_context.event_type}</p>
          </div>
        </div>
        
        {/* Data Sources */}
        <div className="border-t pt-3 mt-3">
          <p className="text-xs font-semibold text-slate-600 mb-2">📦 Data Sources</p>
          <div className="space-y-1.5">
            {briefing.data_requests.map((req, i) => (
              <div key={i} className="flex items-center justify-between text-sm bg-white rounded px-2 py-1.5 border border-slate-100">
                <div className="flex items-center gap-2">
                  <Download className="w-3.5 h-3.5 text-blue-500" />
                  <span className="font-medium text-slate-700">{req.name}</span>
                </div>
                <span className="text-xs text-slate-400">
                  {req.estimated_size_mb > 0.1 ? `${req.estimated_size_mb.toFixed(1)} MB` : ''} 
                  ~{formatTime(req.estimated_time_sec)}
                </span>
              </div>
            ))}
          </div>
        </div>
        
        {/* Resolution Picker (collapsible) */}
        <div className="border-t pt-3">
          <button 
            onClick={() => setShowOptions(!showOptions)}
            className="flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-800"
          >
            ⚙️ {showOptions ? 'Hide' : 'Configure'} Resolution
          </button>
          
          {showOptions && (
            <div className="mt-2 grid grid-cols-2 gap-3 p-3 bg-white rounded-lg border border-slate-200">
              {/* Temporal */}
              <div>
                <label className="text-xs text-slate-500 font-medium mb-1 block">
                  ⏱️ Temporal
                </label>
                <select
                  value={resolution.temporal}
                  onChange={(e) => setResolution({ ...resolution, temporal: e.target.value })}
                  className="w-full px-2 py-1.5 text-sm border border-slate-200 rounded focus:ring-2 focus:ring-blue-500"
                >
                  {TEMPORAL_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                <p className="text-xs text-slate-400 mt-0.5">
                  {TEMPORAL_OPTIONS.find(o => o.value === resolution.temporal)?.desc}
                </p>
              </div>
              
              {/* Spatial */}
              <div>
                <label className="text-xs text-slate-500 font-medium mb-1 block">
                  🗺️ Spatial
                </label>
                <select
                  value={resolution.spatial}
                  onChange={(e) => setResolution({ ...resolution, spatial: e.target.value })}
                  className="w-full px-2 py-1.5 text-sm border border-slate-200 rounded focus:ring-2 focus:ring-blue-500"
                >
                  {SPATIAL_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                <p className="text-xs text-slate-400 mt-0.5">
                  {SPATIAL_OPTIONS.find(o => o.value === resolution.spatial)?.desc}
                </p>
              </div>
            </div>
          )}
        </div>
        
        {/* Total */}
        <div className="bg-blue-100 rounded-lg p-2 text-center">
          <p className="text-sm font-semibold text-blue-800">
            ⏱️ Estimated: ~{formatTime(briefing.total_estimated_time_sec)}
          </p>
        </div>
        
        {/* Actions */}
        <div className="flex gap-2 pt-2">
          <button 
            onClick={onCancel}
            disabled={isLoading}
            className="flex-1 flex items-center justify-center gap-1 px-3 py-2 text-sm font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors disabled:opacity-50"
          >
            <X className="w-4 h-4" />
            Cancel
          </button>
          <button 
            onClick={() => onConfirm(resolution)}
            disabled={isLoading}
            className="flex-1 flex items-center justify-center gap-1 px-3 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Starting...
              </>
            ) : (
              <>
                <Check className="w-4 h-4" />
                Confirm & Start
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

export function ChatPanel({ expanded = false }: ChatPanelProps) {
  const { 
    messages, addMessage, clearMessages, causalGraph, selectedLink, 
    setPendingInvestigationResult, setActiveView, setKnowledgeSearchQuery 
  } = useStore()
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isInvestigating, setIsInvestigating] = useState(false)
  const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([])
  const [pendingBriefing, setPendingBriefing] = useState<InvestigationBriefingData | null>(null)
  const [pendingQuery, setPendingQuery] = useState<string>('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const cleanupRef = useRef<(() => void) | null>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, progressSteps, pendingBriefing])

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (cleanupRef.current) {
        cleanupRef.current()
      }
    }
  }, [])

  const handleInvestigation = useCallback((query: string, resolution?: ResolutionConfig) => {
    setIsInvestigating(true)
    setProgressSteps([])
    setPendingBriefing(null)
    
    const cleanup = investigateStreaming(
      { 
        query,
        // Pass resolution config if provided
        ...(resolution && {
          temporal_resolution: resolution.temporal,
          spatial_resolution: resolution.spatial,
        })
      },
      // On progress
      (progress: InvestigateProgress) => {
        setProgressSteps(prev => [...prev, progress as ProgressStep])
      },
      // On complete
      (result: InvestigateResponse) => {
        setIsInvestigating(false)
        
        // Save investigation result as message with metadata
        addMessage('assistant', '🕵️ Investigation Complete', {
          type: 'investigation_result',
          investigation_result: result
        })
        
        setProgressSteps([])
        setIsLoading(false)
      },
      // On error
      (error: string) => {
        setIsInvestigating(false)
        setProgressSteps([])
        addMessage('assistant', '❌ Investigation failed: ' + error)
        setIsLoading(false)
      }
    )
    
    cleanupRef.current = cleanup
  }, [addMessage])

  // Request briefing first
  const requestBriefing = async (query: string) => {
    setIsLoading(true)
    try {
      const response = await createInvestigationBriefing(query)
      if (response.status === 'success' && response.briefing) {
        setPendingBriefing(response.briefing)
        setPendingQuery(query)
      } else {
        // Fallback to direct investigation if briefing fails
        handleInvestigation(query)
      }
    } catch (error) {
      console.error('Briefing error:', error)
      // Fallback to direct investigation
      handleInvestigation(query)
    }
    setIsLoading(false)
  }

  // Confirm briefing and start investigation with resolution
  const confirmBriefing = (resolution?: ResolutionConfig) => {
    if (pendingQuery) {
      setIsLoading(true)
      handleInvestigation(pendingQuery, resolution)
    }
  }

  // Cancel briefing
  const cancelBriefing = () => {
    setPendingBriefing(null)
    setPendingQuery('')
    addMessage('assistant', '❌ Investigation cancelled')
  }

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    addMessage('user', userMessage)
    setIsLoading(true)

    try {
      // Check if this is an investigation query
      if (isInvestigationQuery(userMessage)) {
        // First get briefing, then let user confirm
        await requestBriefing(userMessage)
      } else {
        // Use regular chat for non-investigation queries
        const context: Record<string, unknown> = {}
        if (causalGraph) {
          context.graph = {
            variables: causalGraph.variables,
            links_count: causalGraph.links.length,
          }
        }
        if (selectedLink) {
          context.selected_link = selectedLink
        }

        const response = await chat(userMessage, context)
        addMessage('assistant', response.response)
        setIsLoading(false)
      }
    } catch (error) {
      addMessage('assistant', 'Sorry, I encountered an error. Please try again.')
      setIsLoading(false)
    }
  }

  const handleSuggestion = (suggestion: string) => {
    setInput(suggestion)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className={clsx(
      'chat-container',
      expanded ? 'h-[calc(100vh-160px)]' : 'h-[600px]'
    )}>
      {/* Header */}
      <div className="flex items-center justify-between px-phi-lg py-phi-md border-b border-slate-200">
        <div className="flex items-center gap-phi-sm">
          <div className="w-8 h-8 rounded-swiss bg-gradient-to-br from-blue-500 to-emerald-500 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-white" />
          </div>
          <div>
            <h4 className="text-phi-sm font-semibold text-slate-900">AI Assistant</h4>
            <p className="text-phi-xs text-slate-500">Powered by Ollama</p>
          </div>
        </div>
        <button 
          onClick={clearMessages}
          className="btn btn-ghost btn-sm"
          title="Clear chat"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Messages */}
      <div className="chat-messages swiss-scrollbar">
        {messages.length === 0 && !isInvestigating ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-phi-xl">
            <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mb-phi-lg">
              <Sparkles className="w-8 h-8 text-slate-400" />
            </div>
            <h5 className="text-phi-base font-medium text-slate-700 mb-phi-sm">
              How can I help?
            </h5>
            <p className="text-phi-sm text-slate-500 mb-phi-lg">
              Ask me to <strong>investigate events</strong> (e.g., "analizza Lago Maggiore 2000") 
              or ask about causal patterns in your data.
            </p>
            
            {/* Suggestions */}
            <div className="flex flex-wrap gap-phi-sm justify-center">
              {suggestions.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => handleSuggestion(suggestion)}
                  className="px-phi-md py-phi-sm text-phi-xs text-slate-600 bg-slate-100 rounded-full hover:bg-slate-200 transition-colors"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div
                key={message.id}
                className={clsx(
                  'chat-message animate-slide-up',
                  message.role === 'user' ? 'chat-message-user' : 'chat-message-assistant'
                )}
              >
                {/* Render Investigation Result if metadata present */}
                {message.metadata?.type === 'investigation_result' ? (
                  <InvestigationResult 
                    result={message.metadata.investigation_result}
                    onRunAnalysis={() => {
                      const result = message.metadata.investigation_result
                      setPendingInvestigationResult(result)
                      setActiveView('graph')
                      addMessage('assistant', '🔬 Navigated to Graph view for causal analysis!\n\n' +
                        `Ready to analyze ${result.data_sources_count} data sources from ${result.location}.\n` +
                        'Configure PCMCI parameters and run discovery.')
                    }}
                    onViewData={() => {
                      const result = message.metadata.investigation_result
                      setActiveView('knowledge')
                      addMessage('assistant', '📊 Navigated to Data Explorer!\n\n' +
                        `Location: **${result.location}**\n` +
                        `Period: **${result.time_range}**\n` +
                        `Sources: **${result.data_sources_count}** cached datasets\n\n` +
                        'Use the cache entries to explore your investigation data.')
                    }}
                    onViewPapers={() => {
                      const result = message.metadata.investigation_result
                      const query = `${result.event_type} ${result.location}`
                      setKnowledgeSearchQuery(query)
                      setActiveView('knowledge')
                      addMessage('assistant', `📚 Navigated to Knowledge Base with **${result.papers_found} papers**!\n\n` +
                        `Search query: "${query}"\n` +
                        `Event: ${result.event_type} in ${result.location}\n\n` +
                        'Switch to Papers tab to explore scientific literature.')
                    }}
                  />
                ) : (
                  <>
                    <p className="text-phi-sm whitespace-pre-wrap">{message.content}</p>
                    <span className="text-phi-xs opacity-60 mt-1 block">
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </>
                )}
              </div>
            ))}
            
            {/* Investigation Progress */}
            {isInvestigating && progressSteps.length > 0 && (
              <div className="chat-message chat-message-assistant animate-slide-up">
                <InvestigationProgress 
                  steps={progressSteps} 
                  isComplete={false}
                />
              </div>
            )}
            
            {/* Pending Briefing - Show before confirmation */}
            {pendingBriefing && !isInvestigating && (
              <div className="chat-message chat-message-assistant animate-slide-up">
                <BriefingCard
                  briefing={pendingBriefing}
                  onConfirm={confirmBriefing}
                  onCancel={cancelBriefing}
                  isLoading={isLoading}
                />
              </div>
            )}
            
            {/* Loading spinner for non-investigation queries */}
            {isLoading && !isInvestigating && !pendingBriefing && (
              <div className="chat-message chat-message-assistant animate-slide-up">
                <div className="flex items-center gap-phi-sm">
                  <div className="spinner" />
                  <span className="text-phi-sm text-slate-500">Thinking...</span>
                </div>
              </div>
            )}
          </>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat-input-container">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about events or data patterns..."
          className="input flex-1"
          disabled={isLoading}
        />
        {input && isInvestigationQuery(input) && (
          <div className="flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-700 rounded-md text-xs">
            <Globe className="w-3 h-3" />
            <span>Investigation</span>
          </div>
        )}
        <button
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
          className="btn btn-primary"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
