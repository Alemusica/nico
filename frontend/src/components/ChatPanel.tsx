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

// Briefing Card Component
function BriefingCard({ 
  briefing, 
  onConfirm, 
  onCancel,
  isLoading 
}: { 
  briefing: InvestigationBriefingData
  onConfirm: () => void
  onCancel: () => void
  isLoading: boolean
}) {
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
            onClick={onConfirm}
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
  const { messages, addMessage, clearMessages, causalGraph, selectedLink } = useStore()
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

  const handleInvestigation = useCallback((query: string) => {
    setIsInvestigating(true)
    setProgressSteps([])
    setPendingBriefing(null)
    
    const cleanup = investigateStreaming(
      { query },
      // On progress
      (progress: InvestigateProgress) => {
        setProgressSteps(prev => [...prev, progress as ProgressStep])
      },
      // On complete
      (result: InvestigateResponse) => {
        setIsInvestigating(false)
        
        // Format investigation response
        let response = '🕵️ **Investigation Complete**\n\n'
        response += '📍 **Location**: ' + (result.location || 'Unknown') + '\n'
        response += '📅 **Period**: ' + (result.time_range || 'Unknown') + '\n'
        response += '🎯 **Event Type**: ' + (result.event_type || 'Unknown') + '\n'
        response += '📊 **Data Sources**: ' + (result.data_sources_count || 0) + '\n'
        response += '📚 **Papers Found**: ' + (result.papers_found || 0) + '\n'
        response += '🎯 **Confidence**: ' + ((result.confidence || 0) * 100).toFixed(0) + '%\n\n'
        
        if (result.key_findings && result.key_findings.length > 0) {
          response += '### 🔍 Key Findings:\n'
          result.key_findings.forEach((finding, i) => {
            response += (i + 1) + '. ' + finding + '\n'
          })
          response += '\n'
        }
        
        if (result.recommendations && result.recommendations.length > 0) {
          response += '### 💡 Recommendations:\n'
          result.recommendations.forEach((rec, i) => {
            response += (i + 1) + '. ' + rec + '\n'
          })
        }
        
        addMessage('assistant', response)
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

  // Confirm briefing and start investigation
  const confirmBriefing = () => {
    if (pendingQuery) {
      setIsLoading(true)
      handleInvestigation(pendingQuery)
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
                <p className="text-phi-sm whitespace-pre-wrap">{message.content}</p>
                <span className="text-phi-xs opacity-60 mt-1 block">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
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
