/**
 * Chat Panel Component - Swiss Design
 * AI Assistant for Causal Discovery
 */

import { useState, useRef, useEffect } from 'react'
import { Send, Sparkles, RefreshCw } from 'lucide-react'
import clsx from 'clsx'
import { useStore } from '../store'
import { chat } from '../api'

interface ChatPanelProps {
  expanded?: boolean
}

const suggestions = [
  'Explain the NAO → SST relationship',
  'What patterns suggest ocean warming?',
  'Find similar events in history',
  'Why is there a 3-day lag?',
]

export function ChatPanel({ expanded = false }: ChatPanelProps) {
  const { messages, addMessage, clearMessages, causalGraph, selectedLink } = useStore()
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    addMessage('user', userMessage)
    setIsLoading(true)

    try {
      // Build context from current state
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
    } catch (error) {
      addMessage('assistant', 'Sorry, I encountered an error. Please try again.')
    } finally {
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
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-phi-xl">
            <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center mb-phi-lg">
              <Sparkles className="w-8 h-8 text-slate-400" />
            </div>
            <h5 className="text-phi-base font-medium text-slate-700 mb-phi-sm">
              How can I help?
            </h5>
            <p className="text-phi-sm text-slate-500 mb-phi-lg">
              Ask me about your causal discoveries, data patterns, or oceanographic phenomena.
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
          messages.map((message) => (
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
          ))
        )}
        
        {isLoading && (
          <div className="chat-message chat-message-assistant animate-slide-up">
            <div className="flex items-center gap-phi-sm">
              <div className="spinner" />
              <span className="text-phi-sm text-slate-500">Thinking...</span>
            </div>
          </div>
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
          placeholder="Ask about your data..."
          className="input flex-1"
          disabled={isLoading}
        />
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
