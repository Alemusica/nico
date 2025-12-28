/**
 * 🎮 Knowledge Cockpit View
 * 
 * Layout:
 * ┌────────────────────────────────────────┐
 * │                                        │
 * │       3D Knowledge Graph               │
 * │       (Full width, ~70% height)        │
 * │                                        │
 * ├────────────────────────────────────────┤
 * │  💬 Chat Strip with LLM               │
 * │  Query graph, API, database, etc.     │
 * └────────────────────────────────────────┘
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import SpriteText from 'three-spritetext'
import { useStore } from '../store'
import { 
  Send, 
  Sparkles, 
  RefreshCw, 
  Maximize2, 
  Minimize2,
  Search,
  Filter,
  Database,
  FileText,
  Zap,
  Globe,
  Clock,
  Loader2,
  ChevronUp,
  ChevronDown,
  X,
  Check
} from 'lucide-react'
import clsx from 'clsx'

// ==================== Types ====================

interface GraphNode {
  id: string
  name: string
  type: string
  properties?: Record<string, any>
  x?: number
  y?: number
  z?: number
}

interface GraphLink {
  source: string
  target: string
  type: string
  weight?: number
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
  stats: {
    total_nodes: number
    total_links: number
    papers: number
    events: number
    patterns: number
  }
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  context?: {
    nodeId?: string
    action?: string
  }
}

// ==================== Constants ====================

const NODE_COLORS: Record<string, string> = {
  paper: '#3b82f6',      // blue
  event: '#f59e0b',      // amber
  pattern: '#10b981',    // emerald
  variable: '#8b5cf6',   // violet
  region: '#ec4899',     // pink
  dataset: '#06b6d4',    // cyan
  default: '#64748b'     // slate
}

const QUICK_ACTIONS = [
  { id: 'search', icon: Search, label: 'Search graph', prompt: 'Find nodes related to ' },
  { id: 'analyze', icon: Zap, label: 'Analyze', prompt: 'Analyze the causal relationships for ' },
  { id: 'explain', icon: FileText, label: 'Explain', prompt: 'Explain the connection between ' },
  { id: 'query', icon: Database, label: 'Query DB', prompt: 'Query the database for ' },
]

// ==================== Main Component ====================

export function KnowledgeCockpit() {
  const { backend } = useStore()
  
  // Graph state
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  
  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [chatExpanded, setChatExpanded] = useState(false)
  
  // Refs
  const fgRef = useRef<any>()
  const containerRef = useRef<HTMLDivElement>(null)
  const chatInputRef = useRef<HTMLInputElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

  // ==================== Effects ====================

  // Resize handler
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const chatHeight = chatExpanded ? 300 : 80
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight - chatHeight
        })
      }
    }
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [chatExpanded])

  // Fetch graph data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const res = await fetch(`/api/v1/knowledge/graph?backend=${backend}`)
        if (!res.ok) throw new Error('Failed to fetch graph')
        const data = await res.json()
        setGraphData(data)
        setError(null)
        
        // Welcome message
        if (messages.length === 0) {
          setMessages([{
            id: 'welcome',
            role: 'system',
            content: `🧠 Knowledge Graph loaded: ${data.stats.total_nodes} nodes, ${data.stats.total_links} links. Click on nodes to explore, or ask me anything!`,
            timestamp: new Date()
          }])
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [backend])

  // ==================== Handlers ====================

  // Node click - zoom and show info
  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node)
    
    // Zoom to node
    if (fgRef.current) {
      const distance = 100
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z)
      fgRef.current.cameraPosition(
        { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
        node,
        1000
      )
    }
    
    // Add context to chat
    setMessages(prev => [...prev, {
      id: `node-${Date.now()}`,
      role: 'system',
      content: `📍 Selected: **${node.name}** (${node.type})`,
      timestamp: new Date(),
      context: { nodeId: node.id }
    }])
  }, [])

  // Chat submit
  const handleChatSubmit = async () => {
    if (!input.trim() || chatLoading) return
    
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date(),
      context: selectedNode ? { nodeId: selectedNode.id } : undefined
    }
    
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setChatLoading(true)
    
    try {
      // Build context for LLM
      const context = {
        selected_node: selectedNode,
        graph_stats: graphData?.stats,
        recent_nodes: graphData?.nodes.slice(0, 10).map(n => n.name)
      }
      
      const res = await fetch('/api/v1/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          context: JSON.stringify(context),
          backend
        })
      })
      
      if (!res.ok) throw new Error('Chat failed')
      
      const data = await res.json()
      
      setMessages(prev => [...prev, {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: data.response || data.message || 'No response',
        timestamp: new Date()
      }])
      
      // Check if response mentions specific nodes
      if (data.highlighted_nodes && fgRef.current) {
        // Could highlight nodes here
      }
      
    } catch (err) {
      setMessages(prev => [...prev, {
        id: `error-${Date.now()}`,
        role: 'system',
        content: `❌ Error: ${err instanceof Error ? err.message : 'Unknown error'}`,
        timestamp: new Date()
      }])
    } finally {
      setChatLoading(false)
    }
  }

  // Quick action
  const handleQuickAction = (action: typeof QUICK_ACTIONS[0]) => {
    const nodeContext = selectedNode ? selectedNode.name : ''
    setInput(action.prompt + nodeContext)
    chatInputRef.current?.focus()
  }

  // Refresh graph
  const handleRefresh = async () => {
    setLoading(true)
    try {
      const res = await fetch(`/api/v1/knowledge/graph?backend=${backend}&refresh=true`)
      if (!res.ok) throw new Error('Refresh failed')
      const data = await res.json()
      setGraphData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Refresh failed')
    } finally {
      setLoading(false)
    }
  }

  // ==================== Render ====================

  if (error && !graphData) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-900 text-white">
        <div className="text-center">
          <p className="text-red-400 mb-4">⚠️ {error}</p>
          <button onClick={handleRefresh} className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-700">
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="h-full flex flex-col bg-slate-900 rounded-xl overflow-hidden">
      
      {/* ========== Graph Area ========== */}
      <div className="flex-1 relative">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/80 z-10">
            <div className="text-center text-white">
              <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
              <p>Loading Knowledge Graph...</p>
            </div>
          </div>
        )}
        
        {graphData && (
          <ForceGraph3D
            ref={fgRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={graphData}
            backgroundColor="#0f172a"
            nodeLabel={(node: any) => `${node.name} (${node.type})`}
            nodeColor={(node: any) => NODE_COLORS[node.type] || NODE_COLORS.default}
            nodeRelSize={6}
            nodeVal={(node: any) => node.type === 'event' ? 12 : 8}
            linkColor={() => 'rgba(255,255,255,0.2)'}
            linkWidth={1}
            linkDirectionalParticles={2}
            linkDirectionalParticleWidth={2}
            linkDirectionalParticleColor={() => '#60a5fa'}
            onNodeClick={handleNodeClick}
            onNodeHover={setHoveredNode}
            nodeThreeObject={(node: any) => {
              const sprite = new SpriteText(node.name)
              sprite.color = NODE_COLORS[node.type] || NODE_COLORS.default
              sprite.textHeight = 4
              sprite.backgroundColor = 'rgba(0,0,0,0.6)'
              sprite.padding = 1
              sprite.borderRadius = 2
              return sprite
            }}
            nodeThreeObjectExtend={true}
          />
        )}

        {/* Stats Overlay */}
        {graphData && (
          <div className="absolute top-4 left-4 bg-slate-800/90 backdrop-blur-sm rounded-lg p-3 text-sm text-white">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                <span>{graphData.stats.papers} papers</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                <span>{graphData.stats.events} events</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                <span>{graphData.stats.patterns} patterns</span>
              </div>
            </div>
          </div>
        )}

        {/* Controls Overlay */}
        <div className="absolute top-4 right-4 flex gap-2">
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="p-2 bg-slate-800/90 backdrop-blur-sm rounded-lg text-white hover:bg-slate-700 disabled:opacity-50"
            title="Refresh"
          >
            <RefreshCw className={clsx("w-5 h-5", loading && "animate-spin")} />
          </button>
          <button
            onClick={() => setChatExpanded(!chatExpanded)}
            className="p-2 bg-slate-800/90 backdrop-blur-sm rounded-lg text-white hover:bg-slate-700"
            title={chatExpanded ? "Collapse chat" : "Expand chat"}
          >
            {chatExpanded ? <ChevronDown className="w-5 h-5" /> : <ChevronUp className="w-5 h-5" />}
          </button>
        </div>

        {/* Selected Node Info */}
        {selectedNode && (
          <div className="absolute bottom-4 left-4 bg-slate-800/90 backdrop-blur-sm rounded-lg p-4 max-w-sm text-white">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-semibold">{selectedNode.name}</h3>
                <span className={clsx(
                  "inline-block px-2 py-0.5 rounded text-xs mt-1",
                  selectedNode.type === 'paper' && "bg-blue-500/30 text-blue-300",
                  selectedNode.type === 'event' && "bg-amber-500/30 text-amber-300",
                  selectedNode.type === 'pattern' && "bg-emerald-500/30 text-emerald-300",
                )}>
                  {selectedNode.type}
                </span>
              </div>
              <button 
                onClick={() => setSelectedNode(null)}
                className="text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            {selectedNode.properties && (
              <div className="mt-2 text-xs text-slate-300 space-y-1">
                {Object.entries(selectedNode.properties).slice(0, 3).map(([k, v]) => (
                  <div key={k}><span className="text-slate-400">{k}:</span> {String(v).slice(0, 50)}</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ========== Chat Strip ========== */}
      <div className={clsx(
        "border-t border-slate-700 bg-slate-800 transition-all duration-300",
        chatExpanded ? "h-[300px]" : "h-[80px]"
      )}>
        {/* Expanded: Show messages */}
        {chatExpanded && (
          <div className="h-[220px] overflow-y-auto p-4 space-y-3">
            {messages.map(msg => (
              <div
                key={msg.id}
                className={clsx(
                  "max-w-[80%] rounded-lg p-3 text-sm",
                  msg.role === 'user' && "ml-auto bg-blue-600 text-white",
                  msg.role === 'assistant' && "bg-slate-700 text-white",
                  msg.role === 'system' && "bg-slate-600/50 text-slate-300 text-xs"
                )}
              >
                {msg.content}
              </div>
            ))}
            {chatLoading && (
              <div className="flex items-center gap-2 text-slate-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Thinking...</span>
              </div>
            )}
          </div>
        )}

        {/* Input area */}
        <div className="p-3 border-t border-slate-700">
          <div className="flex items-center gap-2">
            {/* Quick actions */}
            <div className="flex gap-1">
              {QUICK_ACTIONS.map(action => (
                <button
                  key={action.id}
                  onClick={() => handleQuickAction(action)}
                  className="p-2 text-slate-400 hover:text-white hover:bg-slate-700 rounded-lg transition-colors"
                  title={action.label}
                >
                  <action.icon className="w-4 h-4" />
                </button>
              ))}
            </div>
            
            {/* Input */}
            <div className="flex-1 relative">
              <input
                ref={chatInputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleChatSubmit()}
                placeholder={selectedNode 
                  ? `Ask about "${selectedNode.name}"...` 
                  : "Ask about the knowledge graph, query the database..."
                }
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500"
              />
              {selectedNode && (
                <span className="absolute right-12 top-1/2 -translate-y-1/2 px-2 py-0.5 bg-blue-500/30 text-blue-300 text-xs rounded">
                  @{selectedNode.name}
                </span>
              )}
            </div>
            
            {/* Send button */}
            <button
              onClick={handleChatSubmit}
              disabled={!input.trim() || chatLoading}
              className="p-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white transition-colors"
            >
              {chatLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          
          {/* Hint */}
          {!chatExpanded && messages.length > 1 && (
            <p className="text-xs text-slate-500 mt-1 text-center">
              Click <ChevronUp className="w-3 h-3 inline" /> to see chat history • {messages.length} messages
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

export default KnowledgeCockpit
