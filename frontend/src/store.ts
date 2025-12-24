/**
 * Zustand Store for Global State
 */

import { create } from 'zustand'

export type Backend = 'neo4j' | 'surrealdb'

interface CausalLink {
  source: string
  target: string
  lag: number
  strength: number
  pValue: number
  explanation?: string
  physicsValid?: boolean
}

interface CausalGraph {
  variables: string[]
  links: CausalLink[]
  maxLag: number
  alpha: number
  method: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  metadata?: {
    type?: 'investigation_result'
    investigation_result?: any
  }
}

interface AppState {
  // Backend selection
  backend: Backend
  setBackend: (backend: Backend) => void
  
  // Dataset
  currentDataset: string | null
  setCurrentDataset: (name: string | null) => void
  
  // Causal graph
  causalGraph: CausalGraph | null
  setCausalGraph: (graph: CausalGraph | null) => void
  selectedNode: string | null
  setSelectedNode: (node: string | null) => void
  selectedLink: CausalLink | null
  setSelectedLink: (link: CausalLink | null) => void
  
  // Chat
  messages: Message[]
  addMessage: (role: 'user' | 'assistant', content: string, metadata?: any) => void
  clearMessages: () => void
  
  // Investigation
  pendingInvestigationResult: any | null
  setPendingInvestigationResult: (result: any | null) => void
  
  // UI State
  sidebarCollapsed: boolean
  toggleSidebar: () => void
  isLoading: boolean
  setLoading: (loading: boolean) => void
}

export const useStore = create<AppState>((set) => ({
  // Backend - default to Neo4j
  backend: 'neo4j',
  setBackend: (backend) => set({ backend }),
  
  // Dataset
  currentDataset: null,
  setCurrentDataset: (name) => set({ currentDataset: name }),
  
  // Causal graph
  causalGraph: null,
  setCausalGraph: (graph) => set({ causalGraph: graph }),
  selectedNode: null,
  setSelectedNode: (node) => set({ selectedNode: node }),
  selectedLink: null,
  setSelectedLink: (link) => set({ selectedLink: link }),
  
  // Chat
  messages: [],
  addMessage: (role, content, metadata) => set((state) => ({
    messages: [
      ...state.messages,
      {
        id: crypto.randomUUID(),
        role,
        content,
        timestamp: new Date(),
        metadata,
      },
    ],
  })),
  clearMessages: () => set({ messages: [] }),
  
  // Investigation
  pendingInvestigationResult: null,
  setPendingInvestigationResult: (result) => set({ pendingInvestigationResult: result }),
  
  // UI
  sidebarCollapsed: false,
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  isLoading: false,
  setLoading: (loading) => set({ isLoading: loading }),
}))
