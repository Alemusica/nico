/**
 * 🌊 Causal Discovery Dashboard
 * Swiss Design System - Blue Emerald Oceanography
 */

import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './components/Sidebar'
import { Header } from './components/Header'
import { CausalGraphView } from './components/CausalGraphView'
import { DataPanel } from './components/DataPanel'
import { ChatPanel } from './components/ChatPanel'
import { KnowledgeSearch } from './components/KnowledgeSearch'
import { useStore } from './store'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
    },
  },
})

type ViewType = 'graph' | 'data' | 'knowledge' | 'chat'

export default function App() {
  const [activeView, setActiveView] = useState<ViewType>('graph')
  const { backend, setBackend } = useStore()

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-slate-50">
        {/* Sidebar */}
        <Sidebar activeView={activeView} onViewChange={setActiveView} />
        
        {/* Main Content */}
        <div className="ml-[280px]">
          {/* Header */}
          <Header 
            backend={backend} 
            onBackendChange={setBackend}
          />
          
          {/* Content Area */}
          <main className="p-phi-xl">
            {activeView === 'graph' && (
              <div className="grid grid-cols-swiss-main gap-phi-xl">
                <CausalGraphView />
                <ChatPanel />
              </div>
            )}
            
            {activeView === 'data' && (
              <DataPanel />
            )}
            
            {activeView === 'knowledge' && (
              <KnowledgeSearch />
            )}
            
            {activeView === 'chat' && (
              <div className="max-w-3xl mx-auto">
                <ChatPanel expanded />
              </div>
            )}
          </main>
        </div>
      </div>
    </QueryClientProvider>
  )
}
