/**
 * 🌊 Causal Discovery Dashboard
 * Swiss Design System - Blue Emerald Oceanography
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Sidebar } from './components/Sidebar'
import { Header } from './components/Header'
import { CausalGraphView } from './components/CausalGraphView'
import { DataPanel } from './components/DataPanel'
import { ChatPanel } from './components/ChatPanel'
import { KnowledgeSearch } from './components/KnowledgeSearch'
import { KnowledgeGraphView } from './components/KnowledgeGraphView'
import { HistoricalAnalysis } from './components/HistoricalAnalysis'
import { useStore } from './store'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  const { backend, setBackend, activeView, setActiveView } = useStore()

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
            
            {activeView === 'knowledge-graph' && (
              <div className="h-[calc(100vh-140px)]">
                <KnowledgeGraphView />
              </div>
            )}
            
            {activeView === 'historical' && (
              <HistoricalAnalysis />
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
