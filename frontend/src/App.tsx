/**
 * 🌊 Causal Discovery Dashboard
 * Swiss Design System - Blue Emerald Oceanography
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CausalGraphView } from './components/CausalGraphView'
import { ChatPanel } from './components/ChatPanel'
import { CosmographView } from './components/CosmographView'
import { DataPanel } from './components/DataPanel'
import { Header } from './components/Header'
import { HistoricalAnalysis } from './components/HistoricalAnalysis'
import { KnowledgeCockpit } from './components/KnowledgeCockpit'
import { KnowledgeGraph3DView } from './components/KnowledgeGraph3DView'
import { KnowledgeSearch } from './components/KnowledgeSearch'
import { PCMCIPanel } from './components/PCMCIPanel'
import { Sidebar } from './components/Sidebar'
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
              <div className="grid grid-cols-3 gap-phi-xl">
                <div className="col-span-2">
                  <CausalGraphView />
                </div>
                <div className="space-y-phi-lg">
                  <PCMCIPanel />
                  <ChatPanel />
                </div>
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
                <KnowledgeGraph3DView />
              </div>
            )}

            {activeView === 'cosmograph' && (
              <div className="h-[calc(100vh-140px)]">
                <CosmographView />
              </div>
            )}

            {activeView === 'historical' && (
              <HistoricalAnalysis />
            )}

            {activeView === 'cockpit' && (
              <div className="h-[calc(100vh-140px)]">
                <KnowledgeCockpit />
              </div>
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
