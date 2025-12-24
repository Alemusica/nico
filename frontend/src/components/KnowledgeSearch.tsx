/**
 * Knowledge Search Component - Swiss Design
 * Search papers, events, patterns, climate indices
 * Now includes Data Explorer for direct data downloads
 */

import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { 
  Search, 
  BookOpen, 
  Calendar, 
  Activity, 
  Thermometer,
  ExternalLink,
  Plus,
  Filter,
  Database,
  Server,
  Download
} from 'lucide-react'
import clsx from 'clsx'
import { useStore } from '../store'
import { 
  searchPapers, 
  searchPatterns, 
  searchEvents, 
  listClimateIndices,
  getKnowledgeStats,
  compareBackends
} from '../api'
import { DataExplorer } from './DataExplorer'

type TabType = 'papers' | 'patterns' | 'events' | 'climate' | 'explorer'

const tabs = [
  { id: 'papers', label: 'Papers', icon: BookOpen },
  { id: 'patterns', label: 'Patterns', icon: Activity },
  { id: 'events', label: 'Events', icon: Calendar },
  { id: 'climate', label: 'Climate', icon: Thermometer },
  { id: 'explorer', label: 'Data Explorer', icon: Download },
] as const

export function KnowledgeSearch() {
  const { backend } = useStore()
  const [activeTab, setActiveTab] = useState<TabType>('papers')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any>(null)
  const [showCompare, setShowCompare] = useState(false)

  // Fetch stats
  const { data: stats } = useQuery({
    queryKey: ['knowledge-stats', backend],
    queryFn: () => getKnowledgeStats(backend),
  })

  // Compare backends
  const { data: comparison } = useQuery({
    queryKey: ['compare-backends'],
    queryFn: compareBackends,
    enabled: showCompare,
  })

  // Climate indices list
  const { data: climateData } = useQuery({
    queryKey: ['climate-indices', backend],
    queryFn: () => listClimateIndices(backend),
    enabled: activeTab === 'climate',
  })

  // Search handlers
  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    try {
      let results
      switch (activeTab) {
        case 'papers':
          results = await searchPapers(searchQuery, 20, backend)
          break
        case 'patterns':
          results = await searchPatterns({ variables: searchQuery, limit: 20 }, backend)
          break
        case 'events':
          results = await searchEvents({ event_type: searchQuery, limit: 20 }, backend)
          break
        default:
          results = null
      }
      setSearchResults(results)
    } catch (error) {
      console.error('Search failed:', error)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
    }
  }

  const statistics = stats?.statistics || {}

  return (
    <div className="space-y-phi-xl">
      {/* Stats Overview */}
      <div className="grid grid-cols-5 gap-phi-lg">
        <StatsCard 
          label="Papers" 
          value={statistics.papers || 0} 
          icon={BookOpen}
          color="blue"
        />
        <StatsCard 
          label="Patterns" 
          value={statistics.patterns || 0} 
          icon={Activity}
          color="emerald"
        />
        <StatsCard 
          label="Events" 
          value={statistics.events || 0} 
          icon={Calendar}
          color="cyan"
        />
        <StatsCard 
          label="Climate Indices" 
          value={statistics.climate_indices || 0} 
          icon={Thermometer}
          color="amber"
        />
        <StatsCard 
          label="Relationships" 
          value={statistics.relationships || 0} 
          icon={Activity}
          color="violet"
        />
      </div>

      {/* Backend Comparison */}
      {showCompare && comparison && (
        <div className="card p-phi-lg">
          <h4 className="text-phi-base font-semibold text-slate-900 mb-phi-md">
            Backend Comparison
          </h4>
          <div className="grid grid-cols-2 gap-phi-xl">
            {/* Neo4j */}
            <div className={clsx(
              'p-phi-lg rounded-swiss-lg border-2',
              backend === 'neo4j' ? 'border-blue-500 bg-blue-50' : 'border-slate-200'
            )}>
              <div className="flex items-center gap-phi-sm mb-phi-md">
                <Database className="w-5 h-5 text-blue-600" />
                <span className="font-semibold text-slate-900">Neo4j</span>
                {comparison.neo4j?.status === 'connected' && (
                  <span className="badge badge-blue ml-auto">Connected</span>
                )}
              </div>
              {comparison.neo4j?.statistics && (
                <div className="grid grid-cols-2 gap-phi-sm text-phi-sm">
                  <div>Papers: {comparison.neo4j.statistics.papers}</div>
                  <div>Patterns: {comparison.neo4j.statistics.patterns}</div>
                  <div>Events: {comparison.neo4j.statistics.events}</div>
                  <div>Relations: {comparison.neo4j.statistics.relationships}</div>
                </div>
              )}
            </div>

            {/* SurrealDB */}
            <div className={clsx(
              'p-phi-lg rounded-swiss-lg border-2',
              backend === 'surrealdb' ? 'border-emerald-500 bg-emerald-50' : 'border-slate-200'
            )}>
              <div className="flex items-center gap-phi-sm mb-phi-md">
                <Server className="w-5 h-5 text-emerald-600" />
                <span className="font-semibold text-slate-900">SurrealDB</span>
                {comparison.surrealdb?.status === 'connected' && (
                  <span className="badge badge-emerald ml-auto">Connected</span>
                )}
              </div>
              {comparison.surrealdb?.statistics && (
                <div className="grid grid-cols-2 gap-phi-sm text-phi-sm">
                  <div>Papers: {comparison.surrealdb.statistics.papers}</div>
                  <div>Patterns: {comparison.surrealdb.statistics.patterns}</div>
                  <div>Events: {comparison.surrealdb.statistics.events}</div>
                  <div>Relations: {comparison.surrealdb.statistics.relationships}</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Main Search Card */}
      <div className="card">
        {/* Tabs */}
        <div className="border-b border-slate-200">
          <div className="flex items-center px-phi-lg">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id)
                  setSearchResults(null)
                }}
                className={clsx(
                  'flex items-center gap-phi-sm px-phi-lg py-phi-md text-phi-sm font-medium border-b-2 -mb-px transition-colors',
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-slate-500 hover:text-slate-700'
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
            
            <button
              onClick={() => setShowCompare(!showCompare)}
              className="ml-auto btn btn-ghost btn-sm"
            >
              Compare Backends
            </button>
          </div>
        </div>

        {/* Search Bar */}
        <div className="p-phi-lg border-b border-slate-100">
          <div className="flex gap-phi-md">
            <div className="flex-1 relative">
              <Search className="absolute left-phi-md top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={`Search ${activeTab}...`}
                className="input pl-12"
              />
            </div>
            <button onClick={handleSearch} className="btn btn-primary">
              Search
            </button>
            <button className="btn btn-outline">
              <Filter className="w-4 h-4" />
              Filters
            </button>
            <button className="btn btn-secondary">
              <Plus className="w-4 h-4" />
              Add New
            </button>
          </div>
        </div>

        {/* Results */}
        <div className="p-phi-lg min-h-[400px]">
          {activeTab === 'explorer' ? (
            <DataExplorer />
          ) : activeTab === 'climate' && climateData?.indices ? (
            <ClimateIndicesList indices={climateData.indices} />
          ) : searchResults ? (
            <SearchResults 
              type={activeTab} 
              results={searchResults} 
            />
          ) : (
            <EmptyState type={activeTab} />
          )}
        </div>
      </div>
    </div>
  )
}

// Stats Card Component
interface StatsCardProps {
  label: string
  value: number
  icon: React.ComponentType<{ className?: string }>
  color: 'blue' | 'emerald' | 'cyan' | 'amber' | 'violet'
}

function StatsCard({ label, value, icon: Icon, color }: StatsCardProps) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600 border-blue-100',
    emerald: 'bg-emerald-50 text-emerald-600 border-emerald-100',
    cyan: 'bg-cyan-50 text-cyan-600 border-cyan-100',
    amber: 'bg-amber-50 text-amber-600 border-amber-100',
    violet: 'bg-violet-50 text-violet-600 border-violet-100',
  }

  return (
    <div className={clsx('card p-phi-lg border', colorClasses[color])}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-phi-xs font-medium text-slate-500 uppercase tracking-wider">
            {label}
          </p>
          <p className="text-phi-2xl font-bold mt-phi-xs">
            {value.toLocaleString()}
          </p>
        </div>
        <Icon className="w-8 h-8 opacity-50" />
      </div>
    </div>
  )
}

// Search Results Component
interface SearchResultsProps {
  type: TabType
  results: any
}

function SearchResults({ type, results }: SearchResultsProps) {
  if (type === 'papers' && results?.results) {
    return (
      <div className="space-y-phi-md">
        {results.results.map((item: any, i: number) => (
          <div key={i} className="knowledge-card">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h5 className="text-phi-base font-semibold text-slate-900 line-clamp-2">
                  {item.paper?.title || 'Untitled'}
                </h5>
                <p className="text-phi-sm text-slate-600 mt-phi-xs">
                  {item.paper?.authors?.join(', ') || 'Unknown authors'}
                </p>
                <p className="text-phi-sm text-slate-500 mt-phi-sm line-clamp-2">
                  {item.paper?.abstract || 'No abstract available'}
                </p>
                <div className="flex items-center gap-phi-md mt-phi-md">
                  <span className="badge badge-blue">{item.paper?.year}</span>
                  {item.paper?.journal && (
                    <span className="text-phi-xs text-slate-500">{item.paper.journal}</span>
                  )}
                  <span className="text-phi-xs text-emerald-600 ml-auto">
                    Score: {(item.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              {item.paper?.doi && (
                <a
                  href={`https://doi.org/${item.paper.doi}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-ghost btn-sm"
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    )
  }

  if (type === 'patterns' && results?.patterns) {
    return (
      <div className="space-y-phi-md">
        {results.patterns.map((pattern: any, i: number) => (
          <div key={i} className="knowledge-card">
            <div className="flex items-center gap-phi-md mb-phi-sm">
              <span className="badge badge-emerald">{pattern.pattern_type}</span>
              {pattern.confidence && (
                <span className="text-phi-xs text-slate-500">
                  Confidence: {(pattern.confidence * 100).toFixed(0)}%
                </span>
              )}
            </div>
            <h5 className="text-phi-base font-semibold text-slate-900">
              {pattern.name}
            </h5>
            <p className="text-phi-sm text-slate-600 mt-phi-xs">
              {pattern.description}
            </p>
            <div className="flex flex-wrap gap-phi-xs mt-phi-md">
              {pattern.variables?.map((v: string) => (
                <span key={v} className="badge badge-slate">{v}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    )
  }

  if (type === 'events' && results?.events) {
    return (
      <div className="space-y-phi-md">
        {results.events.map((event: any, i: number) => (
          <div key={i} className="knowledge-card">
            <div className="flex items-center gap-phi-md mb-phi-sm">
              <span className="badge badge-blue">{event.event_type}</span>
              <span className="text-phi-xs text-slate-500">
                {event.start_date?.split('T')[0]}
              </span>
            </div>
            <h5 className="text-phi-base font-semibold text-slate-900">
              {event.name}
            </h5>
            <p className="text-phi-sm text-slate-600 mt-phi-xs">
              {event.description}
            </p>
            {event.severity && (
              <div className="mt-phi-sm">
                <span className={clsx(
                  'badge',
                  event.severity > 0.7 ? 'badge-red' : 
                  event.severity > 0.4 ? 'badge-blue' : 'badge-slate'
                )}>
                  Severity: {(event.severity * 100).toFixed(0)}%
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    )
  }

  return <EmptyState type={type} />
}

// Climate Indices List
function ClimateIndicesList({ indices }: { indices: any[] }) {
  return (
    <div className="grid grid-cols-2 gap-phi-lg">
      {indices.map((index, i) => (
        <div key={i} className="knowledge-card">
          <div className="flex items-center gap-phi-md mb-phi-sm">
            <div className="w-10 h-10 rounded-swiss bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center text-white font-bold">
              {index.abbreviation?.slice(0, 2) || '?'}
            </div>
            <div>
              <h5 className="text-phi-base font-semibold text-slate-900">
                {index.name}
              </h5>
              <span className="text-phi-xs text-slate-500">
                {index.abbreviation}
              </span>
            </div>
          </div>
          <p className="text-phi-sm text-slate-600 line-clamp-3">
            {index.description}
          </p>
          {index.source_url && (
            <a
              href={index.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-phi-xs text-blue-500 hover:underline mt-phi-sm inline-flex items-center gap-phi-xs"
            >
              Data Source <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>
      ))}
    </div>
  )
}

// Empty State
function EmptyState({ type }: { type: TabType }) {
  const messages = {
    papers: 'Search for research papers by title, keywords, or abstract',
    patterns: 'Search for causal patterns by variable names or type',
    events: 'Search for historical events by type or date range',
    climate: 'Browse available climate indices',
  }

  return (
    <div className="h-[300px] flex flex-col items-center justify-center text-center">
      <Search className="w-12 h-12 text-slate-300 mb-phi-lg" />
      <p className="text-phi-base text-slate-500">{messages[type]}</p>
      <p className="text-phi-sm text-slate-400 mt-phi-xs">
        Using {type === 'papers' ? 'vector similarity' : 'graph traversal'} search
      </p>
    </div>
  )
}
