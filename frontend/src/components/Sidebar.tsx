/**
 * Sidebar Component - Swiss Design
 */

import { 
  Activity, 
  Database, 
  MessageSquare, 
  Search,
  Waves,
  GitBranch,
  BookOpen,
  Zap,
  Calendar
} from 'lucide-react'
import clsx from 'clsx'

interface SidebarProps {
  activeView: string
  onViewChange: (view: 'graph' | 'data' | 'knowledge' | 'chat' | 'historical') => void
}

const navigation = [
  { id: 'graph', label: 'Causal Graph', icon: GitBranch },
  { id: 'data', label: 'Data Explorer', icon: Database },
  { id: 'historical', label: 'Historical Analysis', icon: Calendar },
  { id: 'knowledge', label: 'Knowledge Base', icon: BookOpen },
  { id: 'chat', label: 'AI Assistant', icon: MessageSquare },
] as const

export function Sidebar({ activeView, onViewChange }: SidebarProps) {
  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="px-phi-lg py-phi-xl border-b border-slate-800">
        <div className="flex items-center gap-phi-md">
          <div className="w-10 h-10 rounded-swiss-lg bg-gradient-to-br from-blue-500 to-emerald-500 flex items-center justify-center">
            <Waves className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-phi-base font-semibold text-white">
              Causal Discovery
            </h1>
            <p className="text-phi-xs text-slate-400">
              Oceanography Dashboard
            </p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="p-phi-md space-y-phi-xs">
        <p className="px-phi-md py-phi-sm text-phi-xs font-medium text-slate-500 uppercase tracking-wider">
          Analysis
        </p>
        
        {navigation.map((item) => (
          <button
            key={item.id}
            onClick={() => onViewChange(item.id as 'graph' | 'data' | 'knowledge' | 'chat' | 'historical')}
            className={clsx(
              'sidebar-item w-full text-left',
              activeView === item.id && 'sidebar-item-active'
            )}
          >
            <item.icon className="w-5 h-5" />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      {/* Quick Actions */}
      <div className="p-phi-md border-t border-slate-800 mt-auto">
        <p className="px-phi-md py-phi-sm text-phi-xs font-medium text-slate-500 uppercase tracking-wider">
          Quick Actions
        </p>
        
        <button className="sidebar-item w-full text-left">
          <Zap className="w-5 h-5" />
          <span>Run Discovery</span>
        </button>
        
        <button className="sidebar-item w-full text-left">
          <Search className="w-5 h-5" />
          <span>Search Papers</span>
        </button>
        
        <button className="sidebar-item w-full text-left">
          <Activity className="w-5 h-5" />
          <span>View Patterns</span>
        </button>
      </div>

      {/* Status */}
      <div className="p-phi-md border-t border-slate-800">
        <div className="px-phi-md py-phi-sm flex items-center gap-phi-sm">
          <div className="status-dot status-connected" />
          <span className="text-phi-xs text-slate-400">API Connected</span>
        </div>
      </div>
    </aside>
  )
}
