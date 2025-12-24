/**
 * Header Component - Swiss Design
 */

import { Database, Server, RefreshCw } from 'lucide-react'
import clsx from 'clsx'
import type { Backend } from '../store'

interface HeaderProps {
  backend: Backend
  onBackendChange: (backend: Backend) => void
}

export function Header({ backend, onBackendChange }: HeaderProps) {
  return (
    <header className="header">
      <div className="flex-1">
        {/* Breadcrumb or search could go here */}
      </div>

      {/* Backend Toggle */}
      <div className="flex items-center gap-phi-lg">
        <div className="flex items-center gap-phi-sm">
          <span className="text-phi-sm text-slate-500">Backend:</span>
          <div className="flex rounded-swiss bg-slate-100 p-0.5">
            <button
              onClick={() => onBackendChange('neo4j')}
              className={clsx(
                'flex items-center gap-phi-xs px-phi-md py-phi-xs rounded-swiss-sm text-phi-sm font-medium transition-colors',
                backend === 'neo4j'
                  ? 'bg-white text-blue-600 shadow-swiss-sm'
                  : 'text-slate-600 hover:text-slate-900'
              )}
            >
              <Database className="w-4 h-4" />
              Neo4j
            </button>
            <button
              onClick={() => onBackendChange('surrealdb')}
              className={clsx(
                'flex items-center gap-phi-xs px-phi-md py-phi-xs rounded-swiss-sm text-phi-sm font-medium transition-colors',
                backend === 'surrealdb'
                  ? 'bg-white text-emerald-600 shadow-swiss-sm'
                  : 'text-slate-600 hover:text-slate-900'
              )}
            >
              <Server className="w-4 h-4" />
              SurrealDB
            </button>
          </div>
        </div>

        {/* Compare Button */}
        <button className="btn btn-outline btn-sm">
          <RefreshCw className="w-4 h-4" />
          Compare
        </button>
      </div>
    </header>
  )
}
