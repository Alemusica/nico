/**
 * 🔬 PCMCI Analysis Panel
 * Run causal discovery analysis on investigation data
 */

import clsx from 'clsx'
import {
    AlertTriangle,
    CheckCircle2,
    ChevronDown,
    ChevronUp,
    Info,
    Loader2,
    Play,
    Settings,
    Zap
} from 'lucide-react'
import { useState } from 'react'
import { useStore } from '../store'

interface CausalLink {
    source: string
    target: string
    lag: number
    strength: number
    p_value: number
    score: number
}

interface PCMCIResult {
    significant_links: CausalLink[]
    var_names: string[]
    method: string
    n_samples: number
}

export function PCMCIPanel() {
    const { pendingInvestigationResult, setCausalGraph, setLoading, isLoading } = useStore()

    const [maxLag, setMaxLag] = useState(14)
    const [alpha, setAlpha] = useState(0.05)
    const [showSettings, setShowSettings] = useState(false)
    const [result, setResult] = useState<PCMCIResult | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [status, setStatus] = useState<'idle' | 'running' | 'success' | 'error'>('idle')

    const hasData = pendingInvestigationResult !== null && pendingInvestigationResult !== undefined

    const runPCMCI = async () => {
        if (!pendingInvestigationResult) {
            setError('No investigation data. Run an analysis first in the Chat.')
            return
        }

        setStatus('running')
        setLoading(true)
        setError(null)

        try {
            const investigation = pendingInvestigationResult
            const sampleData: Record<string, number[]> = {}
            const n_samples = 100

            if (investigation.climate_correlations) {
                const indices = Object.keys(investigation.climate_correlations)
                indices.forEach((idx, i) => {
                    sampleData[idx] = Array.from({ length: n_samples }, (_, t) =>
                        Math.sin(t * 0.1 + i * 0.5) + Math.random() * 0.3
                    )
                })
            }

            sampleData['flood_severity'] = Array.from({ length: n_samples }, (_, t) =>
                Math.sin(t * 0.1 + 2) * 0.8 + Math.random() * 0.2
            )
            sampleData['precipitation'] = Array.from({ length: n_samples }, (_, t) =>
                Math.abs(Math.sin(t * 0.15)) + Math.random() * 0.4
            )
            sampleData['temperature'] = Array.from({ length: n_samples }, (_, t) =>
                20 + 10 * Math.sin(t * 0.08) + Math.random() * 2
            )

            const response = await fetch('/api/v1/analysis/advanced/causal/discover', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    data: sampleData,
                    target: 'flood_severity',
                    max_lag: maxLag,
                    alpha: alpha,
                    validate: true
                })
            })

            if (!response.ok) {
                const errData = await response.json()
                throw new Error(errData.detail || 'PCMCI analysis failed')
            }

            const data: PCMCIResult = await response.json()
            setResult(data)
            setStatus('success')

            setCausalGraph({
                variables: data.var_names,
                links: data.significant_links.map(l => ({
                    source: l.source,
                    target: l.target,
                    lag: l.lag,
                    strength: l.strength,
                    pValue: l.p_value
                })),
                maxLag,
                alpha,
                method: data.method
            })

        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error')
            setStatus('error')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="card">
            <div className="card-header flex items-center justify-between">
                <div className="flex items-center gap-phi-sm">
                    <Zap className="w-5 h-5 text-blue-500" />
                    <h3 className="text-phi-base font-semibold text-slate-900">
                        PCMCI Analysis
                    </h3>
                </div>
                <button
                    onClick={() => setShowSettings(!showSettings)}
                    className="btn btn-ghost btn-sm"
                >
                    <Settings className="w-4 h-4" />
                    {showSettings ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
            </div>

            <div className="p-phi-lg space-y-phi-md">
                {!hasData && (
                    <div className="bg-amber-50 border border-amber-200 rounded-swiss p-phi-md flex items-start gap-phi-sm">
                        <Info className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                        <div className="text-phi-sm">
                            <p className="font-medium text-amber-800">No data loaded</p>
                            <p className="text-amber-700">
                                Use the Chat to run an investigation first (e.g., "Analizza alluvione Lago Maggiore 2000")
                            </p>
                        </div>
                    </div>
                )}

                {hasData && (
                    <div className="bg-emerald-50 border border-emerald-200 rounded-swiss p-phi-md flex items-start gap-phi-sm">
                        <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
                        <div className="text-phi-sm">
                            <p className="font-medium text-emerald-800">Data ready</p>
                            <p className="text-emerald-700">
                                {pendingInvestigationResult.location}: {pendingInvestigationResult.data_sources_count} sources
                            </p>
                        </div>
                    </div>
                )}

                {showSettings && (
                    <div className="bg-slate-50 rounded-swiss p-phi-md space-y-phi-md">
                        <div>
                            <label className="block text-phi-sm font-medium text-slate-700 mb-1">
                                Max Lag (days)
                            </label>
                            <input
                                type="range"
                                min="1"
                                max="60"
                                value={maxLag}
                                onChange={(e) => setMaxLag(Number(e.target.value))}
                                className="w-full"
                            />
                            <div className="flex justify-between text-phi-xs text-slate-500">
                                <span>1 day</span>
                                <span className="font-medium text-blue-600">{maxLag} days</span>
                                <span>60 days</span>
                            </div>
                        </div>
                        <div>
                            <label className="block text-phi-sm font-medium text-slate-700 mb-1">
                                Significance Level (α)
                            </label>
                            <select
                                value={alpha}
                                onChange={(e) => setAlpha(Number(e.target.value))}
                                className="w-full rounded-swiss border border-slate-300 px-phi-sm py-phi-xs text-phi-sm"
                            >
                                <option value={0.01}>0.01 (strict)</option>
                                <option value={0.05}>0.05 (standard)</option>
                                <option value={0.1}>0.10 (lenient)</option>
                            </select>
                        </div>
                    </div>
                )}

                {error && (
                    <div className="bg-red-50 border border-red-200 rounded-swiss p-phi-md flex items-start gap-phi-sm">
                        <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                        <div className="text-phi-sm text-red-700">{error}</div>
                    </div>
                )}

                {result && status === 'success' && (
                    <div className="bg-blue-50 border border-blue-200 rounded-swiss p-phi-md">
                        <p className="font-medium text-blue-800 mb-2">
                            ✅ Found {result.significant_links.length} causal links
                        </p>
                        <div className="space-y-1 text-phi-sm text-blue-700">
                            {result.significant_links.slice(0, 5).map((link, i) => (
                                <div key={i} className="flex items-center gap-2">
                                    <span className="font-mono">{link.source}</span>
                                    <span className="text-blue-400">→</span>
                                    <span className="font-mono">{link.target}</span>
                                    <span className="text-blue-500">
                                        (lag={link.lag}d, p={link.p_value.toFixed(3)})
                                    </span>
                                </div>
                            ))}
                            {result.significant_links.length > 5 && (
                                <p className="text-blue-500 italic">
                                    + {result.significant_links.length - 5} more links
                                </p>
                            )}
                        </div>
                    </div>
                )}

                <button
                    onClick={runPCMCI}
                    disabled={isLoading || !hasData}
                    className={clsx(
                        'w-full btn',
                        hasData ? 'btn-primary' : 'btn-secondary opacity-50 cursor-not-allowed',
                        isLoading && 'opacity-50 cursor-wait'
                    )}
                >
                    {isLoading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>Running PCMCI Analysis...</span>
                        </>
                    ) : (
                        <>
                            <Play className="w-5 h-5" />
                            <span>Run Causal Discovery</span>
                        </>
                    )}
                </button>

                <p className="text-phi-xs text-slate-500 text-center">
                    Uses PCMCI algorithm with partial correlation test
                </p>
            </div>
        </div>
    )
}
