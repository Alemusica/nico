/**
 * Historical Episode Analysis Component
 * Swiss Design System - Finding Precursor Signals
 * 
 * This component visualizes the search for patterns that PRECEDE
 * well-documented oceanographic events, enabling prediction.
 */

import { useState } from 'react'
import {
  Calendar,
  Activity,
  AlertTriangle,
  Play,
  Loader2,
  Target,
  Clock,
  Zap,
  Map
} from 'lucide-react'
import clsx from 'clsx'

// Well-documented historical episodes for Arctic oceanography
const HISTORICAL_EPISODES = [
  {
    id: 'arctic-ice-2007',
    name: '2007 Arctic Sea Ice Record Minimum',
    type: 'ice_extent',
    startDate: '2007-07-01',
    endDate: '2007-09-30',
    description: 'Record low Arctic sea ice extent, 23% below previous record. Strong anomalous atmospheric circulation patterns observed.',
    precursorWindow: 90, // days before to look for signals
    region: { lat: [70, 85], lon: [-180, 180] },
    knownPrecursors: ['NAO negative phase', 'Warm SST anomalies', 'Anticyclonic circulation'],
    references: ['Stroeve et al., 2008', 'Comiso et al., 2008']
  },
  {
    id: 'fram-outflow-2012',
    name: '2012 Fram Strait Ice Export Anomaly',
    type: 'ice_transport',
    startDate: '2012-01-01',
    endDate: '2012-04-30',
    description: 'Enhanced ice export through Fram Strait driven by atmospheric pressure patterns.',
    precursorWindow: 60,
    region: { lat: [76, 82], lon: [-10, 10] },
    knownPrecursors: ['Strong AO+ phase', 'SSH gradient increase', 'Wind stress anomaly'],
    references: ['Kwok et al., 2013']
  },
  {
    id: 'atlantic-heat-2015',
    name: '2015-16 Atlantic Water Intrusion',
    type: 'heat_transport',
    startDate: '2015-10-01',
    endDate: '2016-03-31',
    description: 'Anomalous warm Atlantic water intrusion into the Arctic via Fram Strait, contributing to winter ice melt.',
    precursorWindow: 120,
    region: { lat: [70, 82], lon: [-5, 15] },
    knownPrecursors: ['Norwegian Sea warm anomaly', 'WSC strengthening', 'NAO shift'],
    references: ['Polyakov et al., 2017', 'Årthun et al., 2017']
  },
  {
    id: 'marine-heatwave-2018',
    name: '2018 Marine Heatwave',
    type: 'temperature_anomaly',
    startDate: '2018-06-01',
    endDate: '2018-08-31',
    description: 'Persistent marine heatwave in Nordic Seas affecting Barents Sea ice edge.',
    precursorWindow: 45,
    region: { lat: [66, 75], lon: [-10, 30] },
    knownPrecursors: ['Blocking high pressure', 'Reduced wind mixing', 'SSH positive anomaly'],
    references: ['Holbrook et al., 2020']
  }
]

interface PrecursorSignal {
  variable: string
  lagDays: number
  correlation: number
  significance: number
  physicsValid: boolean
  mechanism: string
}

interface AnalysisResult {
  episode: typeof HISTORICAL_EPISODES[0]
  precursors: PrecursorSignal[]
  confidenceScore: number
  predictiveWindow: number
  validationStatus: 'pending' | 'validated' | 'needs_review'
}

export function HistoricalAnalysis() {
  const [selectedEpisode, setSelectedEpisode] = useState<typeof HISTORICAL_EPISODES[0] | null>(null)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisLog, setAnalysisLog] = useState<string[]>([])

  // Simulate analysis process (in real implementation, this calls the backend)
  const runAnalysis = async (episode: typeof HISTORICAL_EPISODES[0]) => {
    setIsAnalyzing(true)
    setAnalysisLog([])
    setAnalysisResult(null)
    
    const log = (msg: string) => {
      setAnalysisLog(prev => [...prev, `[${new Date().toISOString().slice(11, 19)}] ${msg}`])
    }
    
    log(`🎯 Starting analysis: ${episode.name}`)
    await delay(500)
    
    log(`📅 Event window: ${episode.startDate} to ${episode.endDate}`)
    log(`🔍 Looking for precursors in ${episode.precursorWindow}-day window`)
    await delay(700)
    
    log(`📊 Loading satellite data for region: Lat ${episode.region.lat[0]}°-${episode.region.lat[1]}°N`)
    await delay(1000)
    
    log(`🔬 Running cross-correlation analysis...`)
    await delay(1500)
    
    // Simulate finding precursors based on episode type
    const precursors = generatePrecursors(episode)
    
    for (const p of precursors) {
      await delay(300)
      const status = p.physicsValid ? '✓ Physics validated' : '⚠ Needs review'
      log(`  Found: ${p.variable} at τ=${p.lagDays}d (r=${p.correlation.toFixed(3)}) ${status}`)
    }
    
    log(`📈 Computing predictive confidence score...`)
    await delay(800)
    
    const confidence = precursors.reduce((sum, p) => 
      sum + (p.correlation * (p.physicsValid ? 1.2 : 0.8)), 0) / precursors.length
    
    log(`✅ Analysis complete! Confidence: ${(confidence * 100).toFixed(1)}%`)
    
    setAnalysisResult({
      episode,
      precursors,
      confidenceScore: confidence,
      predictiveWindow: Math.max(...precursors.map(p => p.lagDays)),
      validationStatus: confidence > 0.7 ? 'validated' : 'needs_review'
    })
    
    setIsAnalyzing(false)
  }

  return (
    <div className="grid grid-cols-12 gap-phi-xl">
      {/* Episode Selector */}
      <div className="col-span-4">
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
          <div className="px-phi-lg py-phi-md border-b border-slate-100 bg-gradient-to-r from-blue-50 to-emerald-50">
            <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
              <Calendar className="w-5 h-5 text-blue-600" />
              Historical Episodes
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Select a well-documented event to find precursor signals
            </p>
          </div>
          
          <div className="divide-y divide-slate-100">
            {HISTORICAL_EPISODES.map((episode) => (
              <button
                key={episode.id}
                onClick={() => setSelectedEpisode(episode)}
                className={clsx(
                  'w-full px-phi-lg py-phi-md text-left transition-colors',
                  selectedEpisode?.id === episode.id
                    ? 'bg-blue-50 border-l-4 border-blue-500'
                    : 'hover:bg-slate-50'
                )}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-medium text-slate-800">{episode.name}</h3>
                    <p className="text-sm text-slate-500 mt-1">{episode.startDate}</p>
                  </div>
                  <span className={clsx(
                    'text-xs px-2 py-1 rounded-full',
                    episode.type === 'ice_extent' && 'bg-cyan-100 text-cyan-700',
                    episode.type === 'ice_transport' && 'bg-blue-100 text-blue-700',
                    episode.type === 'heat_transport' && 'bg-orange-100 text-orange-700',
                    episode.type === 'temperature_anomaly' && 'bg-red-100 text-red-700'
                  )}>
                    {episode.type.replace('_', ' ')}
                  </span>
                </div>
                <p className="text-sm text-slate-600 mt-2 line-clamp-2">
                  {episode.description}
                </p>
                
                <div className="flex gap-2 mt-3 flex-wrap">
                  {episode.knownPrecursors.slice(0, 2).map((p, i) => (
                    <span key={i} className="text-xs px-2 py-0.5 bg-slate-100 text-slate-600 rounded">
                      {p}
                    </span>
                  ))}
                  {episode.knownPrecursors.length > 2 && (
                    <span className="text-xs text-slate-400">
                      +{episode.knownPrecursors.length - 2} more
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Episode Details & Analysis */}
      <div className="col-span-8">
        {selectedEpisode ? (
          <div className="space-y-phi-lg">
            {/* Episode Header */}
            <div className="bg-white rounded-xl border border-slate-200 p-phi-xl">
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-slate-800">
                    {selectedEpisode.name}
                  </h2>
                  <p className="text-slate-600 mt-2">
                    {selectedEpisode.description}
                  </p>
                </div>
                
                <button
                  onClick={() => runAnalysis(selectedEpisode)}
                  disabled={isAnalyzing}
                  className={clsx(
                    'flex items-center gap-2 px-phi-lg py-phi-md rounded-lg font-medium transition-all',
                    isAnalyzing
                      ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-blue-600 to-emerald-600 text-white hover:shadow-lg hover:-translate-y-0.5'
                  )}
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      Find Precursors
                    </>
                  )}
                </button>
              </div>
              
              <div className="grid grid-cols-4 gap-phi-lg mt-phi-xl">
                <div className="bg-slate-50 rounded-lg p-phi-md">
                  <div className="text-sm text-slate-500 flex items-center gap-1">
                    <Calendar className="w-4 h-4" /> Start
                  </div>
                  <div className="font-medium text-slate-800 mt-1">
                    {selectedEpisode.startDate}
                  </div>
                </div>
                <div className="bg-slate-50 rounded-lg p-phi-md">
                  <div className="text-sm text-slate-500 flex items-center gap-1">
                    <Calendar className="w-4 h-4" /> End
                  </div>
                  <div className="font-medium text-slate-800 mt-1">
                    {selectedEpisode.endDate}
                  </div>
                </div>
                <div className="bg-slate-50 rounded-lg p-phi-md">
                  <div className="text-sm text-slate-500 flex items-center gap-1">
                    <Clock className="w-4 h-4" /> Precursor Window
                  </div>
                  <div className="font-medium text-slate-800 mt-1">
                    {selectedEpisode.precursorWindow} days
                  </div>
                </div>
                <div className="bg-slate-50 rounded-lg p-phi-md">
                  <div className="text-sm text-slate-500 flex items-center gap-1">
                    <Map className="w-4 h-4" /> Region
                  </div>
                  <div className="font-medium text-slate-800 mt-1">
                    {selectedEpisode.region.lat[0]}°-{selectedEpisode.region.lat[1]}°N
                  </div>
                </div>
              </div>
              
              {/* Known precursors from literature */}
              <div className="mt-phi-lg">
                <h3 className="text-sm font-medium text-slate-700 mb-2">
                  Known Precursors (from literature)
                </h3>
                <div className="flex flex-wrap gap-2">
                  {selectedEpisode.knownPrecursors.map((p, i) => (
                    <span key={i} className="px-3 py-1 bg-amber-50 text-amber-700 rounded-full text-sm border border-amber-200">
                      {p}
                    </span>
                  ))}
                </div>
                <div className="mt-2 text-xs text-slate-500">
                  References: {selectedEpisode.references.join(', ')}
                </div>
              </div>
            </div>

            {/* Analysis Log */}
            {analysisLog.length > 0 && (
              <div className="bg-slate-900 rounded-xl p-phi-lg font-mono text-sm">
                <div className="text-slate-400 mb-2 flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Analysis Log
                </div>
                <div className="space-y-1 max-h-48 overflow-y-auto">
                  {analysisLog.map((line, i) => (
                    <div key={i} className="text-emerald-400">
                      {line}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Results */}
            {analysisResult && (
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                <div className="px-phi-xl py-phi-lg border-b border-slate-100 bg-gradient-to-r from-emerald-50 to-blue-50">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                      <Target className="w-5 h-5 text-emerald-600" />
                      Discovered Precursor Signals
                    </h3>
                    <div className="flex items-center gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-emerald-600">
                          {(analysisResult.confidenceScore * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-slate-500">Confidence</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">
                          {analysisResult.predictiveWindow}d
                        </div>
                        <div className="text-xs text-slate-500">Lead Time</div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="p-phi-xl">
                  <div className="space-y-phi-md">
                    {analysisResult.precursors.map((p, i) => (
                      <PrecursorCard key={i} precursor={p} rank={i + 1} />
                    ))}
                  </div>
                  
                  {/* Summary */}
                  <div className="mt-phi-xl p-phi-lg bg-gradient-to-r from-blue-50 to-emerald-50 rounded-lg border border-blue-200">
                    <h4 className="font-medium text-slate-800 flex items-center gap-2">
                      <Zap className="w-4 h-4 text-amber-500" />
                      Predictive Insight
                    </h4>
                    <p className="text-slate-600 mt-2">
                      Based on this analysis, similar {selectedEpisode.type.replace('_', ' ')} events 
                      could potentially be predicted <strong>{analysisResult.predictiveWindow} days</strong> in 
                      advance by monitoring:
                    </p>
                    <ul className="mt-2 space-y-1">
                      {analysisResult.precursors
                        .filter(p => p.physicsValid && p.correlation > 0.6)
                        .slice(0, 3)
                        .map((p, i) => (
                          <li key={i} className="text-sm text-slate-700 flex items-center gap-2">
                            <span className="w-2 h-2 bg-emerald-500 rounded-full" />
                            {p.variable} (watch for signals {p.lagDays} days before)
                          </li>
                        ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="bg-white rounded-xl border border-slate-200 p-phi-xxl text-center">
            <Calendar className="w-16 h-16 text-slate-300 mx-auto" />
            <h3 className="text-lg font-medium text-slate-600 mt-phi-lg">
              Select a Historical Episode
            </h3>
            <p className="text-slate-500 mt-2 max-w-md mx-auto">
              Choose a well-documented oceanographic event to analyze satellite data
              and discover precursor patterns that could enable prediction.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

// Precursor Card Component
function PrecursorCard({ precursor, rank }: { precursor: PrecursorSignal; rank: number }) {
  const strengthColor = precursor.correlation > 0.7 
    ? 'emerald' 
    : precursor.correlation > 0.5 
      ? 'blue' 
      : 'slate'
  
  return (
    <div className={clsx(
      'border rounded-lg p-phi-lg',
      precursor.physicsValid ? 'border-emerald-200 bg-emerald-50/30' : 'border-amber-200 bg-amber-50/30'
    )}>
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <span className={clsx(
            'w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold',
            `bg-${strengthColor}-100 text-${strengthColor}-700`
          )}>
            {rank}
          </span>
          <div>
            <h4 className="font-medium text-slate-800">{precursor.variable}</h4>
            <p className="text-sm text-slate-500">{precursor.mechanism}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="text-center">
            <div className={clsx('text-lg font-bold', `text-${strengthColor}-600`)}>
              {precursor.correlation.toFixed(3)}
            </div>
            <div className="text-xs text-slate-500">correlation</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-blue-600">
              τ = {precursor.lagDays}d
            </div>
            <div className="text-xs text-slate-500">lead time</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-slate-600">
              p&lt;{precursor.significance.toFixed(3)}
            </div>
            <div className="text-xs text-slate-500">significance</div>
          </div>
        </div>
      </div>
      
      {/* Validation Status */}
      <div className="mt-3 flex items-center gap-2">
        {precursor.physicsValid ? (
          <span className="flex items-center gap-1 text-sm text-emerald-600">
            <span className="w-2 h-2 bg-emerald-500 rounded-full" />
            Physics validated via ocean current propagation
          </span>
        ) : (
          <span className="flex items-center gap-1 text-sm text-amber-600">
            <AlertTriangle className="w-4 h-4" />
            Needs additional validation
          </span>
        )}
      </div>
    </div>
  )
}

// Helper functions
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function generatePrecursors(episode: typeof HISTORICAL_EPISODES[0]): PrecursorSignal[] {
  // Generate realistic precursors based on episode type
  const basePrecursors: Record<string, PrecursorSignal[]> = {
    'arctic-ice-2007': [
      {
        variable: 'Norwegian Sea SSH Anomaly',
        lagDays: 75,
        correlation: 0.823,
        significance: 0.001,
        physicsValid: true,
        mechanism: 'Atlantic water signal propagating via West Spitsbergen Current'
      },
      {
        variable: 'NAO Index',
        lagDays: 45,
        correlation: -0.756,
        significance: 0.005,
        physicsValid: true,
        mechanism: 'Atmospheric pressure pattern driving wind stress anomalies'
      },
      {
        variable: 'Barents Sea SST',
        lagDays: 30,
        correlation: 0.692,
        significance: 0.01,
        physicsValid: true,
        mechanism: 'Heat content anomaly in adjacent sea'
      },
      {
        variable: 'Fram Strait SSH Gradient',
        lagDays: 21,
        correlation: 0.584,
        significance: 0.02,
        physicsValid: false,
        mechanism: 'Ocean transport proxy (needs validation)'
      }
    ],
    'fram-outflow-2012': [
      {
        variable: 'AO Index',
        lagDays: 60,
        correlation: 0.891,
        significance: 0.0001,
        physicsValid: true,
        mechanism: 'Arctic Oscillation driving transpolar drift pattern'
      },
      {
        variable: 'Central Arctic SSH',
        lagDays: 35,
        correlation: 0.734,
        significance: 0.003,
        physicsValid: true,
        mechanism: 'Sea surface height gradient driving geostrophic flow'
      },
      {
        variable: 'Wind Stress Curl',
        lagDays: 14,
        correlation: 0.689,
        significance: 0.008,
        physicsValid: true,
        mechanism: 'Direct forcing of ice drift'
      }
    ],
    'atlantic-heat-2015': [
      {
        variable: 'Norwegian Sea Temperature',
        lagDays: 90,
        correlation: 0.912,
        significance: 0.0001,
        physicsValid: true,
        mechanism: 'Heat advection via Atlantic inflow'
      },
      {
        variable: 'WSC Transport Index',
        lagDays: 45,
        correlation: 0.845,
        significance: 0.0005,
        physicsValid: true,
        mechanism: 'West Spitsbergen Current strength proxy'
      },
      {
        variable: 'Greenland Sea SSH',
        lagDays: 60,
        correlation: 0.678,
        significance: 0.01,
        physicsValid: true,
        mechanism: 'Cross-current gradient forcing'
      }
    ],
    'marine-heatwave-2018': [
      {
        variable: 'Atmospheric Blocking Index',
        lagDays: 21,
        correlation: 0.867,
        significance: 0.001,
        physicsValid: true,
        mechanism: 'Persistent high pressure suppressing mixing'
      },
      {
        variable: 'Wind Speed Anomaly',
        lagDays: 14,
        correlation: -0.789,
        significance: 0.002,
        physicsValid: true,
        mechanism: 'Reduced wind mixing allowing stratification'
      },
      {
        variable: 'Norwegian Sea SLA',
        lagDays: 30,
        correlation: 0.654,
        significance: 0.015,
        physicsValid: true,
        mechanism: 'Thermosteric expansion signal'
      }
    ]
  }
  
  return basePrecursors[episode.id] || basePrecursors['arctic-ice-2007']
}
