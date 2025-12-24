/**
 * API Client for Causal Discovery Backend
 */

const API_BASE = 'http://localhost:8000'

export type Backend = 'neo4j' | 'surrealdb'

// =========================================================================
// Data Endpoints
// =========================================================================

export async function listFiles() {
  const res = await fetch(`${API_BASE}/data/files`)
  return res.json()
}

export async function uploadFile(file: File) {
  const formData = new FormData()
  formData.append('file', file)
  
  const res = await fetch(`${API_BASE}/data/upload`, {
    method: 'POST',
    body: formData,
  })
  return res.json()
}

export async function loadFile(filePath: string) {
  const res = await fetch(`${API_BASE}/data/load/${filePath}`)
  return res.json()
}

export async function getDatasetInfo(name: string) {
  const res = await fetch(`${API_BASE}/data/${name}`)
  return res.json()
}

export async function getDataPreview(name: string, rows = 100) {
  const res = await fetch(`${API_BASE}/data/${name}/preview?rows=${rows}`)
  return res.json()
}

// =========================================================================
// Discovery Endpoints
// =========================================================================

export interface DiscoveryRequest {
  dataset_name: string
  variables?: string[]
  time_column?: string
  max_lag?: number
  alpha_level?: number
  domain?: string
  use_llm?: boolean
}

export async function discoverCausality(request: DiscoveryRequest) {
  const res = await fetch(`${API_BASE}/discover`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return res.json()
}

export async function interpretDataset(datasetName: string) {
  const res = await fetch(`${API_BASE}/interpret`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_name: datasetName }),
  })
  return res.json()
}

// =========================================================================
// Chat Endpoints
// =========================================================================

export async function chat(message: string, context?: Record<string, unknown>) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, context }),
  })
  return res.json()
}

// =========================================================================
// Investigation Agent Endpoints
// =========================================================================

const WS_BASE = 'ws://localhost:8000'

export interface InvestigateRequest {
  query: string
  collect_satellite?: boolean
  collect_reanalysis?: boolean
  collect_climate_indices?: boolean
  collect_papers?: boolean
  collect_news?: boolean
  run_correlation?: boolean
  expand_search?: boolean
  // Resolution configuration
  temporal_resolution?: 'hourly' | '6-hourly' | 'daily'
  spatial_resolution?: '0.1' | '0.25' | '0.5'
}

export interface InvestigateProgress {
  step: number | string
  substep?: string
  status: 'started' | 'progress' | 'complete' | 'error'
  message: string
  data?: Record<string, unknown>
  progress?: number
  result?: Record<string, unknown>
}

export interface InvestigateResponse {
  status: string
  query: string
  location?: string
  event_type?: string
  time_range?: string
  data_sources_count?: number
  papers_found?: number
  correlations?: Record<string, unknown>[]
  key_findings?: string[]
  recommendations?: string[]
  confidence?: number
  raw_result?: Record<string, unknown>
}

/**
 * Run investigation with WebSocket streaming for real-time progress.
 */
export function investigateStreaming(
  request: InvestigateRequest,
  onProgress: (progress: InvestigateProgress) => void,
  onComplete: (result: InvestigateResponse) => void,
  onError: (error: string) => void,
): () => void {
  const ws = new WebSocket(`${WS_BASE}/ws/investigate`)
  
  ws.onopen = () => {
    ws.send(JSON.stringify(request))
  }
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data) as InvestigateProgress
    
    if (data.step === 'complete' && data.result) {
      // Investigation finished
      const result: InvestigateResponse = {
        status: 'success',
        query: request.query,
        location: data.result.location as string,
        event_type: data.result.event_type as string,
        time_range: data.result.time_range as string,
        data_sources_count: data.result.data_sources_count as number,
        papers_found: data.result.papers_found as number,
        correlations: data.result.correlations as Record<string, unknown>[],
        key_findings: data.result.key_findings as string[],
        recommendations: data.result.recommendations as string[],
        confidence: data.result.confidence as number,
        raw_result: data.result,
      }
      onComplete(result)
    } else if (data.step === 'error') {
      onError(data.message)
    } else {
      onProgress(data)
    }
  }
  
  ws.onerror = () => {
    onError('WebSocket connection failed')
  }
  
  ws.onclose = () => {
    // Connection closed
  }
  
  // Return cleanup function
  return () => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.close()
    }
  }
}

/**
 * Run a full investigation using the Investigation Agent (HTTP fallback).
 * Accepts natural language queries like:
 * - "analizza alluvioni Lago Maggiore 2000"
 * - "investigate floods in Po Valley 1994"
 */
export async function investigate(request: InvestigateRequest): Promise<InvestigateResponse> {
  const res = await fetch(`${API_BASE}/investigate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  return res.json()
}

/**
 * Check if Investigation Agent is available and configured.
 */
export async function getInvestigateStatus() {
  const res = await fetch(`${API_BASE}/investigate/status`)
  return res.json()
}

/**
 * Investigation Briefing - shows what data will be downloaded before starting.
 */
export interface InvestigationBriefingData {
  query: string
  event_context: {
    location_name: string
    event_type: string
    start_date: string
    end_date: string
    keywords: string[]
  }
  location: {
    lat: number
    lon: number
    bbox: [number, number, number, number]
    name: string
    country: string
    region: string
  } | null
  data_requests: Array<{
    source: string
    name: string
    variables: string[]
    lat_range: [number, number] | null
    lon_range: [number, number] | null
    time_range: [string, string] | null
    estimated_size_mb: number
    estimated_time_sec: number
    description: string
  }>
  total_estimated_size_mb: number
  total_estimated_time_sec: number
  summary: string
}

/**
 * Create an investigation briefing before starting the download.
 * Returns estimated sizes, times, and what data will be collected.
 */
export async function createInvestigationBriefing(
  query: string,
  options?: {
    collect_satellite?: boolean
    collect_reanalysis?: boolean
    collect_climate_indices?: boolean
    collect_papers?: boolean
  }
): Promise<{ status: string; briefing: InvestigationBriefingData }> {
  const res = await fetch(`${API_BASE}/investigate/briefing`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      collect_satellite: options?.collect_satellite ?? true,
      collect_reanalysis: options?.collect_reanalysis ?? true,
      collect_climate_indices: options?.collect_climate_indices ?? true,
      collect_papers: options?.collect_papers ?? true,
    }),
  })
  return res.json()
}

/**
 * Detect if a message is an investigation query (vs general chat).
 * Investigation queries mention locations, dates, events, or use keywords like:
 * "analizza", "investigate", "cerca", "studia", "alluvione", "flood", etc.
 */
export function isInvestigationQuery(message: string): boolean {
  const investigationKeywords = [
    // Italian
    'analizza', 'analisi', 'studia', 'cerca', 'investiga', 'indaga',
    'alluvione', 'alluvioni', 'inondazione', 'esondazione', 'piena',
    'evento', 'eventi', 'estremo', 'estremi',
    'lago', 'fiume', 'mare', 'oceano', 'costa',
    // English
    'analyze', 'analysis', 'study', 'investigate', 'search', 'find',
    'flood', 'flooding', 'inundation', 'surge', 'extreme',
    'event', 'events', 'lake', 'river', 'sea', 'ocean', 'coast',
    // Dates patterns
    '19', '20', 'gennaio', 'febbraio', 'marzo', 'aprile', 'maggio',
    'giugno', 'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre',
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
  ]
  
  const lowerMessage = message.toLowerCase()
  return investigationKeywords.some(kw => lowerMessage.includes(kw))
}

// =========================================================================
// Knowledge Base Endpoints
// =========================================================================

export interface Paper {
  id?: string
  title: string
  authors: string[]
  abstract: string
  doi?: string
  year: number
  journal?: string
  keywords: string[]
  embedding?: number[]
}

export interface HistoricalEvent {
  id?: string
  name: string
  description: string
  event_type: string
  start_date: string
  end_date?: string
  location?: Record<string, unknown>
  severity?: number
  source?: string
}

export interface ClimateIndex {
  id?: string
  name: string
  abbreviation: string
  description: string
  source_url?: string
  time_series?: Record<string, unknown>[]
}

export interface CausalPattern {
  id?: string
  name: string
  description: string
  pattern_type: string
  variables: string[]
  lag_days?: number
  strength?: number
  confidence?: number
  metadata?: Record<string, unknown>
}

// Papers
export async function addPaper(paper: Paper, backend: Backend = 'neo4j') {
  const res = await fetch(`${API_BASE}/knowledge/papers?backend=${backend}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(paper),
  })
  return res.json()
}

export async function searchPapers(query: string, limit = 10, backend: Backend = 'neo4j') {
  const res = await fetch(
    `${API_BASE}/knowledge/papers/search?query=${encodeURIComponent(query)}&limit=${limit}&backend=${backend}`
  )
  return res.json()
}

// Events
export async function addEvent(event: HistoricalEvent, backend: Backend = 'neo4j') {
  const res = await fetch(`${API_BASE}/knowledge/events?backend=${backend}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(event),
  })
  return res.json()
}

export async function searchEvents(
  params: { event_type?: string; start_date?: string; end_date?: string; limit?: number },
  backend: Backend = 'neo4j'
) {
  const searchParams = new URLSearchParams()
  if (params.event_type) searchParams.set('event_type', params.event_type)
  if (params.start_date) searchParams.set('start_date', params.start_date)
  if (params.end_date) searchParams.set('end_date', params.end_date)
  if (params.limit) searchParams.set('limit', params.limit.toString())
  searchParams.set('backend', backend)
  
  const res = await fetch(`${API_BASE}/knowledge/events/search?${searchParams}`)
  return res.json()
}

// Climate Indices
export async function listClimateIndices(backend: Backend = 'neo4j') {
  const res = await fetch(`${API_BASE}/knowledge/climate-indices?backend=${backend}`)
  return res.json()
}

export async function getClimateIndex(indexId: string, backend: Backend = 'neo4j') {
  const res = await fetch(`${API_BASE}/knowledge/climate-indices/${indexId}?backend=${backend}`)
  return res.json()
}

// Patterns
export async function addPattern(pattern: CausalPattern, backend: Backend = 'neo4j') {
  const res = await fetch(`${API_BASE}/knowledge/patterns?backend=${backend}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(pattern),
  })
  return res.json()
}

export async function searchPatterns(
  params: { pattern_type?: string; variables?: string; min_confidence?: number; limit?: number },
  backend: Backend = 'neo4j'
) {
  const searchParams = new URLSearchParams()
  if (params.pattern_type) searchParams.set('pattern_type', params.pattern_type)
  if (params.variables) searchParams.set('variables', params.variables)
  if (params.min_confidence) searchParams.set('min_confidence', params.min_confidence.toString())
  if (params.limit) searchParams.set('limit', params.limit.toString())
  searchParams.set('backend', backend)
  
  const res = await fetch(`${API_BASE}/knowledge/patterns/search?${searchParams}`)
  return res.json()
}

// Graph Traversal
export async function getCausalChain(patternId: string, maxDepth = 5, backend: Backend = 'neo4j') {
  const res = await fetch(
    `${API_BASE}/knowledge/patterns/${patternId}/causal-chain?max_depth=${maxDepth}&backend=${backend}`
  )
  return res.json()
}

export async function getPatternEvidence(patternId: string, backend: Backend = 'neo4j') {
  const res = await fetch(`${API_BASE}/knowledge/patterns/${patternId}/evidence?backend=${backend}`)
  return res.json()
}

export async function getTeleconnections(indexId: string, minCorrelation = 0.5, backend: Backend = 'neo4j') {
  const res = await fetch(
    `${API_BASE}/knowledge/climate-indices/${indexId}/teleconnections?min_correlation=${minCorrelation}&backend=${backend}`
  )
  return res.json()
}

// Statistics
export async function getKnowledgeStats(backend: Backend = 'neo4j') {
  const res = await fetch(`${API_BASE}/knowledge/stats?backend=${backend}`)
  return res.json()
}

export async function compareBackends() {
  const res = await fetch(`${API_BASE}/knowledge/compare`)
  return res.json()
}

// =========================================================================
// Health
// =========================================================================

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`)
  return res.json()
}

// =========================================================================
// Historical Analysis Endpoints
// =========================================================================

export interface HistoricalEpisode {
  id: string
  name: string
  event_type: string
  start_date: string
  end_date: string
  description: string
  precursor_window_days: number
  region: { lat: [number, number]; lon: [number, number] }
  known_precursors: string[]
  references: string[]
}

export interface PrecursorSignal {
  variable: string
  source_region: string
  lag_days: number
  correlation: number
  p_value: number
  physics_validated: boolean
  mechanism: string
  confidence: number
}

export interface AnalysisResult {
  episode_id: string
  precursors: PrecursorSignal[]
  overall_confidence: number
  max_lead_time: number
  validated_count: number
  analysis_timestamp: string
}

export async function listHistoricalEpisodes(): Promise<{ episodes: HistoricalEpisode[]; count: number }> {
  const res = await fetch(`${API_BASE}/historical/episodes`)
  return res.json()
}

export async function getHistoricalEpisode(episodeId: string): Promise<HistoricalEpisode> {
  const res = await fetch(`${API_BASE}/historical/episodes/${episodeId}`)
  return res.json()
}

export async function analyzeHistoricalEpisode(episodeId: string): Promise<AnalysisResult> {
  const res = await fetch(`${API_BASE}/historical/analyze/${episodeId}`, {
    method: 'POST',
  })
  return res.json()
}

export async function getCrossPatterns(): Promise<{
  cross_patterns: Array<{
    variable: string
    appearances: string[]
    average_lag_days: number
    average_correlation: number
    mechanism: string
    predictive_reliability: string
  }>
  recommendation: string
}> {
  const res = await fetch(`${API_BASE}/historical/cross-patterns`)
  return res.json()
}

// =========================================================================
// Data Manager Endpoints
// =========================================================================

export interface DataSource {
  name: string
  description: string
  enabled: boolean
  connected: boolean
  variables: string[]
  default_resolution: {
    temporal: string
    spatial: string
  }
  min_start_date: string
}

export interface DataRequest {
  source: string
  variables: string[]
  lat_range: [number, number]
  lon_range: [number, number]
  time_range: [string, string]
  resolution: {
    temporal: string
    spatial: string
  }
  description: string
  estimated_size_mb: number
  estimated_time_sec: number
}

export interface Briefing {
  query: string
  location_name: string
  location_bbox: [number, number, number, number]
  event_type: string
  time_period: [string, string]
  precursor_period: [string, string]
  data_requests: DataRequest[]
  total_estimated_size_mb: number
  total_estimated_time_sec: number
  cached_sources: string[]
  needs_download: string[]
}

export interface BriefingResponse {
  briefing_id: string
  briefing: Briefing
  summary: string
}

/**
 * Get available data sources and their status.
 */
export async function getDataSources(): Promise<{
  sources: Record<string, DataSource>
  default_resolution: { temporal: string; spatial: string }
}> {
  const res = await fetch(`${API_BASE}/data/sources`)
  return res.json()
}

/**
 * Get available resolution options.
 */
export async function getResolutions(): Promise<{
  temporal: Array<{ value: string; label: string; description: string }>
  spatial: Array<{ value: string; label: string; description: string }>
}> {
  const res = await fetch(`${API_BASE}/data/resolutions`)
  return res.json()
}

/**
 * Create an investigation briefing for user review.
 */
export async function createBriefing(params: {
  query: string
  location_name: string
  location_bbox: [number, number, number, number]
  event_type: string
  event_start: string
  event_end: string
  precursor_days?: number
  resolution?: { temporal: string; spatial: string }
}): Promise<BriefingResponse> {
  const res = await fetch(`${API_BASE}/data/briefing`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  return res.json()
}

/**
 * Confirm a briefing and start download.
 */
export async function confirmBriefing(
  briefingId: string,
  resolution?: { temporal: string; spatial: string }
): Promise<{
  status: string
  job_id?: string
  briefing_id?: string
}> {
  const res = await fetch(`${API_BASE}/data/briefing/${briefingId}/confirm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      briefing_id: briefingId,
      confirmed: true,
      modified_resolution: resolution,
    }),
  })
  return res.json()
}

/**
 * Get download job status.
 */
export async function getDownloadStatus(jobId: string): Promise<{
  status: string
  briefing_id: string
  progress: Record<string, { progress: number; message: string }>
  results: Record<string, string> | null
  error?: string
}> {
  const res = await fetch(`${API_BASE}/data/download/${jobId}/status`)
  return res.json()
}

/**
 * Get cache statistics.
 */
export async function getCacheStats(): Promise<{
  total_entries: number
  total_size_mb: number
  sources: Record<string, { count: number; size_mb: number; accesses: number }>
}> {
  const res = await fetch(`${API_BASE}/data/cache/stats`)
  return res.json()
}

/**
 * Clear cache.
 */
export async function clearCache(source?: string): Promise<{ status: string }> {
  const url = source 
    ? `${API_BASE}/data/cache?source=${source}`
    : `${API_BASE}/data/cache`
  const res = await fetch(url, { method: 'DELETE' })
  return res.json()
}

/**
 * List cache entries.
 */
export async function listCacheEntries(source?: string): Promise<{
  entries: Array<{
    id: string
    source: string
    variables: string[]
    lat_range: [number, number]
    lon_range: [number, number]
    time_range: [string, string]
    resolution_temporal: string
    resolution_spatial: string
    size_mb: number
    created_at: string
    access_count: number
  }>
}> {
  const url = source 
    ? `${API_BASE}/data/cache/entries?source=${source}`
    : `${API_BASE}/data/cache/entries`
  const res = await fetch(url)
  return res.json()
}

/**
 * Load cached data as dataset for analysis.
 */
export async function loadCachedAsDataset(
  entryId: string,
  datasetName?: string
): Promise<{
  status: string
  dataset_name: string
  rows: number
  columns: string[]
}> {
  const res = await fetch(`${API_BASE}/data/cache/load_as_dataset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      entry_id: entryId,
      dataset_name: datasetName 
    }),
  })
  return res.json()
}

/**
 * Update default resolution.
 */
export async function updateResolution(
  temporal: string,
  spatial: string
): Promise<{ status: string; resolution: { temporal: string; spatial: string } }> {
  const res = await fetch(`${API_BASE}/data/resolution`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ temporal, spatial }),
  })
  return res.json()
}
