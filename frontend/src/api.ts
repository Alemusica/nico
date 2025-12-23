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
