/**
 * 🔧 Frontend Configuration
 * 
 * Gestione centralizzata degli endpoint API con supporto per:
 * - Environment variables (Vite)
 * - Service discovery
 * - Dynamic endpoint resolution
 * - Multi-environment support
 */

interface ApiConfig {
  baseUrl: string
  wsBaseUrl: string
  apiVersion: string
  fullApiUrl: string
  fullWsUrl: string
  env: 'development' | 'production' | 'staging'
  debug: boolean
}

/**
 * Service Discovery: Auto-detect API endpoint
 * 
 * Priority order:
 * 1. Environment variable (VITE_API_BASE_URL)
 * 2. Window location (same origin)
 * 3. Fallback to localhost
 */
function discoverApiEndpoint(): string {
  // 1. Check environment variable
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL
  }
  
  // 2. Try same origin (for production deployments)
  if (typeof window !== 'undefined') {
    const { protocol, hostname, port } = window.location
    
    // If frontend is served from same domain as API
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return `${protocol}//${hostname}${port ? ':' + port : ''}`
    }
  }
  
  // 3. Fallback to localhost (development)
  return 'http://localhost:8000'
}

/**
 * WebSocket Discovery: Auto-detect WS endpoint
 */
function discoverWsEndpoint(): string {
  if (import.meta.env.VITE_WS_BASE_URL) {
    return import.meta.env.VITE_WS_BASE_URL
  }
  
  if (typeof window !== 'undefined') {
    const { protocol, hostname, port } = window.location
    const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:'
    
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      return `${wsProtocol}//${hostname}${port ? ':' + port : ''}`
    }
  }
  
  return 'ws://localhost:8000'
}

/**
 * Build configuration object
 */
function buildConfig(): ApiConfig {
  const baseUrl = discoverApiEndpoint()
  const wsBaseUrl = discoverWsEndpoint()
  const apiVersion = import.meta.env.VITE_API_VERSION || 'v1'
  const env = (import.meta.env.VITE_ENV || import.meta.env.MODE || 'development') as ApiConfig['env']
  const debug = import.meta.env.VITE_DEBUG === 'true' || env === 'development'
  
  const fullApiUrl = `${baseUrl}/api/${apiVersion}`
  const fullWsUrl = wsBaseUrl
  
  return {
    baseUrl,
    wsBaseUrl,
    apiVersion,
    fullApiUrl,
    fullWsUrl,
    env,
    debug,
  }
}

/**
 * Global configuration singleton
 */
export const config = buildConfig()

/**
 * Debug logging in development
 */
if (config.debug) {
  console.log('🔧 API Configuration:', {
    baseUrl: config.baseUrl,
    fullApiUrl: config.fullApiUrl,
    wsBaseUrl: config.wsBaseUrl,
    env: config.env,
  })
}

/**
 * Health check helper
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${config.fullApiUrl}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    })
    return response.ok
  } catch {
    return false
  }
}

/**
 * Get legacy endpoint (without /api/v1 prefix)
 * Used for endpoints still on main.py root
 */
export function getLegacyEndpoint(path: string): string {
  return `${config.baseUrl}${path}`
}

/**
 * Get versioned API endpoint
 */
export function getApiEndpoint(path: string): string {
  return `${config.fullApiUrl}${path}`
}

/**
 * Get WebSocket endpoint
 */
export function getWsEndpoint(path: string): string {
  // fullWsUrl already includes base, just add /api/v1 prefix
  return `${config.wsBaseUrl}/api/${config.apiVersion}${path}`
}

export default config
