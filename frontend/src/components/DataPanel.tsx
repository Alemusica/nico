/**
 * Data Panel Component - Swiss Design
 * File explorer and data preview
 */

import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { 
  Upload, 
  File, 
  FileSpreadsheet, 
  Database, 
  ChevronRight,
  Loader2,
  Check,
  HardDrive,
  Play
} from 'lucide-react'
import clsx from 'clsx'
import { 
  listFiles, 
  uploadFile, 
  loadFile, 
  getDataPreview, 
  interpretDataset,
  listCacheEntries,
  loadCachedAsDataset,
  getCacheStats
} from '../api'
import { useStore } from '../store'

export function DataPanel() {
  const { currentDataset, setCurrentDataset, pendingInvestigationResult } = useStore()
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [previewData, setPreviewData] = useState<any[] | null>(null)
  const [interpretation, setInterpretation] = useState<any | null>(null)
  const [showCacheTab, setShowCacheTab] = useState(false)

  // Fetch available files
  const { data: filesData, isLoading: filesLoading, refetch: refetchFiles } = useQuery({
    queryKey: ['files'],
    queryFn: listFiles,
  })
  
  // Fetch cache entries
  const { data: cacheData, refetch: refetchCache } = useQuery({
    queryKey: ['cache-entries'],
    queryFn: () => listCacheEntries(),
  })
  
  // Fetch cache stats
  const { data: cacheStats } = useQuery({
    queryKey: ['cache-stats'],
    queryFn: getCacheStats,
  })

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: uploadFile,
    onSuccess: () => {
      refetchFiles()
    },
  })

  // Load file mutation
  const loadMutation = useMutation({
    mutationFn: loadFile,
    onSuccess: (data) => {
      if (data.success && data.dataset) {
        setCurrentDataset(data.dataset.name)
        setSelectedFile(data.dataset.name)
        fetchPreview(data.dataset.name)
      }
    },
  })
  
  // Load cached data mutation
  const loadCacheMutation = useMutation({
    mutationFn: ({ entryId, name }: { entryId: string; name?: string }) => 
      loadCachedAsDataset(entryId, name),
    onSuccess: (data) => {
      setCurrentDataset(data.dataset_name)
      setSelectedFile(data.dataset_name)
      refetchFiles()
      alert(`Loaded ${data.rows} rows with ${data.columns.length} columns as dataset "${data.dataset_name}"`)
    },
  })

  // Interpret mutation
  const interpretMutation = useMutation({
    mutationFn: interpretDataset,
    onSuccess: (data) => {
      setInterpretation(data)
    },
  })

  const fetchPreview = async (name: string) => {
    try {
      const data = await getDataPreview(name, 20)
      setPreviewData(data)
    } catch (error) {
      console.error('Failed to fetch preview:', error)
    }
  }

  const handleFileSelect = async (filePath: string) => {
    loadMutation.mutate(filePath)
  }

  const handleInterpret = () => {
    if (currentDataset) {
      interpretMutation.mutate(currentDataset)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      uploadMutation.mutate(files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      uploadMutation.mutate(files[0])
    }
  }

  const files = filesData?.files || []
  const cacheEntries = cacheData?.entries || []

  return (
    <div className="grid grid-cols-12 gap-phi-xl">
      {/* Investigation Cache Alert */}
      {pendingInvestigationResult && cacheEntries.length > 0 && (
        <div className="col-span-12 bg-blue-50 border border-blue-200 rounded-lg p-phi-lg">
          <div className="flex items-start gap-phi-md">
            <HardDrive className="w-5 h-5 text-blue-600 mt-1" />
            <div className="flex-1">
              <h4 className="font-semibold text-blue-900">Investigation Data Ready</h4>
              <p className="text-phi-sm text-blue-700 mt-1">
                {cacheEntries.length} cached datasets from {pendingInvestigationResult.location} investigation
              </p>
              <button
                onClick={() => setShowCacheTab(true)}
                className="mt-phi-sm btn btn-sm btn-primary"
              >
                View Cached Data →
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* File Browser */}
      <div className="col-span-4">
        <div className="card">
          {/* Tab switcher */}
          <div className="flex border-b border-slate-200">
            <button
              onClick={() => setShowCacheTab(false)}
              className={clsx(
                'flex-1 px-phi-lg py-phi-md text-phi-sm font-medium transition-colors',
                !showCacheTab 
                  ? 'text-blue-600 border-b-2 border-blue-600' 
                  : 'text-slate-600 hover:text-slate-800'
              )}
            >
              Files
            </button>
            <button
              onClick={() => setShowCacheTab(true)}
              className={clsx(
                'flex-1 px-phi-lg py-phi-md text-phi-sm font-medium transition-colors',
                showCacheTab 
                  ? 'text-emerald-600 border-b-2 border-emerald-600' 
                  : 'text-slate-600 hover:text-slate-800'
              )}
            >
              Cache ({cacheEntries.length})
            </button>
          </div>
          
          {!showCacheTab ? (
            /* File list */
            <>
            <div className="card-header">
              <h3 className="text-phi-lg font-semibold text-slate-900">
                Data Files
              </h3>
              <p className="text-phi-sm text-slate-500 mt-1">
                Select or upload datasets
              </p>
            </div>

            {/* Upload Zone */}
            <div className="p-phi-lg border-b border-slate-100">
            <label
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="flex flex-col items-center justify-center p-phi-xl border-2 border-dashed border-slate-300 rounded-swiss-lg cursor-pointer hover:border-blue-500 hover:bg-blue-50/50 transition-colors"
            >
              <input
                type="file"
                onChange={handleFileInput}
                accept=".csv,.nc,.netcdf"
                className="hidden"
              />
              {uploadMutation.isPending ? (
                <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
              ) : (
                <Upload className="w-8 h-8 text-slate-400 mb-phi-sm" />
              )}
              <span className="text-phi-sm text-slate-600">
                Drop files here or click to upload
              </span>
              <span className="text-phi-xs text-slate-400 mt-1">
                CSV, NetCDF supported
              </span>
            </label>
          </div>

          {/* File List */}
          <div className="flex-1 overflow-y-auto swiss-scrollbar">
            {filesLoading ? (
              <div className="flex items-center justify-center p-phi-xl">
                <Loader2 className="w-6 h-6 text-slate-400 animate-spin" />
              </div>
            ) : files.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-phi-xl text-center">
                <Database className="w-12 h-12 text-slate-300 mb-phi-md" />
                <p className="text-phi-sm text-slate-500">
                  No data files found
                </p>
                <p className="text-phi-xs text-slate-400 mt-1">
                  Upload a file to get started
                </p>
              </div>
            ) : (
              <div className="p-phi-md space-y-phi-xs">
                {files.map((file: any) => (
                  <button
                    key={file.path || file.name}
                    onClick={() => handleFileSelect(file.path || file.name)}
                    className={clsx(
                      'w-full flex items-center gap-phi-md p-phi-md rounded-swiss-lg text-left transition-colors',
                      selectedFile === (file.path || file.name)
                        ? 'bg-blue-50 text-blue-900 border border-blue-200'
                        : 'hover:bg-slate-50 text-slate-700'
                    )}
                  >
                    {file.name?.endsWith('.nc') ? (
                      <Database className="w-5 h-5 text-emerald-500" />
                    ) : (
                      <FileSpreadsheet className="w-5 h-5 text-blue-500" />
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-phi-sm font-medium truncate">
                        {file.name}
                      </p>
                      <p className="text-phi-xs text-slate-500">
                        {file.type || 'unknown'} • {file.size || '—'}
                      </p>
                    </div>
                    <ChevronRight className="w-4 h-4 text-slate-400" />
                  </button>
                ))}
              </div>
            )}
          </div>
          </>
          ) : (
            /* Cache list */
            <>
            <div className="card-header">
              <h3 className="text-phi-lg font-semibold text-slate-900">
                Cached Investigation Data
              </h3>
              <p className="text-phi-sm text-slate-500 mt-1">
                {cacheStats?.total_entries || 0} entries • {cacheStats?.total_size_mb.toFixed(2) || 0} MB
              </p>
            </div>
            
            <div className="flex-1 overflow-y-auto swiss-scrollbar p-phi-md space-y-phi-sm">
              {cacheEntries.length === 0 ? (
                <div className="flex flex-col items-center justify-center p-phi-xl text-center">
                  <HardDrive className="w-12 h-12 text-slate-300 mb-phi-md" />
                  <p className="text-phi-sm text-slate-500">
                    No cached data
                  </p>
                  <p className="text-phi-xs text-slate-400 mt-1">
                    Run an investigation to cache data
                  </p>
                </div>
              ) : (
                cacheEntries.map((entry: any) => (
                  <div
                    key={entry.id}
                    className="p-phi-md bg-white border border-slate-200 rounded-lg hover:border-emerald-300 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-phi-sm">
                      <div className="flex-1">
                        <div className="flex items-center gap-phi-sm">
                          <span className="font-medium text-slate-900">{entry.source}</span>
                          <span className="text-phi-xs px-2 py-0.5 bg-slate-100 text-slate-600 rounded">
                            {entry.size_mb.toFixed(2)} MB
                          </span>
                        </div>
                        <div className="text-phi-xs text-slate-500 mt-1">
                          {entry.variables.join(', ')}
                        </div>
                      </div>
                      <button
                        onClick={() => loadCacheMutation.mutate({ entryId: entry.id })}
                        disabled={loadCacheMutation.isPending}
                        className="btn btn-sm btn-primary flex items-center gap-1"
                      >
                        {loadCacheMutation.isPending ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Play className="w-3 h-3" />
                        )}
                        Load
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-phi-xs text-phi-xs text-slate-600">
                      <div>📍 {entry.lat_range[0].toFixed(2)}° to {entry.lat_range[1].toFixed(2)}°N</div>
                      <div>📅 {entry.time_range[0]} to {entry.time_range[1]}</div>
                      <div>⏱️ {entry.resolution_temporal}</div>
                      <div>📐 {entry.resolution_spatial}°</div>
                    </div>
                  </div>
                ))
              )}
            </div>
            </>
          )}
        </div>
      </div>

      {/* Data Preview */}
      <div className="col-span-8">
        <div className="card h-[calc(100vh-160px)] flex flex-col">
          <div className="card-header flex items-center justify-between">
            <div>
              <h3 className="text-phi-lg font-semibold text-slate-900">
                {currentDataset || 'Data Preview'}
              </h3>
              <p className="text-phi-sm text-slate-500 mt-1">
                {previewData ? `${previewData.length} rows shown` : 'Select a file to preview'}
              </p>
            </div>
            {currentDataset && (
              <button
                onClick={handleInterpret}
                disabled={interpretMutation.isPending}
                className="btn btn-primary btn-sm"
              >
                {interpretMutation.isPending ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  'Interpret with AI'
                )}
              </button>
            )}
          </div>

          {/* Interpretation Results */}
          {interpretation && (
            <div className="px-phi-lg py-phi-md bg-emerald-50 border-b border-emerald-100">
              <div className="flex items-start gap-phi-md">
                <Check className="w-5 h-5 text-emerald-600 mt-0.5" />
                <div>
                  <p className="text-phi-sm font-medium text-emerald-900">
                    AI Interpretation
                  </p>
                  <p className="text-phi-sm text-emerald-700 mt-1">
                    {interpretation.summary}
                  </p>
                  {interpretation.domain && (
                    <span className="badge badge-emerald mt-phi-sm">
                      Domain: {interpretation.domain}
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Table */}
          <div className="flex-1 overflow-auto swiss-scrollbar">
            {previewData && previewData.length > 0 ? (
              <table className="data-grid">
                <thead className="sticky top-0">
                  <tr>
                    {Object.keys(previewData[0]).map((col) => (
                      <th key={col}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((val: any, j) => (
                        <td key={j} className="font-mono text-phi-xs">
                          {typeof val === 'number' 
                            ? val.toFixed(4) 
                            : String(val).slice(0, 50)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <File className="w-16 h-16 text-slate-200 mb-phi-lg" />
                <p className="text-phi-base text-slate-500">
                  No data to display
                </p>
                <p className="text-phi-sm text-slate-400 mt-1">
                  Select a file from the browser to preview its contents
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
