/**
 * Causal Graph View - D3.js Force-Directed Graph
 * Swiss Design with Blue-Emerald theme
 */

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { useStore } from '../store'
import { Maximize2, ZoomIn, ZoomOut, RefreshCw } from 'lucide-react'

interface Link {
  source: string
  target: string
  lag: number
  strength: number
  pValue: number
}

// Demo data for visualization
const demoData = {
  nodes: [
    { id: 'SST', group: 1 },
    { id: 'SSH', group: 1 },
    { id: 'NAO', group: 2 },
    { id: 'Wind', group: 3 },
    { id: 'Precipitation', group: 3 },
    { id: 'SLA', group: 1 },
    { id: 'Current', group: 1 },
    { id: 'AMO', group: 2 },
  ],
  links: [
    { source: 'NAO', target: 'SST', lag: 3, strength: 0.75, pValue: 0.001 },
    { source: 'NAO', target: 'Wind', lag: 1, strength: 0.82, pValue: 0.0001 },
    { source: 'Wind', target: 'SSH', lag: 2, strength: 0.68, pValue: 0.005 },
    { source: 'SST', target: 'Precipitation', lag: 5, strength: 0.45, pValue: 0.02 },
    { source: 'SSH', target: 'SLA', lag: 0, strength: 0.92, pValue: 0.0001 },
    { source: 'AMO', target: 'SST', lag: 12, strength: 0.55, pValue: 0.01 },
    { source: 'Current', target: 'SST', lag: 1, strength: 0.38, pValue: 0.03 },
    { source: 'Wind', target: 'Current', lag: 0, strength: 0.71, pValue: 0.002 },
  ],
}

function getStrengthColor(strength: number): string {
  if (strength >= 0.7) return '#059669' // emerald-600 - strong
  if (strength >= 0.5) return '#2563eb' // blue-500 - moderate
  if (strength >= 0.3) return '#94a3b8' // slate-400 - weak
  return '#dc2626' // red-600 - negative/very weak
}

function getStrengthClass(strength: number): string {
  if (strength >= 0.7) return 'Strong'
  if (strength >= 0.5) return 'Moderate'
  if (strength >= 0.3) return 'Weak'
  return 'Very Weak'
}

export function CausalGraphView() {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const { setSelectedNode, setSelectedLink } = useStore()
  const [hoveredLink, setHoveredLink] = useState<Link | null>(null)

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .attr('class', 'graph-svg')

    // Define arrow markers
    svg.append('defs').selectAll('marker')
      .data(['strong', 'moderate', 'weak'])
      .join('marker')
      .attr('id', d => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('fill', d => {
        if (d === 'strong') return '#059669'
        if (d === 'moderate') return '#2563eb'
        return '#94a3b8'
      })
      .attr('d', 'M0,-5L10,0L0,5')

    // Create force simulation
    const simulation = d3.forceSimulation(demoData.nodes as d3.SimulationNodeDatum[])
      .force('link', d3.forceLink(demoData.links)
        .id((d: any) => d.id)
        .distance(120))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(40))

    // Draw links
    const link = svg.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(demoData.links)
      .join('line')
      .attr('class', 'link')
      .attr('stroke', d => getStrengthColor(d.strength))
      .attr('stroke-width', d => Math.max(1, d.strength * 4))
      .attr('marker-end', d => {
        const cls = getStrengthClass(d.strength).toLowerCase()
        return `url(#arrow-${cls === 'very weak' ? 'weak' : cls})`
      })
      .on('mouseover', (_event, d) => setHoveredLink(d))
      .on('mouseout', () => setHoveredLink(null))
      .on('click', (_event, d) => setSelectedLink({
        source: (d.source as any).id || d.source as string,
        target: (d.target as any).id || d.target as string,
        lag: d.lag,
        strength: d.strength,
        pValue: d.pValue,
      }))

    // Draw link labels (lag)
    const linkLabels = svg.append('g')
      .attr('class', 'link-labels')
      .selectAll('text')
      .data(demoData.links)
      .join('text')
      .attr('class', 'link-label')
      .attr('text-anchor', 'middle')
      .text(d => d.lag > 0 ? `τ=${d.lag}d` : '')

    // Draw nodes
    const node = svg.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(demoData.nodes)
      .join('g')
      .attr('class', 'node')
      .call(d3.drag<any, any>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))
      .on('click', (_event, d) => setSelectedNode(d.id))

    // Node circles
    node.append('circle')
      .attr('r', 10)
      .attr('fill', d => {
        if (d.group === 1) return '#2563eb' // Ocean vars
        if (d.group === 2) return '#059669' // Climate indices
        return '#06b6d4' // Atmospheric
      })
      .attr('stroke', '#0f172a')
      .attr('stroke-width', 2)

    // Node labels
    node.append('text')
      .text(d => d.id)
      .attr('dy', -16)
      .attr('text-anchor', 'middle')

    // Simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)

      linkLabels
        .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
        .attr('y', (d: any) => (d.source.y + d.target.y) / 2 - 5)

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`)
    })

    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      event.subject.fx = event.subject.x
      event.subject.fy = event.subject.y
    }

    function dragged(event: any) {
      event.subject.fx = event.x
      event.subject.fy = event.y
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0)
      event.subject.fx = null
      event.subject.fy = null
    }

    return () => {
      simulation.stop()
    }
  }, [setSelectedNode, setSelectedLink])

  return (
    <div className="card flex flex-col h-[600px]">
      {/* Header */}
      <div className="card-header flex items-center justify-between">
        <div>
          <h3 className="text-phi-lg font-semibold text-slate-900">
            Causal Graph
          </h3>
          <p className="text-phi-sm text-slate-500">
            {demoData.nodes.length} variables • {demoData.links.length} causal links
          </p>
        </div>
        <div className="flex items-center gap-phi-xs">
          <button className="btn btn-ghost btn-sm">
            <ZoomOut className="w-4 h-4" />
          </button>
          <button className="btn btn-ghost btn-sm">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button className="btn btn-ghost btn-sm">
            <RefreshCw className="w-4 h-4" />
          </button>
          <button className="btn btn-ghost btn-sm">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Graph Container */}
      <div ref={containerRef} className="flex-1 graph-container m-phi-lg">
        <svg ref={svgRef} />
        
        {/* Hover Tooltip */}
        {hoveredLink && (
          <div className="absolute bottom-phi-lg left-phi-lg bg-slate-800 text-white px-phi-md py-phi-sm rounded-swiss text-phi-sm">
            <div className="font-medium">
              {(hoveredLink.source as any).id || hoveredLink.source} → {(hoveredLink.target as any).id || hoveredLink.target}
            </div>
            <div className="text-slate-300 text-phi-xs mt-1">
              Strength: {(hoveredLink.strength * 100).toFixed(0)}% • 
              Lag: {hoveredLink.lag} days • 
              p-value: {hoveredLink.pValue.toFixed(4)}
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="px-phi-lg pb-phi-lg flex items-center gap-phi-xl text-phi-xs">
        <div className="flex items-center gap-phi-sm">
          <div className="w-8 h-0.5 bg-emerald-600" />
          <span className="text-slate-600">Strong (≥70%)</span>
        </div>
        <div className="flex items-center gap-phi-sm">
          <div className="w-8 h-0.5 bg-blue-500" />
          <span className="text-slate-600">Moderate (50-70%)</span>
        </div>
        <div className="flex items-center gap-phi-sm">
          <div className="w-8 h-0.5 bg-slate-400" />
          <span className="text-slate-600">Weak (&lt;50%)</span>
        </div>
        <div className="flex items-center gap-phi-sm ml-auto">
          <span className="text-slate-500">τ = time lag (days)</span>
        </div>
      </div>
    </div>
  )
}
