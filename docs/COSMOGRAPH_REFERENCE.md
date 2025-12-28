# Cosmograph React v2.0 Reference

> ⚠️ **DEPRECATED** (Dec 2025): Cosmograph è stato rimpiazzato da D3.js/react-force-graph-3d.
> Vedi commit `7318bd6` per la migrazione. Questo file è mantenuto per riferimento storico.

Reference documentation for `@cosmograph/react` v2.0.1 integration.

## Installation

```bash
npm install @cosmograph/react
```

## Core Pattern - prepareCosmographData

Cosmograph v2.0 requires data preparation via `prepareCosmographData()`:

```tsx
import { Cosmograph, prepareCosmographData, CosmographConfig } from '@cosmograph/react'

// Raw data arrays
const rawPoints = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]
const rawLinks = [
  { source: 'a', target: 'b' },
  { source: 'b', target: 'c' },
]

// Data config maps your fields to Cosmograph's internal format
const dataConfig = {
  points: {
    pointIdBy: 'id',           // REQUIRED: unique identifier field
    pointLabelBy: 'label',     // optional: label field
    pointColorBy: 'color',     // optional: color field
    pointSizeBy: 'size',       // optional: size field
  },
  links: {
    linkSourceBy: 'source',    // REQUIRED: source node id field
    linkTargetsBy: ['target'], // REQUIRED: array of target fields!
  },
}

// Prepare data - returns { points, links, cosmographConfig }
const result = await prepareCosmographData(dataConfig, rawPoints, rawLinks)

if (result) {
  const { points, links, cosmographConfig } = result
  // Use these in Cosmograph component
  setConfig({ points, links, ...cosmographConfig })
}
```

## React Component Usage

### Basic Setup

```tsx
import { Cosmograph, CosmographProvider, prepareCosmographData } from '@cosmograph/react'
import type { CosmographRef, CosmographConfig } from '@cosmograph/react'

function GraphView() {
  const [config, setConfig] = useState<CosmographConfig>({})
  
  useEffect(() => {
    const loadData = async () => {
      const result = await prepareCosmographData(dataConfig, points, links)
      if (result) {
        setConfig({ 
          points: result.points, 
          links: result.links, 
          ...result.cosmographConfig,
          // Additional config options
          backgroundColor: '#0f172a',
          fitViewOnInit: true,
        })
      }
    }
    loadData()
  }, [])

  return (
    <CosmographProvider>
      <Cosmograph {...config} />
    </CosmographProvider>
  )
}
```

### With Ref Access

```tsx
const cosmographRef = useRef<CosmographRef>(null)

// Methods available via ref:
cosmographRef.current?.fitView()
cosmographRef.current?.pause()
cosmographRef.current?.start()
cosmographRef.current?.selectPoint(index)
cosmographRef.current?.unselectPoint()

<Cosmograph ref={cosmographRef} {...config} />
```

### Event Handlers

```tsx
<Cosmograph
  {...config}
  onClick={(pointIndex, pointPosition, event) => {
    // pointIndex: number | undefined
    // pointPosition: [x, y] | undefined
    console.log('Clicked point:', pointIndex)
  }}
  onMouseMove={(pointIndex, pointPosition, event) => {
    // Fires continuously on mouse move
  }}
/>
```

## Companion Components

All companion components must be inside `CosmographProvider`:

### Timeline

```tsx
import { CosmographTimeline } from '@cosmograph/react'

// Points must have a date field
const rawPoints = [
  { id: 'a', date: '2024-01-15' },
  { id: 'b', date: '2024-02-20' },
]

const dataConfig = {
  points: {
    pointIdBy: 'id',
    // Include date field mapping
  },
  // ...
}

<CosmographProvider>
  <Cosmograph {...config} />
  <CosmographTimeline accessor="date" />
</CosmographProvider>
```

### Histogram

```tsx
import { CosmographHistogram } from '@cosmograph/react'

// For numeric field filtering
<CosmographHistogram accessor="value" />
```

### Bars (Categorical)

```tsx
import { CosmographBars } from '@cosmograph/react'

// For categorical filtering
<CosmographBars accessor="category" />
```

### Search

```tsx
import { CosmographSearch } from '@cosmograph/react'

<CosmographSearch />
```

### Legends

```tsx
import { 
  CosmographSizeLegend,
  CosmographTypeColorLegend,
  CosmographRangeColorLegend 
} from '@cosmograph/react'

<CosmographSizeLegend />
<CosmographTypeColorLegend accessor="type" />
<CosmographRangeColorLegend accessor="value" />
```

### Control Buttons

```tsx
import {
  CosmographButtonFitView,
  CosmographButtonPlayPause,
  CosmographButtonZoomInOut,
  CosmographButtonRectangularSelection,
  CosmographButtonPolygonalSelection,
} from '@cosmograph/react'

<CosmographButtonFitView />
<CosmographButtonPlayPause />
```

## useCosmograph Hook

Access Cosmograph instance from child components:

```tsx
import { useCosmograph } from '@cosmograph/react'

function Controls() {
  const { cosmograph } = useCosmograph()
  
  return (
    <button onClick={() => cosmograph?.fitView()}>
      Fit View
    </button>
  )
}
```

## Configuration Options

Common `CosmographConfig` properties:

```tsx
{
  // Data (set via prepareCosmographData)
  points: PreparedData,
  links: PreparedData,
  
  // Visual
  backgroundColor: '#000000',
  pointSize: 4,
  pointSizeRange: [2, 10],
  linkWidth: 1,
  linkArrows: false,
  
  // Layout
  spaceSize: 4096,
  fitViewOnInit: true,
  fitViewDelay: 500,
  
  // Simulation
  simulationGravity: 0.1,
  simulationRepulsion: 1,
  simulationLinkSpring: 1,
  simulationLinkDistance: 10,
  simulationFriction: 0.9,
  
  // Labels
  showLabels: true,
  showLabelsFor: 100,           // Show top N labels
  labelWeight: (p) => p.value,  // Weight function
  
  // Interaction
  disableZoom: false,
  disableDrag: false,
  enableClustering: true,
}
```

## Data Formats Supported

- Array of objects
- CSV/TSV files
- Parquet files
- Arrow files
- JSON files
- URL strings to files
- Apache Arrow Table
- DuckDB table name

## Example: Complete Implementation

```tsx
import React, { useEffect, useState, useRef } from 'react'
import {
  Cosmograph,
  CosmographProvider,
  CosmographTimeline,
  CosmographSearch,
  CosmographButtonFitView,
  CosmographButtonPlayPause,
  prepareCosmographData,
} from '@cosmograph/react'
import type { CosmographRef, CosmographConfig } from '@cosmograph/react'

interface Point {
  id: string
  label: string
  type: string
  color: string
  date?: string
}

interface Link {
  source: string
  target: string
}

export function KnowledgeGraphExplorer() {
  const cosmographRef = useRef<CosmographRef>(null)
  const [config, setConfig] = useState<CosmographConfig>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const loadGraph = async () => {
      // Fetch data
      const response = await fetch('/api/graph')
      const data = await response.json()
      
      // Transform to Cosmograph format
      const points: Point[] = data.nodes.map(n => ({
        id: n.id,
        label: n.name,
        type: n.type,
        color: getColorForType(n.type),
        date: n.created_at,
      }))
      
      const links: Link[] = data.edges.map(e => ({
        source: e.from,
        target: e.to,
      }))
      
      // Prepare for Cosmograph
      const dataConfig = {
        points: {
          pointIdBy: 'id',
          pointLabelBy: 'label',
          pointColorBy: 'color',
        },
        links: {
          linkSourceBy: 'source',
          linkTargetsBy: ['target'],
        },
      }
      
      const result = await prepareCosmographData(dataConfig, points, links)
      
      if (result) {
        setConfig({
          points: result.points,
          links: result.links,
          ...result.cosmographConfig,
          backgroundColor: '#0f172a',
          fitViewOnInit: true,
          pointSize: 6,
          linkWidth: 0.5,
        })
      }
      
      setLoading(false)
    }
    
    loadGraph()
  }, [])

  if (loading) return <div>Loading...</div>

  return (
    <CosmographProvider>
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        {/* Graph */}
        <div style={{ flex: 1 }}>
          <Cosmograph ref={cosmographRef} {...config} />
        </div>
        
        {/* Timeline */}
        <CosmographTimeline accessor="date" />
        
        {/* Controls */}
        <div style={{ display: 'flex', gap: 8 }}>
          <CosmographButtonFitView />
          <CosmographButtonPlayPause />
          <CosmographSearch />
        </div>
      </div>
    </CosmographProvider>
  )
}
```

## Important Notes

1. **Always use `prepareCosmographData`** - Direct props like `pointIndexBy` won't work
2. **`linkTargetsBy` is an array** - Even for single target: `['target']`
3. **Companion components need `CosmographProvider`** - Wrap everything
4. **`onClick` receives index** - Use index to look up original data
5. **Data stays local** - Cosmograph uses DuckDB in-browser

## Links

- [NPM Package](https://www.npmjs.com/package/@cosmograph/react)
- [Examples Gallery](https://cosmograph.app/dev/)
- [Documentation](https://cosmograph.app/docs-general/)
- [GitHub Issues](https://github.com/cosmograph-org/cosmograph-issues)
- [Discord](https://discord.gg/Rv8RUQuzsx)
