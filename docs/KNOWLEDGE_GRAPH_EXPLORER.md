# 🌌 Knowledge Graph Explorer - Roadmap

**Visione**: Esplorazione visuale 3D del knowledge base con AI-powered discovery di correlazioni nascoste.

**Tecnologia Core**: [Cosmograph](https://cosmograph.app/) / [cosmos.gl](https://github.com/cosmosgl/graph) by Nikita Rokotyan
- GPU-accelerated WebGL rendering
- Handles 100k+ nodes in real-time
- Force-directed layout
- TypeScript/React ready

---

## 🎯 Use Cases Principali

### 1. **Event-Centric Exploration**
```
Utente seleziona evento (es: "Alluvione Lago Maggiore 2000")
    │
    ├── 📚 Papers correlati (18 trovati)
    │      └── Collegamenti per: keywords, location, event_type
    │
    ├── 📊 Data Sources
    │      ├── CMEMS Sea Level (satellite)
    │      ├── ERA5 Reanalysis (meteo)
    │      └── Climate Indices (NAO, EA, AO)
    │
    ├── 🔗 Pattern Causali
    │      └── NAO- → Precipitation+ → Flood
    │
    └── 🌍 Eventi Simili (temporal/spatial)
           ├── Po Valley 1994
           ├── Ticino 2014
           └── Verbano 1993
```

### 2. **LLM Cockpit Commands**
| Comando | Azione | Esempio |
|---------|--------|---------|
| "Expand geographically" | Trova eventi simili in regioni adiacenti | Piemonte → Lombardia → Veneto |
| "Find physical correlations" | Cerca correlazioni non ancora aggregate | SST Mediterraneo ↔ Precipitazioni Alpine |
| "Show precursors" | Mostra segnali anticipatori | NAO phase 30 giorni prima |
| "Current risk assessment" | Condizioni globali simili a precursori storici | "Oggi NAO=-2.1, simile a Oct 2000" |
| "Missing data gaps" | Evidenzia dati non raccolti per limiti tecnologici | "No satellite SLA prima del 1993" |

### 3. **Discovery Questions**
- ✅ "Ci sono correlazioni fisiche tra dati non ancora aggregati?"
- ✅ "Esistono condizioni attuali simili ai precursori storici?"
- ✅ "Quali precursori non sono stati documentati per limiti tecnologici?"
- ✅ "Quali pattern emergono cross-region?"

---

## 🏗️ Architettura Tecnica

### Graph Schema
```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH NODES                        │
├─────────────────────────────────────────────────────────────────┤
│  📍 EVENT                                                       │
│     - id, name, type, location, start_date, end_date           │
│     - severity, source, description                             │
│                                                                 │
│  📚 PAPER                                                       │
│     - id, title, authors, year, doi, abstract                  │
│     - keywords, embedding (384-dim vector)                      │
│                                                                 │
│  📊 DATA_SOURCE                                                 │
│     - id, source_type (satellite/reanalysis/index)             │
│     - variables, time_range, spatial_extent                    │
│                                                                 │
│  🔗 PATTERN                                                     │
│     - id, pattern_type, variables, confidence                  │
│     - lag_days, description                                     │
│                                                                 │
│  🌡️ CLIMATE_INDEX                                              │
│     - id, name (NAO, ENSO, AO, EA, etc.)                       │
│     - current_value, historical_series                         │
│                                                                 │
│  🌍 LOCATION                                                    │
│     - id, name, lat, lon, bbox, region, country                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH EDGES                        │
├─────────────────────────────────────────────────────────────────┤
│  PAPER ──[DISCUSSES]──► EVENT                                   │
│  PAPER ──[USES_DATA]──► DATA_SOURCE                            │
│  PAPER ──[CITES]──► PAPER                                       │
│  EVENT ──[OCCURRED_AT]──► LOCATION                             │
│  EVENT ──[HAS_PRECURSOR]──► CLIMATE_INDEX                      │
│  EVENT ──[SIMILAR_TO]──► EVENT (semantic/temporal)             │
│  PATTERN ──[EXPLAINS]──► EVENT                                  │
│  PATTERN ──[INVOLVES]──► CLIMATE_INDEX                         │
│  DATA_SOURCE ──[COVERS]──► LOCATION                            │
│  DATA_SOURCE ──[MEASURES]──► CLIMATE_INDEX                     │
└─────────────────────────────────────────────────────────────────┘
```

### Frontend Integration (cosmos.gl)
```typescript
// frontend/src/components/KnowledgeGraphExplorer.tsx
import { Graph } from '@cosmos.gl/graph'

interface GraphNode {
  id: string
  type: 'event' | 'paper' | 'data_source' | 'pattern' | 'climate_index' | 'location'
  label: string
  x?: number
  y?: number
  size?: number
  color?: string
  metadata?: Record<string, any>
}

interface GraphEdge {
  source: string
  target: string
  type: 'DISCUSSES' | 'USES_DATA' | 'CITES' | 'OCCURRED_AT' | 'HAS_PRECURSOR' | 'SIMILAR_TO' | 'EXPLAINS' | 'INVOLVES'
  weight?: number
}

const config = {
  spaceSize: 8192,
  simulationFriction: 0.15,
  simulationGravity: 0.1,
  simulationRepulsion: 1.0,
  curvedLinks: true,
  fitViewOnInit: true,
  pointSizeScale: 2,
  linkWidthScale: 1,
  
  // Color scheme by node type
  pointColor: (index: number) => {
    const types = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']
    return types[nodes[index].typeIndex]
  },
  
  // Click handler
  onClick: (pointIndex: number) => {
    const node = nodes[pointIndex]
    onNodeSelect(node)
  }
}
```

### Backend API Extensions
```python
# api/routers/graph_router.py

@router.get("/graph/event/{event_id}")
async def get_event_graph(
    event_id: str,
    depth: int = 2,
    include_papers: bool = True,
    include_data: bool = True,
    include_similar: bool = True,
    backend: KnowledgeBackend = KnowledgeBackend.SURREALDB
) -> GraphResponse:
    """
    Get graph centered on an event.
    Returns nodes and edges for Cosmograph visualization.
    """
    
@router.post("/graph/expand")
async def expand_graph(
    center_node_id: str,
    expansion_type: Literal["geographic", "temporal", "semantic", "causal"],
    radius: float = 1.0,  # degrees for geo, days for temporal, similarity for semantic
) -> GraphExpansionResponse:
    """
    LLM Cockpit: Expand graph in a specific direction.
    """

@router.post("/graph/discover")
async def discover_correlations(
    node_ids: List[str],
    discovery_type: Literal["physical", "precursor", "gap_analysis", "risk_assessment"],
    use_llm: bool = True,
) -> DiscoveryResponse:
    """
    LLM-powered discovery of hidden correlations.
    """

@router.get("/graph/current-risk")
async def assess_current_risk(
    event_type: str = "flood",
    region: Optional[str] = None,
) -> RiskAssessmentResponse:
    """
    Compare current climate conditions to historical precursors.
    """
```

---

## 📋 Implementation Phases

### Phase 1: Basic Graph View (1-2 weeks)
- [ ] Install `@cosmos.gl/graph` in frontend
- [ ] Create `KnowledgeGraphExplorer` component
- [ ] API endpoint: `GET /graph/event/{id}` returns nodes/edges
- [ ] Basic node coloring by type
- [ ] Click to show node details panel

### Phase 2: Event-Centric Exploration (1 week)
- [ ] "View in Graph" button from Investigation results
- [ ] Show connected papers, data sources, patterns
- [ ] Highlight causal chains
- [ ] Filter by node type

### Phase 3: LLM Cockpit Integration (2 weeks)
- [ ] "Expand geographically" command → adjacent regions
- [ ] "Find correlations" → LLM analyzes unlinked nodes
- [ ] "Show precursors" → historical climate indices
- [ ] Natural language graph queries

### Phase 4: Discovery Engine (2 weeks)
- [ ] "Missing data gaps" → identify what wasn't measured
- [ ] "Current risk assessment" → compare today vs historical
- [ ] "Cross-region patterns" → semantic similarity clustering
- [ ] Export graph as embeddable widget

### Phase 5: Time Dimension (1 week)
- [ ] Timeline slider for temporal exploration
- [ ] Animate graph evolution over time
- [ ] Show precursor → event → aftermath sequence

---

## 🎨 UI/UX Design

### Main View
```
┌────────────────────────────────────────────────────────────────┐
│  Knowledge Graph Explorer                    [🔍] [⚙️] [📤]   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                                                          │ │
│  │                    3D GRAPH VIEW                        │ │
│  │                   (Cosmograph)                          │ │
│  │                                                          │ │
│  │       📚──────📍──────📊                                │ │
│  │       │       │       │                                  │ │
│  │       └───🔗──┴───🌡️──┘                                │ │
│  │                                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌────────────────────┐  ┌──────────────────────────────────┐ │
│  │ 🎛️ LLM Cockpit    │  │ 📋 Selected Node                │ │
│  │                    │  │                                  │ │
│  │ [Expand Geo]       │  │ Alluvione Lago Maggiore 2000    │ │
│  │ [Find Correlations]│  │ Type: flood                     │ │
│  │ [Show Precursors]  │  │ Location: 45.9°N, 8.6°E        │ │
│  │ [Current Risk]     │  │ Period: Oct 10-20, 2000        │ │
│  │ [Gap Analysis]     │  │                                  │ │
│  │                    │  │ Connected: 18 papers, 2 sources │ │
│  │ Ask AI: [________] │  │ Patterns: NAO→Precip→Flood     │ │
│  └────────────────────┘  └──────────────────────────────────┘ │
│                                                                │
│  Timeline: [1990|----●----|2025]  ▶️ Play                    │
└────────────────────────────────────────────────────────────────┘
```

### Node Type Colors
| Type | Color | Hex |
|------|-------|-----|
| Event | Blue | `#3B82F6` |
| Paper | Green | `#10B981` |
| Data Source | Amber | `#F59E0B` |
| Pattern | Red | `#EF4444` |
| Climate Index | Purple | `#8B5CF6` |
| Location | Pink | `#EC4899` |

### Edge Type Styles
| Type | Style | Width |
|------|-------|-------|
| DISCUSSES | Solid | 1 |
| USES_DATA | Dashed | 1 |
| CITES | Dotted | 0.5 |
| SIMILAR_TO | Curved | 2 |
| HAS_PRECURSOR | Arrow | 2 |
| EXPLAINS | Bold | 3 |

---

## 🔮 Future Enhancements

### AI-Powered Features
1. **Embedding Clustering**: Papers/events clustered by semantic similarity
2. **Anomaly Detection**: Highlight unusual patterns
3. **Prediction Mode**: "What if NAO drops to -3?"
4. **Auto-Discovery**: Background job finds new correlations

### Data Integration
1. **Real-time Climate**: Live NAO/ENSO/AO values
2. **News Integration**: Recent flood events auto-added
3. **Satellite Feeds**: Near-real-time CMEMS data
4. **Social Signals**: Twitter/X flood mentions

### Collaboration
1. **Shared Graphs**: Team annotations
2. **Export**: PNG, SVG, interactive embed
3. **Reports**: Auto-generated from graph exploration

---

## 📚 References

- [Cosmograph App](https://cosmograph.app/)
- [cosmos.gl GitHub](https://github.com/cosmosgl/graph)
- [Cosmograph Docs](https://cosmograph.app/docs-general)
- [Force-Directed Graph Layouts](https://en.wikipedia.org/wiki/Force-directed_graph_drawing)

---

**Autore**: NICO Project Team  
**Data**: 2025-12-25  
**Status**: 📝 Planning
