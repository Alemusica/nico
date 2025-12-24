# 🔗 Agent Layer Architecture

## Overview

The Agent Layer models **intermediate actors** that mediate causality between patterns and outcomes. This is a "second layer" in causal discovery that captures the entities that:
1. **Receive** signals/patterns from the environment
2. **Have capabilities** (what they can do)
3. **Are constrained** (physical laws, regulations, capacity)
4. **Take actions** that result in outcomes

```
┌─────────────────────────────────────────────────────────────────┐
│                     CAUSAL FLOW                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Pattern ──→ Agent (state) ──→ Action ──→ Outcome              │
│      │           │                │            │                 │
│      │           │                │            │                 │
│      │      Capabilities      Parameters    Event/Effect         │
│      │      Constraints       Timestamp                          │
│      │                                                           │
│   Examples:                                                      │
│   • High temp → Operator(tired) → Wrong speed → Defect          │
│   • Cold snap → City(heating) → Thermal emission → Vortex       │
│   • NAO index → Arctic(ice) → Melt rate → Sea level             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Types

```python
class AgentType(Enum):
    OPERATOR = "operator"           # Human operators
    INFRASTRUCTURE = "infrastructure"  # Cities, plants, systems
    PHYSICAL_PROCESS = "physical_process"  # Thermal convection, ocean currents
    CLIMATE_PATTERN = "climate_pattern"  # NAO, ENSO, Arctic Oscillation
```

### 1. OPERATOR (Human Actors)

Manufacturing context - humans who interact with machines:

```yaml
Operator:
  id: "operator_mario"
  type: OPERATOR
  capabilities:
    - adjust_speed (range: 60-140 rpm)
    - load_pellets (batch: 25kg)
    - emergency_stop
  constraints:
    - shift_duration: 8 hours (labor law)
    - certification_required: true
    - fatigue_limit: 0.8 (performance degrades)
  states:
    - fatigue: 0.0 → 1.0
    - attention: 0.0 → 1.0
```

### 2. INFRASTRUCTURE (Systems/Cities)

Climate/urban context - systems that emit energy or modify environment:

```yaml
City:
  id: "city_milan"
  type: INFRASTRUCTURE
  capabilities:
    - heat_emission (500-8000 MW)
    - albedo_modification
  constraints:
    - thermodynamics_first_law: energy_conservation
    - thermal_inertia: lag_hours=6
    - heating_capacity: max_8000_mw
  states:
    - heating_efficiency: 0.0 → 1.0
    - demand_level: low/medium/high/extreme
```

### 3. PHYSICAL_PROCESS (Natural Phenomena)

Intermediate physical processes that mediate causality:

```yaml
ThermalVortex:
  id: "vortex_001"
  type: PHYSICAL_PROCESS
  capabilities:
    - vertical_convection (updraft: 2.5 m/s)
    - horizontal_inflow (radius: 30 km)
  constraints:
    - buoyancy_driven: delta_T > 3°C threshold
  operates_on: regional_wind_pattern
```

### 4. CLIMATE_PATTERN (Large-scale Oscillations)

Climate indices that modulate regional effects:

```yaml
NAO:
  id: "nao_index"
  type: CLIMATE_PATTERN
  capabilities:
    - modulate_storm_tracks
    - influence_precipitation
  constraints:
    - quasi_periodic: ~7-10 year cycle
    - physically_linked_to: pressure_gradient
```

## Capability & Constraint Model

### Capabilities (What an agent CAN do)

```python
@dataclass
class Capability:
    name: str              # "adjust_speed", "heat_emission"
    type: CapabilityType   # CONTROL, EMIT, MODULATE, TRANSFORM
    parameters: Dict       # {"range": (60, 140), "unit": "rpm"}
    constraints: List[str] # ["must_not_exceed_140"]
```

**Capability Types:**
- `CONTROL`: Direct manipulation (operator adjusts machine)
- `EMIT`: Energy/mass release (city emits heat)
- `MODULATE`: Modify existing process (albedo changes reflection)
- `TRANSFORM`: Convert energy form (convection lifts air mass)

### Constraints (What LIMITS an agent)

```python
@dataclass
class Constraint:
    name: str              # "thermodynamics_first_law"
    type: ConstraintType   # PHYSICAL_LAW, CAPACITY, TEMPORAL, REGULATORY
    value: Any             # "energy_conservation"
    description: str       # "Heat emission equals energy consumption"
```

**Constraint Types:**
- `PHYSICAL_LAW`: Nature's rules (conservation, buoyancy)
- `CAPACITY`: Maximum limits (power output, speed range)
- `TEMPORAL`: Time constraints (shift duration, lag)
- `REGULATORY`: Human rules (certifications, safety limits)

## Paper ↔ Event Relationships

Bidirectional relationships between papers and events:

```
┌─────────────────────────────────────────────────────────────────┐
│                PAPER ↔ EVENT RELATIONSHIPS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Paper ──[DOCUMENTS]──→ Event                                   │
│   (Paper studies an event that happened)                         │
│                                                                  │
│   Event ──[INSPIRES]──→ Paper                                    │
│   (Event leads to research paper)                                │
│                                                                  │
│   Paper ──[PREDICTS]──→ Event                                    │
│   (Paper predicted event before it occurred)                     │
│                                                                  │
│   Paper ──[VALIDATES]──→ Event                                   │
│   (Paper confirms the causal mechanism of event)                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Graph Schema (Neo4j)

```cypher
// Node Types
(:Agent {id, agent_type, name, capabilities, constraints, metadata})
(:Capability {name, type, parameters, constraints})
(:AgentState {agent_id, state_name, value, timestamp, context})
(:Action {id, capability_used, parameters, timestamp})
(:Event {id, event_type, date, description, magnitude})
(:Pattern {id, source_variable, target_variable, lag_days})
(:Paper {id, title, authors, abstract, doi})

// Relationships
(Agent)-[:HAS_CAPABILITY]->(Capability)
(Agent)-[:HAS_STATE]->(AgentState)
(Agent)-[:OPERATES_ON]->(System|Process)
(Agent)-[:TOOK_ACTION]->(Action)
(Action)-[:RESULTED_IN]->(Event)
(Pattern)-[:INFLUENCES]->(Agent)
(Paper)-[:DOCUMENTS]->(Event)
(Event)-[:INSPIRES]->(Paper)
(Paper)-[:PREDICTS]->(Event)
(Paper)-[:VALIDATES]->(Event)
```

## API Endpoints

### Agent Management

```bash
# Create agent
POST /knowledge/agents
{
  "id": "operator_mario",
  "agent_type": "OPERATOR",
  "name": "Mario Rossi",
  "capabilities": [...],
  "constraints": [...]
}

# Get agent with network
GET /knowledge/agents/{agent_id}

# Update state
POST /knowledge/agents/state
{
  "agent_id": "operator_mario",
  "state_name": "fatigue",
  "value": 0.85
}

# Record action
POST /knowledge/agents/actions
{
  "agent_id": "operator_mario",
  "action_id": "action_001",
  "capability_used": "adjust_speed",
  "parameters": {"new_speed": 85},
  "resulted_in": "event_defect_001"
}
```

### Causal Chain Queries

```bash
# Find causal chains leading to outcome
GET /knowledge/events/{event_id}/causal-chains?max_depth=4

# Response:
{
  "chains": [
    {
      "outcome": {...},
      "action": {"capability_used": "adjust_speed", ...},
      "agent": {"name": "Mario Rossi", "agent_type": "OPERATOR"},
      "agent_states": [{"state_name": "fatigue", "value": 0.85}],
      "causal_narrative": "Mario Rossi (OPERATOR) [state: fatigue=0.8] → adjust_speed → QUALITY_DEFECT"
    }
  ]
}
```

### Panama Papers-style Network Query

```bash
# Find all entities connected to agent
GET /knowledge/agents/{agent_id}/network?max_hops=3

# Response:
{
  "agent": {...},
  "network": {
    "systems_operated": [...],
    "actions_taken": [...],
    "outcomes_caused": [...],
    "influencing_patterns": [...],
    "related_agents": [...],
    "documenting_papers": [...]
  }
}
```

### Pattern Discovery by Agent State

```bash
# Find patterns where agent state correlates with outcomes
GET /knowledge/patterns/by-agent-state?agent_type=OPERATOR&state_name=fatigue&state_threshold=0.7

# Response:
{
  "patterns": [
    {
      "pattern": {
        "agent_type": "OPERATOR",
        "state_factor": "fatigue",
        "avg_state_value": 0.82,
        "outcome_type": "QUALITY_DEFECT"
      },
      "statistics": {
        "occurrence_count": 23,
        "confidence": 0.85
      },
      "sample_cases": [...]
    }
  ]
}
```

## Use Cases

### 1. Manufacturing Quality Investigation

**Question:** "Why do defects increase on Friday night shifts?"

```cypher
MATCH (agent:Agent {agent_type: 'OPERATOR'})
MATCH (agent)-[:HAS_STATE]->(state:AgentState {state_name: 'fatigue'})
WHERE state.value > 0.7
  AND state.context.day_of_week = 'Friday'
  AND state.context.shift = 'night'
MATCH (agent)-[:TOOK_ACTION]->(action:Action)
MATCH (action)-[:RESULTED_IN]->(outcome:Event {event_type: 'QUALITY_DEFECT'})
RETURN agent.name, state.value AS fatigue, action.capability_used, outcome
```

### 2. Climate Impact Assessment

**Question:** "How does Milan's heating affect regional weather?"

```cypher
MATCH (city:Agent {id: 'city_milan', agent_type: 'INFRASTRUCTURE'})
MATCH (city)-[:TOOK_ACTION]->(action:Action {capability_used: 'heat_emission'})
MATCH (action)-[:RESULTED_IN]->(vortex:Agent {agent_type: 'PHYSICAL_PROCESS'})
MATCH (vortex)-[:OPERATES_ON]->(pattern)
MATCH (pattern)-[:CAUSES]->(outcome:Event)
RETURN city.name, action.parameters.emission_mw, vortex.name, outcome
```

### 3. Research Coverage Analysis

**Question:** "What events have been well-documented vs. under-studied?"

```cypher
MATCH (e:Event)
OPTIONAL MATCH (p:Paper)-[:DOCUMENTS]->(e)
WITH e, count(p) AS paper_count
RETURN e.event_type, e.id, paper_count
ORDER BY paper_count ASC
```

## Integration with Existing System

The Agent Layer extends the existing knowledge graph:

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH LAYERS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: Raw Data                                               │
│    └── Measurements, time series, satellite data                 │
│                                                                  │
│  Layer 2: Patterns (existing)                                    │
│    └── CausalPattern, ClimateIndex, correlations                 │
│                                                                  │
│  Layer 3: Agents (NEW)                      ◄── This layer       │
│    └── Operators, Infrastructure, Processes                      │
│    └── Capabilities, Constraints, States                         │
│                                                                  │
│  Layer 4: Evidence                                               │
│    └── Papers, Events, Validation                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

The agent layer provides the **missing link** between observed patterns and real-world outcomes by modeling the intermediate actors that transform signals into effects.
