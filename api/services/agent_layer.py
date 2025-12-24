"""
🔧 Agent Layer - Intermediate Causal Actors

Models the "operator" concept: entities that mediate between
systems/patterns and outcomes.

Examples:
- Manufacturing: Operator → tweaks machine → affects product quality
- Oceanography: City heating → creates thermal vortex → affects wind
- Climate: NAO index → modulates jet stream → affects regional weather

Key Concepts:
- Agent: The mediating entity (operator, city, physical process)
- Capability: What the agent CAN do (tweak speed, emit heat)
- Constraint: What limits the agent (physics, fatigue, regulations)
- State: Current condition (tired, inefficient, active)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class AgentType(str, Enum):
    """Types of causal agents."""
    OPERATOR = "operator"           # Human operator (manufacturing)
    INFRASTRUCTURE = "infrastructure"  # City, building, factory
    PHYSICAL_PROCESS = "physical_process"  # Heat transfer, vortex formation
    CLIMATE_PATTERN = "climate_pattern"    # NAO, ENSO as active agents
    INSTRUMENT = "instrument"       # Sensor, satellite
    POLICY = "policy"              # Regulation, decision
    BIOLOGICAL = "biological"       # Ecosystem, species migration


class CapabilityType(str, Enum):
    """Types of agent capabilities."""
    CONTROL = "control"         # Can adjust parameters
    EMIT = "emit"              # Can produce output (heat, pollution)
    MODULATE = "modulate"      # Can amplify/dampen signals
    TRANSFORM = "transform"    # Can change state of something
    OBSERVE = "observe"        # Can measure/record
    TRANSPORT = "transport"    # Can move mass/energy


class ConstraintType(str, Enum):
    """Types of constraints on agents."""
    PHYSICAL_LAW = "physical_law"      # Thermodynamics, conservation
    CAPACITY = "capacity"              # Maximum throughput
    TEMPORAL = "temporal"              # Time-based limits (fatigue, seasons)
    REGULATORY = "regulatory"          # Rules, policies
    RESOURCE = "resource"              # Available materials, energy
    COUPLING = "coupling"              # Dependencies on other agents


@dataclass
class Capability:
    """What an agent CAN do."""
    id: Optional[str] = None
    name: str = ""
    capability_type: CapabilityType = CapabilityType.CONTROL
    description: str = ""
    parameters: list[str] = field(default_factory=list)  # What can be adjusted
    range_min: Optional[float] = None  # Minimum value
    range_max: Optional[float] = None  # Maximum value
    unit: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"cap_{uuid4().hex[:12]}"


@dataclass
class Constraint:
    """What limits an agent."""
    id: Optional[str] = None
    name: str = ""
    constraint_type: ConstraintType = ConstraintType.PHYSICAL_LAW
    description: str = ""
    formula: Optional[str] = None  # Mathematical expression if applicable
    threshold: Optional[float] = None
    unit: Optional[str] = None
    violation_consequence: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"con_{uuid4().hex[:12]}"


@dataclass
class AgentState:
    """Current condition of an agent."""
    id: Optional[str] = None
    name: str = ""
    value: Optional[float] = None
    category: Optional[str] = None  # e.g., "tired", "normal", "stressed"
    timestamp: Optional[datetime] = None
    duration_hours: Optional[float] = None
    affects_capability: Optional[str] = None  # Which capability is affected
    severity: float = 0.5  # 0-1 scale of how much state affects performance
    
    def __post_init__(self):
        if not self.id:
            self.id = f"state_{uuid4().hex[:12]}"


@dataclass 
class Agent:
    """
    An intermediate causal actor that mediates between systems and outcomes.
    
    Manufacturing example:
        Agent(
            name="Operator_A",
            agent_type=AgentType.OPERATOR,
            capabilities=[
                Capability(name="speed_control", parameters=["rpm"], range_max=3000),
                Capability(name="material_loading", parameters=["pellet_type", "quantity"]),
            ],
            constraints=[
                Constraint(name="fatigue", constraint_type=ConstraintType.TEMPORAL,
                          description="Performance degrades after 8h shift"),
            ],
            states=[
                AgentState(name="shift_fatigue", category="tired", severity=0.8),
            ]
        )
    
    Oceanography example:
        Agent(
            name="Milan_Heating_District",
            agent_type=AgentType.INFRASTRUCTURE,
            capabilities=[
                Capability(name="heat_emission", capability_type=CapabilityType.EMIT,
                          parameters=["thermal_output"], unit="MW"),
            ],
            constraints=[
                Constraint(name="thermodynamics", constraint_type=ConstraintType.PHYSICAL_LAW,
                          formula="Q = mcΔT"),
            ],
        )
    """
    id: Optional[str] = None
    name: str = ""
    agent_type: AgentType = AgentType.OPERATOR
    description: str = ""
    
    # What the agent can do
    capabilities: list[Capability] = field(default_factory=list)
    
    # What limits the agent
    constraints: list[Constraint] = field(default_factory=list)
    
    # Current conditions affecting the agent
    states: list[AgentState] = field(default_factory=list)
    
    # Location/context
    location: Optional[dict] = None
    
    # Links to systems it operates on
    operates_on: list[str] = field(default_factory=list)  # Pattern/System IDs
    
    # Metadata
    metadata: Optional[dict] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"agent_{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now()


@dataclass
class AgentAction:
    """A specific action taken by an agent."""
    id: Optional[str] = None
    agent_id: str = ""
    capability_used: str = ""  # Which capability was exercised
    
    # Action details
    action_type: str = ""  # "adjust", "load", "emit", etc.
    parameters: dict = field(default_factory=dict)  # Actual values used
    
    # Timing
    timestamp: datetime = field(default_factory=datetime.now)
    duration_minutes: Optional[float] = None
    
    # Context
    agent_state_during: Optional[str] = None  # State ID
    environmental_conditions: Optional[dict] = None
    
    # Outcome link
    resulted_in: Optional[str] = None  # Event or Pattern ID that resulted
    
    def __post_init__(self):
        if not self.id:
            self.id = f"action_{uuid4().hex[:12]}"


@dataclass
class DocumentsRelation:
    """
    Bidirectional relationship between Papers and Events.
    
    - Paper DOCUMENTS Event (paper was written ABOUT the event)
    - Event INSPIRES Paper (event led to research)
    - Paper PREDICTS Event (paper anticipated what happened)
    - Paper VALIDATES Event (paper confirms event's causal structure)
    """
    id: Optional[str] = None
    paper_id: str = ""
    event_id: str = ""
    
    relation_type: str = "documents"  # documents, inspires, predicts, validates
    direction: str = "paper_to_event"  # or "event_to_paper"
    
    # Temporal relationship
    paper_date: Optional[datetime] = None
    event_date: Optional[datetime] = None
    temporal_gap_days: Optional[int] = None
    
    # Strength of connection
    relevance_score: float = 0.5
    citation_context: Optional[str] = None  # How the paper mentions the event
    
    # Discovery metadata
    discovered_by: Optional[str] = None  # "manual", "llm", "pattern_match"
    confidence: float = 0.5
    
    def __post_init__(self):
        if not self.id:
            self.id = f"docrel_{uuid4().hex[:12]}"
        if self.paper_date and self.event_date:
            self.temporal_gap_days = (self.paper_date - self.event_date).days


# =============================================================================
# Panama Papers Style: Network Analysis Patterns
# =============================================================================

@dataclass
class CausalChainWithAgents:
    """
    Full causal chain including agent mediation.
    
    Example (Manufacturing):
    Pattern(material_viscosity) 
        → Agent(operator, state=tired) 
        → Action(adjusted_speed, wrong_value)
        → Event(defective_batch)
        → Paper(root_cause_analysis)
    
    Example (Oceanography):
    Pattern(urban_heat_island)
        → Agent(city_heating, capabilities=[heat_emission])
        → Action(winter_heating, thermal_output=500MW)
        → Pattern(thermal_vortex)
        → Event(unusual_wind_pattern)
        → Paper(study_on_urban_climate_effects)
    """
    id: Optional[str] = None
    name: str = ""
    description: str = ""
    
    # Chain elements
    source_pattern: Optional[str] = None
    agent: Optional[str] = None
    action: Optional[str] = None
    resulting_pattern: Optional[str] = None
    resulting_event: Optional[str] = None
    documenting_papers: list[str] = field(default_factory=list)
    
    # Analysis
    total_lag_days: Optional[int] = None
    confidence: float = 0.5
    validated: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = f"chain_{uuid4().hex[:12]}"


# =============================================================================
# Query Patterns (inspired by Panama Papers investigations)
# =============================================================================

def build_agent_influence_query_neo4j(agent_id: str) -> str:
    """
    Find all patterns/events influenced by an agent.
    Similar to Panama Papers: "Find all entities connected to this person"
    """
    return f"""
    MATCH (a:Agent {{id: $agent_id}})
    OPTIONAL MATCH (a)-[op:OPERATES_ON]->(p:Pattern)
    OPTIONAL MATCH (a)-[inf:INFLUENCES]->(e:Event)
    OPTIONAL MATCH (a)-[took:TOOK_ACTION]->(act:Action)-[res:RESULTED_IN]->(outcome)
    RETURN a, 
           collect(DISTINCT p) as patterns,
           collect(DISTINCT e) as events,
           collect(DISTINCT {{action: act, outcome: outcome}}) as action_outcomes
    """


def build_agent_influence_query_surreal(agent_id: str) -> str:
    """SurrealDB version of agent influence query."""
    return f"""
    SELECT 
        *,
        ->operates_on->pattern.* AS patterns,
        ->influences->event.* AS events,
        ->took_action->action->resulted_in.* AS outcomes
    FROM agent
    WHERE id = $agent_id
    """


def build_event_paper_bidirectional_query_neo4j() -> str:
    """
    Find bidirectional relationships between events and papers.
    Events that inspired papers, papers that predicted events.
    """
    return """
    MATCH (e:Event)
    OPTIONAL MATCH (p:Paper)-[d:DOCUMENTS]->(e)
    OPTIONAL MATCH (e)-[i:INSPIRED]->(p2:Paper)
    OPTIONAL MATCH (p3:Paper)-[pred:PREDICTED]->(e)
    RETURN e,
           collect(DISTINCT {paper: p, relation: 'documented_by'}) as documented_by,
           collect(DISTINCT {paper: p2, relation: 'inspired'}) as inspired_papers,
           collect(DISTINCT {paper: p3, relation: 'predicted_by'}) as predicted_by
    ORDER BY e.start_date DESC
    """


def build_tired_operator_pattern_query_neo4j() -> str:
    """
    Find patterns where agent state (e.g., fatigue) correlates with outcomes.
    The Friday evening tired operator scenario.
    """
    return """
    MATCH (a:Agent)-[:HAS_STATE]->(s:AgentState)
    WHERE s.category = 'tired' OR s.severity > 0.7
    MATCH (a)-[:TOOK_ACTION]->(act:Action)-[:RESULTED_IN]->(e:Event)
    WHERE e.severity > 0.5
    RETURN a.name as agent,
           s.name as state,
           s.severity as state_severity,
           act.timestamp as action_time,
           e.name as event,
           e.severity as event_severity,
           duration.between(act.timestamp, e.start_date).days as lag_days
    ORDER BY e.severity DESC
    """


def build_heat_cascade_query_neo4j() -> str:
    """
    Find thermal cascade: City heating → vortex → wind pattern.
    The urban heat island scenario.
    """
    return """
    MATCH (city:Agent {agent_type: 'infrastructure'})
    WHERE city.capabilities CONTAINS 'heat_emission'
    MATCH (city)-[:INFLUENCES]->(thermal:Pattern {pattern_type: 'thermal'})
    MATCH (thermal)-[:CAUSES]->(wind:Pattern)
    WHERE wind.variables CONTAINS 'wind' OR wind.pattern_type = 'atmospheric'
    OPTIONAL MATCH (paper:Paper)-[:VALIDATES]->(wind)
    RETURN city.name as source_city,
           thermal.name as thermal_effect,
           wind.name as wind_pattern,
           collect(paper.title) as validating_papers
    """
