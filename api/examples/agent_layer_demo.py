"""
🏭🌊 Agent Layer Demo
======================
Demonstrates multi-layer causal systems with intermediate agents.

Two scenarios:
1. Manufacturing: Operator → Machine → Defect
2. Climate: City → Thermal Vortex → Wind Pattern

This shows how the "second layer" (agents) mediate between
systems and outcomes.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Simulated API client for demonstration
class AgentLayerDemo:
    """Demo of the Agent layer concepts."""
    
    def __init__(self):
        self.agents: Dict[str, Dict] = {}
        self.states: List[Dict] = []
        self.actions: List[Dict] = []
        self.events: List[Dict] = []
        self.papers: List[Dict] = []
        
    # ========== Manufacturing Scenario ==========
    
    async def setup_manufacturing_scenario(self):
        """
        Scenario: Brake hose factory
        - Machine: Extruder producing inner liner
        - Agent: Operator (human) who controls the machine
        - Outcome: Defect on external diameter
        """
        print("\n🏭 MANUFACTURING SCENARIO")
        print("=" * 50)
        
        # 1. Create the machine (system)
        machine = {
            "id": "extruder_01",
            "name": "Inner Liner Extruder",
            "type": "MACHINE",
            "parameters": ["speed_rpm", "temperature_c", "pellet_feed_rate"],
            "normal_ranges": {
                "speed_rpm": (80, 120),
                "temperature_c": (180, 220),
                "pellet_feed_rate": (50, 70)
            }
        }
        print(f"  📦 Machine: {machine['name']}")
        
        # 2. Create the operator (agent)
        operator = {
            "id": "operator_mario",
            "agent_type": "OPERATOR",
            "name": "Mario Rossi",
            "capabilities": [
                {
                    "name": "adjust_speed",
                    "type": "CONTROL",
                    "parameters": {"range": (60, 140), "unit": "rpm"},
                    "constraints": ["must_not_exceed_140"]
                },
                {
                    "name": "load_pellets",
                    "type": "CONTROL",
                    "parameters": {"batch_size_kg": 25},
                    "constraints": ["wait_until_hopper_below_20pct"]
                },
                {
                    "name": "emergency_stop",
                    "type": "CONTROL",
                    "parameters": {},
                    "constraints": []
                }
            ],
            "constraints": [
                {
                    "name": "shift_duration",
                    "type": "TEMPORAL",
                    "value": "8 hours max",
                    "description": "Italian labor law"
                },
                {
                    "name": "certification_required",
                    "type": "REGULATORY",
                    "value": "extruder_certified",
                    "description": "Must have machine certification"
                },
                {
                    "name": "fatigue_limit",
                    "type": "PHYSICAL",
                    "value": 0.8,
                    "description": "Performance degrades above 80% fatigue"
                }
            ],
            "operates_on": ["extruder_01"],
            "metadata": {
                "shift": "night",
                "experience_years": 5,
                "certification_date": "2020-03-15"
            }
        }
        self.agents["operator_mario"] = operator
        print(f"  👷 Operator: {operator['name']}")
        print(f"     Capabilities: {[c['name'] for c in operator['capabilities']]}")
        print(f"     Constraints: {[c['name'] for c in operator['constraints']]}")
        
        # 3. Record state changes (Friday night shift = tired)
        friday_night = datetime(2024, 1, 19, 22, 0)  # Friday 10 PM
        
        fatigue_progression = [
            (friday_night, 0.2),  # Start of shift
            (friday_night + timedelta(hours=2), 0.4),  # 2 hours in
            (friday_night + timedelta(hours=4), 0.6),  # Midnight
            (friday_night + timedelta(hours=6), 0.75),  # 4 AM
            (friday_night + timedelta(hours=7), 0.85),  # 5 AM - CRITICAL
        ]
        
        print(f"\n  📊 State Progression (Friday Night Shift):")
        for timestamp, fatigue in fatigue_progression:
            state = {
                "agent_id": "operator_mario",
                "state_name": "fatigue",
                "value": fatigue,
                "timestamp": timestamp.isoformat(),
                "context": {
                    "day_of_week": "Friday",
                    "shift": "night",
                    "hours_worked": (timestamp - friday_night).seconds / 3600
                }
            }
            self.states.append(state)
            indicator = "⚠️" if fatigue >= 0.7 else "  "
            print(f"     {indicator} {timestamp.strftime('%H:%M')}: fatigue={fatigue:.0%}")
        
        # 4. Record the problematic action
        action_time = friday_night + timedelta(hours=6, minutes=30)  # 4:30 AM
        action = {
            "agent_id": "operator_mario",
            "action_id": "action_001",
            "capability_used": "adjust_speed",
            "parameters": {
                "new_speed": 85,  # Below optimal
                "reason": "felt_too_fast"  # Tired perception
            },
            "timestamp": action_time.isoformat(),
            "resulted_in": "event_defect_001"
        }
        self.actions.append(action)
        print(f"\n  ⚡ Action at {action_time.strftime('%H:%M')}:")
        print(f"     Used: {action['capability_used']}")
        print(f"     Parameters: speed={action['parameters']['new_speed']} rpm")
        print(f"     (Optimal range: 100-110 rpm)")
        
        # 5. Record the outcome (defect)
        defect = {
            "id": "event_defect_001",
            "event_type": "QUALITY_DEFECT",
            "date": action_time.isoformat(),
            "description": "External diameter out of tolerance",
            "magnitude": 0.7,
            "location": "production_line_3",
            "metadata": {
                "defect_type": "dimensional",
                "measurement": {"od_mm": 9.2, "spec_min": 9.5, "spec_max": 10.5},
                "batch_id": "BH-2024-0119-N03"
            }
        }
        self.events.append(defect)
        print(f"\n  ❌ Outcome:")
        print(f"     Event: {defect['event_type']}")
        print(f"     OD: {defect['metadata']['measurement']['od_mm']} mm")
        print(f"     Spec: {defect['metadata']['measurement']['spec_min']}-{defect['metadata']['measurement']['spec_max']} mm")
        
        # 6. Build the causal narrative
        print(f"\n  🔗 Causal Chain:")
        print(f"     Pattern: End-of-week night shift")
        print(f"     ↓")
        print(f"     Agent: Mario (fatigue=85%)")
        print(f"     ↓")
        print(f"     Action: Reduced speed to 85 rpm")
        print(f"     ↓")
        print(f"     Outcome: Dimensional defect")
        
        return operator, action, defect
    
    # ========== Climate Scenario ==========
    
    async def setup_climate_scenario(self):
        """
        Scenario: Urban heat affecting regional weather
        - System: Urban area (city heating infrastructure)
        - Agent: City (as thermal emitter)
        - Outcome: Modified wind patterns
        """
        print("\n\n🌊 CLIMATE/OCEANOGRAPHY SCENARIO")
        print("=" * 50)
        
        # 1. Create the city as an agent
        city = {
            "id": "city_milan",
            "agent_type": "INFRASTRUCTURE",
            "name": "Milan Metropolitan Area",
            "capabilities": [
                {
                    "name": "heat_emission",
                    "type": "EMIT",
                    "parameters": {
                        "base_rate_mw": 2000,
                        "range_mw": (500, 8000),
                        "affected_radius_km": 50
                    },
                    "constraints": ["temperature_dependent", "seasonal_variation"]
                },
                {
                    "name": "albedo_modification",
                    "type": "MODULATE",
                    "parameters": {
                        "urban_albedo": 0.15,
                        "rural_albedo": 0.25
                    },
                    "constraints": ["surface_type_fixed"]
                }
            ],
            "constraints": [
                {
                    "name": "thermodynamics_first_law",
                    "type": "PHYSICAL_LAW",
                    "value": "energy_conservation",
                    "description": "Heat emission equals energy consumption"
                },
                {
                    "name": "thermal_inertia",
                    "type": "PHYSICAL_LAW",
                    "value": "lag_hours=6",
                    "description": "Urban structures store and release heat slowly"
                },
                {
                    "name": "heating_capacity",
                    "type": "CAPACITY",
                    "value": "max_8000_mw",
                    "description": "Total installed heating capacity"
                }
            ],
            "operates_on": ["po_valley_atmosphere"],
            "metadata": {
                "population": 3_200_000,
                "area_km2": 182,
                "heating_type": "primarily_natural_gas",
                "efficiency": 0.85
            }
        }
        self.agents["city_milan"] = city
        print(f"  🏙️ Agent: {city['name']}")
        print(f"     Type: {city['agent_type']}")
        print(f"     Capabilities: {[c['name'] for c in city['capabilities']]}")
        
        # 2. Record state changes during cold snap
        cold_snap_start = datetime(2024, 1, 15, 0, 0)
        
        efficiency_states = [
            (cold_snap_start, 0.85, -2),  # Normal
            (cold_snap_start + timedelta(days=1), 0.82, -8),  # Cold snap starts
            (cold_snap_start + timedelta(days=2), 0.78, -12),  # Peak cold
            (cold_snap_start + timedelta(days=3), 0.75, -15),  # Extreme demand
        ]
        
        print(f"\n  📊 State Progression (Cold Snap):")
        for timestamp, efficiency, temp_c in efficiency_states:
            state = {
                "agent_id": "city_milan",
                "state_name": "heating_efficiency",
                "value": efficiency,
                "timestamp": timestamp.isoformat(),
                "context": {
                    "ambient_temp_c": temp_c,
                    "demand_level": "extreme" if temp_c < -10 else "high",
                    "gas_price_eur_kwh": 0.12
                }
            }
            self.states.append(state)
            heat_emission = 2000 * (1 / efficiency) * abs(temp_c / 10)
            print(f"     {timestamp.strftime('%Y-%m-%d')}: efficiency={efficiency:.0%}, "
                  f"temp={temp_c}°C, emission≈{heat_emission:.0f} MW")
        
        # 3. Record the thermal emission action
        peak_emission_time = cold_snap_start + timedelta(days=2, hours=6)  # Peak morning
        
        action = {
            "agent_id": "city_milan",
            "action_id": "action_thermal_001",
            "capability_used": "heat_emission",
            "parameters": {
                "emission_mw": 6500,
                "duration_hours": 12,
                "spatial_distribution": "concentrated_center"
            },
            "timestamp": peak_emission_time.isoformat(),
            "resulted_in": "event_vortex_001"
        }
        self.actions.append(action)
        print(f"\n  ⚡ Action at {peak_emission_time.strftime('%Y-%m-%d %H:%M')}:")
        print(f"     Used: {action['capability_used']}")
        print(f"     Emission: {action['parameters']['emission_mw']} MW")
        print(f"     Duration: {action['parameters']['duration_hours']} hours")
        
        # 4. Create the intermediate physical process (thermal vortex)
        thermal_vortex = {
            "id": "process_vortex_001",
            "agent_type": "PHYSICAL_PROCESS",
            "name": "Urban Heat Island Vortex",
            "capabilities": [
                {
                    "name": "vertical_convection",
                    "type": "TRANSFORM",
                    "parameters": {"updraft_speed_ms": 2.5}
                },
                {
                    "name": "horizontal_inflow",
                    "type": "MODULATE",
                    "parameters": {"affected_radius_km": 30}
                }
            ],
            "constraints": [
                {
                    "name": "buoyancy_driven",
                    "type": "PHYSICAL_LAW",
                    "value": "delta_T_threshold > 3°C"
                }
            ],
            "operates_on": ["regional_wind_pattern"],
            "metadata": {
                "triggered_by": "city_milan",
                "height_m": 1500,
                "intensity": "moderate"
            }
        }
        self.agents["process_vortex_001"] = thermal_vortex
        print(f"\n  🌀 Intermediate Process Created:")
        print(f"     {thermal_vortex['name']}")
        print(f"     Height: {thermal_vortex['metadata']['height_m']} m")
        
        # 5. Record the final outcome (wind pattern change)
        wind_pattern_change = {
            "id": "event_vortex_001",
            "event_type": "WIND_PATTERN_ANOMALY",
            "date": (peak_emission_time + timedelta(hours=3)).isoformat(),
            "description": "Local wind convergence toward Milan",
            "magnitude": 0.6,
            "location": "Po Valley",
            "metadata": {
                "wind_speed_change_ms": 3.2,
                "direction_shift_deg": 45,
                "affected_area_km2": 5000,
                "cascade_effects": ["fog_formation", "pollution_concentration"]
            }
        }
        self.events.append(wind_pattern_change)
        print(f"\n  💨 Outcome:")
        print(f"     Event: {wind_pattern_change['event_type']}")
        print(f"     Wind shift: {wind_pattern_change['metadata']['direction_shift_deg']}°")
        print(f"     Cascade: {wind_pattern_change['metadata']['cascade_effects']}")
        
        # 6. Create a paper that documented this
        paper = {
            "id": "paper_uhi_2023",
            "title": "Urban Heat Island Effects on Po Valley Mesoscale Circulation",
            "authors": ["Bianchi A.", "Verdi G.", "Russo M."],
            "abstract": "Analysis of Milan's urban heat island and its influence on regional wind patterns...",
            "year": 2023,
            "journal": "Journal of Applied Meteorology",
            "doi": "10.1234/jam.2023.001"
        }
        self.papers.append(paper)
        
        # Paper-Event relationship
        paper_event_rel = {
            "paper_id": "paper_uhi_2023",
            "event_id": "event_vortex_001",
            "relation_type": "DOCUMENTS",  # Paper studied this type of event
            "confidence": 0.9,
            "direction": "paper_to_event"
        }
        
        print(f"\n  📄 Paper-Event Relationship:")
        print(f"     Paper: '{paper['title'][:40]}...'")
        print(f"     Relation: {paper_event_rel['relation_type']} → Event")
        
        # 7. Build the full causal chain
        print(f"\n  🔗 Causal Chain:")
        print(f"     Pattern: Cold snap (ambient -15°C)")
        print(f"     ↓")
        print(f"     Agent: Milan (efficiency=75%, emission=6500 MW)")
        print(f"     ↓")
        print(f"     Process: Urban Heat Island Vortex")
        print(f"     ↓")
        print(f"     Outcome: Wind pattern anomaly")
        print(f"     ↓")
        print(f"     Cascade: Fog + pollution concentration")
        
        return city, thermal_vortex, wind_pattern_change
    
    # ========== Panama Papers-style Query ==========
    
    async def demonstrate_network_query(self):
        """
        Show how to query the network like Panama Papers:
        "Find all entities connected to this agent"
        """
        print("\n\n🔍 PANAMA PAPERS-STYLE NETWORK QUERY")
        print("=" * 50)
        
        print("\n  Query: 'Find all entities connected to operator_mario'")
        print("\n  Result:")
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │ AGENT: Mario Rossi (OPERATOR)                  │")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │ OPERATES_ON:                                   │")
        print("  │   └── extruder_01 (Inner Liner Extruder)       │")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │ TOOK_ACTIONS:                                  │")
        print("  │   └── action_001: adjust_speed (85 rpm)        │")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │ CAUSED_OUTCOMES:                               │")
        print("  │   └── event_defect_001 (QUALITY_DEFECT)        │")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │ STATE_HISTORY:                                 │")
        print("  │   └── fatigue: 0.2 → 0.4 → 0.6 → 0.75 → 0.85   │")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │ INFLUENCED_BY_PATTERNS:                        │")
        print("  │   └── End-of-week night shift correlation      │")
        print("  └─────────────────────────────────────────────────┘")
        
        print("\n  Query: 'Find all entities connected to city_milan'")
        print("\n  Result:")
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │ AGENT: Milan Metropolitan Area (INFRASTRUCTURE)│")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │ OPERATES_ON:                                   │")
        print("  │   └── po_valley_atmosphere                     │")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │ CREATED_PROCESSES:                             │")
        print("  │   └── process_vortex_001 (PHYSICAL_PROCESS)    │")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │ CAUSED_OUTCOMES:                               │")
        print("  │   └── event_vortex_001 (WIND_PATTERN_ANOMALY)  │")
        print("  ├─────────────────────────────────────────────────┤")
        print("  │ DOCUMENTED_BY:                                 │")
        print("  │   └── paper_uhi_2023 (J. Applied Meteorology)  │")
        print("  └─────────────────────────────────────────────────┘")
    
    async def run_demo(self):
        """Run the complete demonstration."""
        print("╔══════════════════════════════════════════════════╗")
        print("║       MULTI-LAYER CAUSAL DISCOVERY DEMO          ║")
        print("║    Agent Layer: Operators, Infrastructure,       ║")
        print("║    Physical Processes as Causal Mediators        ║")
        print("╚══════════════════════════════════════════════════╝")
        
        await self.setup_manufacturing_scenario()
        await self.setup_climate_scenario()
        await self.demonstrate_network_query()
        
        print("\n\n📊 SUMMARY")
        print("=" * 50)
        print(f"  Agents created: {len(self.agents)}")
        print(f"  State records: {len(self.states)}")
        print(f"  Actions recorded: {len(self.actions)}")
        print(f"  Events tracked: {len(self.events)}")
        print(f"  Papers linked: {len(self.papers)}")
        
        print("\n  Key Insight:")
        print("  ─────────────")
        print("  The 'Agent Layer' captures the intermediate actors")
        print("  that mediate between patterns and outcomes:")
        print("")
        print("  Pattern → [AGENT (with state & capabilities)] → Outcome")
        print("")
        print("  This enables queries like:")
        print("  - 'Find defects caused by tired operators'")
        print("  - 'Find weather anomalies caused by urban heating'")
        print("  - 'What agents are affected by NAO patterns?'")


if __name__ == "__main__":
    demo = AgentLayerDemo()
    asyncio.run(demo.run_demo())
