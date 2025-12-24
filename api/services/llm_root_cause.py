"""
🔬 Extended LLM Service for Root Cause Analysis
=================================================

Extension methods for OllamaLLMService to support:
- Ishikawa diagram generation (6M adapted for oceanography)
- FMEA analysis for satellite data quality
- 5-Why causal chain drilling
- Physics-validated scoring

Uses the root_cause module's data structures and templates.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import root cause module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.root_cause import (
    IshikawaCategory, IshikawaCause, IshikawaDiagram,
    FMEAItem, FMEAAnalysis,
    WhyStep, FiveWhyAnalysis,
    FloodPhysicsScore,
    create_flood_ishikawa_template,
    create_satellite_fmea_template,
)


@dataclass
class RootCauseConfig:
    """Configuration for root cause analysis."""
    max_why_depth: int = 5
    min_rpn_threshold: int = 100  # High risk
    physics_weight: float = 0.4
    chain_weight: float = 0.3
    experience_weight: float = 0.3


class RootCauseLLMExtension:
    """
    Extension methods for root cause analysis using LLM.
    
    Designed to be mixed into OllamaLLMService or used standalone.
    """
    
    def __init__(self, base_llm_service=None, config: RootCauseConfig = None):
        self.llm = base_llm_service  # OllamaLLMService instance
        self.config = config or RootCauseConfig()
    
    async def _generate(self, prompt: str, system: str = None) -> str:
        """Generate response using base LLM service."""
        if self.llm:
            return await self.llm._generate(prompt, system)
        else:
            return "LLM service not configured"
    
    # =========================================================================
    # ISHIKAWA DIAGRAM GENERATION
    # =========================================================================
    
    async def generate_ishikawa_diagram(
        self,
        event_description: str,
        event_location: str,
        event_time: str,
        observed_data: Dict[str, Any] = None,
        domain: str = "flood",
    ) -> IshikawaDiagram:
        """
        Generate Ishikawa (fishbone) diagram for an event.
        
        LLM analyzes the event and populates causes across 6 categories:
        - ATMOSPHERE: Wind, pressure, precipitation
        - OCEAN: Tides, currents, stratification  
        - CRYOSPHERE: Ice, freshwater flux
        - MEASUREMENT: Sensor issues, calibration
        - MODEL: Forecast errors, resolution
        - EXTERNAL: Rivers, anthropogenic, seismic
        
        Args:
            event_description: What happened
            event_location: Where (e.g., "Fram Strait, 78°N 0°E")
            event_time: When (e.g., "2024-01-15 12:00 UTC")
            observed_data: Measurements at time of event
            domain: Analysis domain
            
        Returns:
            IshikawaDiagram with populated causes
        """
        system = """You are an expert in root cause analysis for oceanographic/flood events.

Generate an Ishikawa (fishbone) diagram identifying potential causes across categories:
1. ATMOSPHERE: Wind, pressure, storms, teleconnections
2. OCEAN: Tides, currents, density, waves
3. CRYOSPHERE: Sea ice, freshwater, glacial
4. MEASUREMENT: Sensor, calibration, processing
5. MODEL: Resolution, forcing, boundaries
6. EXTERNAL: Rivers, anthropogenic, seismic

For each cause, provide:
- Concise description
- Evidence level (confirmed/likely/possible/speculative)
- Contributing factors (sub-causes)

Be specific to the event and location. Use physical reasoning.
Respond in JSON format."""

        prompt = f"""Generate Ishikawa diagram for this event:

**Event**: {event_description}
**Location**: {event_location}
**Time**: {event_time}
**Domain**: {domain}

Observed data:
{json.dumps(observed_data or {}, indent=2)}

Generate causes for each category. Respond with JSON:
{{
    "effect": "description of main effect",
    "categories": {{
        "ATMOSPHERE": [
            {{"description": "cause", "evidence": "confirmed/likely/possible", "factors": ["sub-causes"]}}
        ],
        "OCEAN": [...],
        "CRYOSPHERE": [...],
        "MEASUREMENT": [...],
        "MODEL": [...],
        "EXTERNAL": [...]
    }},
    "primary_cause": "most likely primary cause",
    "confidence": 0.0-1.0
}}"""

        response = await self._generate(prompt, system)
        
        # Parse response
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            diagram = IshikawaDiagram(effect=data.get("effect", event_description))
            
            for cat_name, causes in data.get("categories", {}).items():
                try:
                    category = IshikawaCategory[cat_name.upper()]
                except KeyError:
                    continue
                
                for cause_data in causes:
                    cause = IshikawaCause(
                        category=category,
                        description=cause_data.get("description", ""),
                        evidence_level=cause_data.get("evidence", "possible"),
                        contributing_factors=cause_data.get("factors", []),
                    )
                    diagram.add_cause(cause)
            
            return diagram
            
        except json.JSONDecodeError:
            # Return template with LLM response as note
            template = create_flood_ishikawa_template()
            template.effect = event_description
            return template
    
    # =========================================================================
    # FMEA ANALYSIS
    # =========================================================================
    
    async def generate_fmea_analysis(
        self,
        component: str,
        function: str,
        data_source: str,
        known_issues: List[str] = None,
    ) -> FMEAAnalysis:
        """
        Generate FMEA analysis for a data component/satellite.
        
        Identifies:
        - Potential failure modes
        - Effects of failure
        - Root causes
        - Severity, Occurrence, Detection ratings
        - Recommended actions
        
        Args:
            component: What is being analyzed (e.g., "Sentinel-3A SSH")
            function: What it's supposed to do
            data_source: Where data comes from
            known_issues: Previously identified issues
            
        Returns:
            FMEAAnalysis with failure mode items
        """
        system = """You are a reliability engineer analyzing satellite oceanographic data systems.

Generate FMEA (Failure Mode and Effects Analysis) identifying:
- Potential failure modes for the component
- Effects on downstream analysis
- Root causes
- Severity (1-10): Impact if failure occurs
- Occurrence (1-10): How likely is this failure
- Detection (1-10): How hard is it to detect BEFORE it causes problems

For each failure mode, calculate RPN = Severity × Occurrence × Detection
High RPN (>100) = high priority

Suggest specific recommended actions for high-risk items.
Respond in JSON format."""

        known_str = ""
        if known_issues:
            known_str = f"\nKnown issues to consider:\n- " + "\n- ".join(known_issues)

        prompt = f"""Generate FMEA for:

**Component**: {component}
**Function**: {function}
**Data Source**: {data_source}
{known_str}

Identify at least 5 potential failure modes. Respond with JSON:
{{
    "component": "{component}",
    "items": [
        {{
            "failure_mode": "what can go wrong",
            "effect": "impact on analysis",
            "cause": "root cause",
            "severity": 1-10,
            "occurrence": 1-10,
            "detection": 1-10,
            "recommended_action": "what to do",
            "responsible": "who should act"
        }}
    ],
    "highest_risk_items": ["top 3 failure modes by RPN"]
}}"""

        response = await self._generate(prompt, system)
        
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            analysis = FMEAAnalysis(component=data.get("component", component))
            
            for item_data in data.get("items", []):
                item = FMEAItem(
                    failure_mode=item_data.get("failure_mode", ""),
                    effect=item_data.get("effect", ""),
                    cause=item_data.get("cause", ""),
                    severity=item_data.get("severity", 5),
                    occurrence=item_data.get("occurrence", 5),
                    detection=item_data.get("detection", 5),
                    recommended_action=item_data.get("recommended_action", ""),
                    responsible=item_data.get("responsible", ""),
                )
                analysis.add_item(item)
            
            return analysis
            
        except json.JSONDecodeError:
            return create_satellite_fmea_template()
    
    # =========================================================================
    # 5-WHY ANALYSIS
    # =========================================================================
    
    async def run_five_why_analysis(
        self,
        symptom: str,
        context: Dict[str, Any] = None,
        max_depth: int = None,
    ) -> FiveWhyAnalysis:
        """
        Run LLM-driven 5-Why analysis to find root cause.
        
        Starting from a symptom, iteratively asks "Why?" to drill down
        to the fundamental root cause. Uses physics and domain knowledge
        to guide the analysis.
        
        Args:
            symptom: Initial problem/observation
            context: Additional context (measurements, location, etc.)
            max_depth: Maximum depth (default 5)
            
        Returns:
            FiveWhyAnalysis with chain of why-because pairs
        """
        max_depth = max_depth or self.config.max_why_depth
        
        system = """You are an expert in root cause analysis using the 5-Why technique (Kaizen/Toyota method).

Given a symptom, iteratively ask "Why?" and provide physics-grounded answers until you reach a fundamental root cause.

Guidelines:
- Each "because" should be specific and verifiable
- Use physical laws and oceanographic principles
- Stop when you reach a root cause that:
  - Is outside the system's control, OR
  - Requires management/design change, OR
  - Is a fundamental physical constraint
- Identify whether each level is measurable

Typical flood/surge root causes:
- Atmospheric forcing (can't control)
- Topographic constraints (can't change)
- Data availability (can improve)
- Model limitations (can improve)

Respond in JSON format with the full chain."""

        context_str = json.dumps(context or {}, indent=2) if context else "None provided"

        prompt = f"""Perform 5-Why analysis:

**Initial Symptom**: {symptom}
**Context**: {context_str}
**Max Depth**: {max_depth}

Drill down to root cause. Respond with JSON:
{{
    "symptom": "{symptom}",
    "chain": [
        {{"level": 1, "why": "Why did this happen?", "because": "explanation", "is_measurable": true/false, "evidence": "supporting data"}},
        {{"level": 2, "why": "Why [previous because]?", "because": "deeper explanation", ...}},
        ...
    ],
    "root_cause": "fundamental root cause identified",
    "root_cause_category": "atmospheric|oceanic|instrumental|data|model|design",
    "controllability": "controllable|partially_controllable|uncontrollable",
    "recommended_actions": ["what can be done at each level"]
}}"""

        response = await self._generate(prompt, system)
        
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            analysis = FiveWhyAnalysis(symptom=data.get("symptom", symptom))
            
            for step_data in data.get("chain", []):
                step = WhyStep(
                    level=step_data.get("level", 0),
                    why=step_data.get("why", ""),
                    because=step_data.get("because", ""),
                    is_measurable=step_data.get("is_measurable", False),
                    evidence=step_data.get("evidence"),
                )
                analysis.add_step(step)
            
            analysis.root_cause = data.get("root_cause")
            
            return analysis
            
        except json.JSONDecodeError:
            return FiveWhyAnalysis(symptom=symptom)
    
    # =========================================================================
    # HYBRID SCORING (Physics + Chain + Experience)
    # =========================================================================
    
    async def calculate_hybrid_score(
        self,
        event: Dict[str, Any],
        causal_chain: List[Dict[str, Any]],
        physics_data: Dict[str, float] = None,
        historical_matches: int = 0,
    ) -> Dict[str, float]:
        """
        Calculate hybrid knowledge score combining:
        1. Physics-based validation (using equations)
        2. Chain-based scoring (causal path strength)
        3. Experience-based scoring (historical pattern matching)
        
        Like a hybrid car: physics engine + experience engine working together.
        
        Args:
            event: Event details (ssh_anomaly, wind_speed, pressure, etc.)
            causal_chain: Discovered causal relationships
            physics_data: Pre-calculated physics values
            historical_matches: Number of similar historical events
            
        Returns:
            Dict with individual scores and combined hybrid score
        """
        # Physics score
        if physics_data:
            physics_scorer = FloodPhysicsScore()
            for key, value in physics_data.items():
                if key == "wind_speed":
                    physics_scorer.wind_speed_ms = value
                elif key == "pressure":
                    physics_scorer.pressure_hpa = value
                elif key == "reference_pressure":
                    physics_scorer.ref_pressure_hpa = value
                elif key == "fetch":
                    physics_scorer.fetch_km = value
                elif key == "depth":
                    physics_scorer.depth_m = value
                elif key == "observed_surge":
                    physics_scorer.observed_surge_m = value
            
            physics_score = physics_scorer.validation_score()
            physics_detail = {
                "wind_setup_m": physics_scorer.wind_setup(),
                "ib_effect_m": physics_scorer.inverse_barometer(),
                "total_expected_m": physics_scorer.total_surge(),
            }
        else:
            physics_score = 0.5  # Neutral if no data
            physics_detail = {}
        
        # Chain score (strength of causal path)
        if causal_chain:
            total_strength = sum(abs(link.get("strength", 0)) for link in causal_chain)
            avg_strength = total_strength / len(causal_chain)
            min_pvalue = min(link.get("p_value", 1.0) for link in causal_chain)
            
            # Chain score: high strength + low p-value = good
            chain_score = avg_strength * (1 - min_pvalue)
        else:
            chain_score = 0.0
        
        # Experience score (historical pattern matching)
        if historical_matches > 0:
            # Logarithmic scaling: 1 match = 0.3, 10 matches = 0.7, 100 matches = 1.0
            import math
            experience_score = min(1.0, 0.3 + 0.35 * math.log10(1 + historical_matches))
        else:
            experience_score = 0.0
        
        # Hybrid combination
        w_physics = self.config.physics_weight
        w_chain = self.config.chain_weight
        w_experience = self.config.experience_weight
        
        hybrid_score = (
            w_physics * physics_score +
            w_chain * chain_score +
            w_experience * experience_score
        )
        
        return {
            "hybrid_score": hybrid_score,
            "physics_score": physics_score,
            "chain_score": chain_score,
            "experience_score": experience_score,
            "physics_detail": physics_detail,
            "weights": {
                "physics": w_physics,
                "chain": w_chain,
                "experience": w_experience,
            },
            "interpretation": self._interpret_score(hybrid_score),
        }
    
    def _interpret_score(self, score: float) -> str:
        """Interpret hybrid score."""
        if score >= 0.8:
            return "High confidence - strongly supported by physics, chain, and experience"
        elif score >= 0.6:
            return "Moderate confidence - good support, some uncertainty"
        elif score >= 0.4:
            return "Low confidence - partial support, needs verification"
        else:
            return "Very low confidence - limited evidence, speculative"
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response."""
        if "```json" in response:
            return response.split("```json")[1].split("```")[0]
        elif "```" in response:
            parts = response.split("```")
            for part in parts[1::2]:  # Odd indices are code blocks
                if part.strip().startswith("{"):
                    return part
        return response


# =============================================================================
# INTEGRATION WITH BASE LLM SERVICE
# =============================================================================

def extend_llm_service(llm_service):
    """
    Extend an OllamaLLMService instance with root cause methods.
    
    Usage:
        from api.services.llm_service import get_llm_service
        from api.services.llm_root_cause import extend_llm_service
        
        llm = get_llm_service()
        llm = extend_llm_service(llm)
        
        # Now has additional methods:
        diagram = await llm.generate_ishikawa_diagram(...)
        fmea = await llm.generate_fmea_analysis(...)
        why5 = await llm.run_five_why_analysis(...)
        score = await llm.calculate_hybrid_score(...)
    """
    extension = RootCauseLLMExtension(llm_service)
    
    # Add methods to service
    llm_service.generate_ishikawa_diagram = extension.generate_ishikawa_diagram
    llm_service.generate_fmea_analysis = extension.generate_fmea_analysis
    llm_service.run_five_why_analysis = extension.run_five_why_analysis
    llm_service.calculate_hybrid_score = extension.calculate_hybrid_score
    
    return llm_service


# Quick test
if __name__ == "__main__":
    async def test():
        from api.services.llm_service import OllamaLLMService
        
        llm = OllamaLLMService()
        available = await llm.check_availability()
        
        if not available:
            print("⚠️ Ollama not available, testing without LLM")
            
        # Test with extension
        ext = RootCauseLLMExtension(llm if available else None)
        
        # Test hybrid scoring without LLM
        score = await ext.calculate_hybrid_score(
            event={"ssh_anomaly": 0.5, "location": "Fram Strait"},
            causal_chain=[
                {"source": "wind_speed", "target": "ssh", "strength": 0.7, "p_value": 0.01},
                {"source": "pressure", "target": "ssh", "strength": 0.5, "p_value": 0.02},
            ],
            physics_data={
                "wind_speed": 15.0,
                "pressure": 980.0,
                "reference_pressure": 1013.25,
                "fetch": 200.0,
                "depth": 50.0,
                "observed_surge": 0.4,
            },
            historical_matches=5,
        )
        
        print("Hybrid Score Results:")
        print(f"  Physics Score: {score['physics_score']:.3f}")
        print(f"  Chain Score: {score['chain_score']:.3f}")
        print(f"  Experience Score: {score['experience_score']:.3f}")
        print(f"  HYBRID SCORE: {score['hybrid_score']:.3f}")
        print(f"  Interpretation: {score['interpretation']}")
        
        if score.get("physics_detail"):
            print(f"\nPhysics Detail:")
            print(f"  Wind Setup: {score['physics_detail'].get('wind_setup_m', 0):.3f} m")
            print(f"  IB Effect: {score['physics_detail'].get('ib_effect_m', 0):.3f} m")
            print(f"  Total Expected: {score['physics_detail'].get('total_expected_m', 0):.3f} m")
    
    asyncio.run(test())
