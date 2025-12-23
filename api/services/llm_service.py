"""
🤖 Ollama LLM Service
======================
Local LLM integration for data understanding, causal explanation, and physics validation.
Uses qwen3-coder:30b for scientific reasoning.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
import ollama
from ollama import AsyncClient


@dataclass
class LLMConfig:
    """Configuration for Ollama LLM."""
    model: str = "qwen3-coder:30b"
    temperature: float = 0.3  # Lower for more factual responses
    context_length: int = 8192
    timeout: float = 120.0
    host: str = "http://localhost:11434"


@dataclass
class DataColumn:
    """Metadata about a data column."""
    name: str
    dtype: str
    sample_values: List[Any]
    unique_count: int
    null_count: int
    interpretation: Optional[str] = None
    is_temporal: bool = False
    unit: Optional[str] = None


@dataclass
class DataInterpretation:
    """LLM interpretation of a dataset."""
    columns: List[DataColumn]
    temporal_column: Optional[str] = None
    suggested_targets: List[str] = field(default_factory=list)
    domain: Optional[str] = None  # "flood", "manufacturing", "energy", etc.
    summary: str = ""


class OllamaLLMService:
    """
    LLM service using Ollama for local inference.
    
    Features:
    - Data interpretation: Understand column meanings
    - Temporal detection: Find time dimension
    - Causal explanation: Explain discovered relationships
    - Physics validation: Check if patterns make physical sense
    - Hypothesis generation: Suggest new correlations to explore
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.client = AsyncClient(host=self.config.host)
        self._available = None
        
    async def check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = await self.client.list()
            # Handle both dict and object responses
            if hasattr(response, 'models'):
                models_list = response.models
            elif isinstance(response, dict):
                models_list = response.get('models', [])
            else:
                models_list = []
            
            model_names = []
            for m in models_list:
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif isinstance(m, dict):
                    model_names.append(m.get('name', m.get('model', '')))
            
            self._available = any(self.config.model in name for name in model_names)
            if self._available:
                print(f"✅ Found model: {self.config.model}")
            else:
                print(f"⚠️ Model {self.config.model} not in: {model_names}")
            return self._available
        except Exception as e:
            print(f"Ollama not available: {e}")
            self._available = False
            return False
    
    async def _generate(self, prompt: str, system: str = None) -> str:
        """Generate response from LLM."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.context_length,
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"LLM Error: {str(e)}"
    
    async def _generate_stream(self, prompt: str, system: str = None) -> AsyncGenerator[str, None]:
        """Stream response from LLM."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            async for chunk in await self.client.chat(
                model=self.config.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.context_length,
                }
            ):
                if chunk.get('message', {}).get('content'):
                    yield chunk['message']['content']
        except Exception as e:
            yield f"LLM Error: {str(e)}"

    # ============== DATA INTERPRETATION ==============
    
    async def interpret_dataset(
        self,
        columns_info: List[Dict[str, Any]],
        filename: str = "",
        sample_data: str = ""
    ) -> DataInterpretation:
        """
        Analyze dataset structure and interpret column meanings.
        
        Args:
            columns_info: List of dicts with name, dtype, samples, etc.
            filename: Original filename for context
            sample_data: First few rows as string
            
        Returns:
            DataInterpretation with column meanings, temporal column, domain
        """
        system = """You are a scientific data analyst specializing in:
- Oceanography and satellite altimetry (sea level, DOT, SLA, SSH)
- Flood forecasting and storm surge prediction
- Climate data and meteorological variables
- Manufacturing process data
- Energy systems

Analyze datasets to identify:
1. What each column represents (physical meaning, units)
2. Which column is the TIME dimension (crucial for causal analysis)
3. What domain this data belongs to
4. Which variables could be targets for prediction

Respond in JSON format only."""

        prompt = f"""Analyze this dataset:

Filename: {filename}

Columns:
{json.dumps(columns_info, indent=2)}

Sample data:
{sample_data}

Respond with JSON:
{{
    "columns": [
        {{"name": "col_name", "interpretation": "what it means", "is_temporal": true/false, "unit": "unit or null"}}
    ],
    "temporal_column": "name of time column or null",
    "suggested_targets": ["potential target variables for prediction"],
    "domain": "flood|manufacturing|energy|oceanography|climate|other",
    "summary": "Brief description of this dataset"
}}"""

        response = await self._generate(prompt, system)
        
        # Parse JSON from response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            data = json.loads(json_str)
            
            columns = []
            for col in data.get("columns", []):
                # Find original column info
                orig = next((c for c in columns_info if c["name"] == col["name"]), {})
                columns.append(DataColumn(
                    name=col["name"],
                    dtype=orig.get("dtype", "unknown"),
                    sample_values=orig.get("samples", []),
                    unique_count=orig.get("unique_count", 0),
                    null_count=orig.get("null_count", 0),
                    interpretation=col.get("interpretation"),
                    is_temporal=col.get("is_temporal", False),
                    unit=col.get("unit"),
                ))
            
            return DataInterpretation(
                columns=columns,
                temporal_column=data.get("temporal_column"),
                suggested_targets=data.get("suggested_targets", []),
                domain=data.get("domain"),
                summary=data.get("summary", ""),
            )
        except json.JSONDecodeError:
            # Return basic interpretation if JSON parsing fails
            return DataInterpretation(
                columns=[DataColumn(
                    name=c["name"],
                    dtype=c.get("dtype", "unknown"),
                    sample_values=c.get("samples", []),
                    unique_count=c.get("unique_count", 0),
                    null_count=c.get("null_count", 0),
                ) for c in columns_info],
                summary=response[:500],
            )

    # ============== CAUSAL EXPLANATION ==============
    
    async def explain_causal_relationship(
        self,
        source: str,
        target: str,
        lag: int,
        strength: float,
        p_value: float,
        domain: str = "flood",
        context: Dict[str, Any] = None,
    ) -> str:
        """
        Generate natural language explanation of a causal relationship.
        
        Args:
            source: Cause variable name
            target: Effect variable name
            lag: Time lag in steps (e.g., days, hours)
            strength: Correlation/effect strength (-1 to 1)
            p_value: Statistical significance
            domain: Domain for context (flood, manufacturing, etc.)
            context: Additional context (units, location, etc.)
            
        Returns:
            Natural language explanation
        """
        system = """You are a scientific advisor explaining causal relationships discovered in data.

For FLOOD/OCEANOGRAPHY domain, consider:
- Storm surge mechanisms (wind setup, inverse barometer effect)
- Typical lag times: pressure systems 24-72h, wind effects 6-24h
- Atlantic storm tracks, NAO influence
- Tidal interactions

For MANUFACTURING domain, consider:
- Process delays (heating, cooling, curing times)
- Material flow and batch processing
- Quality control feedback loops

Provide clear, scientifically grounded explanations that a domain expert would find useful."""

        prompt = f"""Explain this discovered causal relationship:

**Cause**: {source}
**Effect**: {target}
**Time Lag**: {lag} time steps
**Strength**: {strength:.3f} (range -1 to 1)
**P-value**: {p_value:.4f}
**Domain**: {domain}
**Additional Context**: {json.dumps(context or {})}

Provide:
1. Plain language explanation of what this relationship means
2. Physical/scientific mechanism that could explain it
3. Whether the lag makes sense given known dynamics
4. Confidence assessment (is this likely real or spurious?)
5. What additional evidence would strengthen this finding

Be concise but informative."""

        return await self._generate(prompt, system)

    # ============== PHYSICS VALIDATION ==============
    
    async def validate_pattern_physics(
        self,
        pattern_description: str,
        variables: List[str],
        domain: str,
        statistical_confidence: float,
    ) -> Dict[str, Any]:
        """
        Validate if a discovered pattern makes physical sense.
        
        Returns dict with:
        - is_valid: bool
        - physics_score: 0-1
        - explanation: str
        - concerns: List[str]
        - supporting_evidence: List[str]
        """
        system = """You are a physics validator for data-driven pattern discovery.

Your job is to assess whether statistically discovered patterns are physically plausible.

Key physical constraints by domain:

FLOOD/SURGE:
- Wind setup: η ∝ U²·L/(g·h) - setup proportional to wind squared
- Inverse barometer: ~1 cm per hPa pressure drop
- Storm surge lag: 12-48 hours from atmospheric forcing
- Tidal modulation: 12.42h (M2) and 23.93h (K1) periods

MANUFACTURING:
- Arrhenius: reaction rate doubles per 10°C
- Viscosity decreases with temperature
- Mixing time scales with vessel size

Respond in JSON format."""

        prompt = f"""Validate this pattern:

Pattern: {pattern_description}
Variables involved: {variables}
Domain: {domain}
Statistical confidence: {statistical_confidence:.2%}

Respond with JSON:
{{
    "is_valid": true/false,
    "physics_score": 0.0-1.0,
    "explanation": "why this is/isn't physically plausible",
    "concerns": ["list of physical concerns if any"],
    "supporting_evidence": ["physical laws/mechanisms that support this"],
    "suggested_checks": ["additional analyses to confirm"]
}}"""

        response = await self._generate(prompt, system)
        
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "is_valid": None,
                "physics_score": 0.5,
                "explanation": response[:500],
                "concerns": [],
                "supporting_evidence": [],
            }

    # ============== HYPOTHESIS GENERATION ==============
    
    async def generate_hypotheses(
        self,
        variables: List[str],
        domain: str,
        known_relationships: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate hypotheses about potential causal relationships to explore.
        
        Returns list of hypotheses with:
        - source, target, expected_lag, rationale, priority
        """
        system = """You are a scientific hypothesis generator for causal discovery.

Based on domain knowledge, suggest potential causal relationships that should be investigated.

Consider:
- Known physical mechanisms
- Literature-documented teleconnections
- Process dynamics and typical time scales
- Counterintuitive relationships worth exploring

Prioritize hypotheses by scientific interest and testability."""

        known_str = ""
        if known_relationships:
            known_str = f"\n\nAlready discovered relationships:\n{json.dumps(known_relationships, indent=2)}"

        prompt = f"""Generate hypotheses for causal relationships:

Variables available: {variables}
Domain: {domain}
{known_str}

Suggest 5-10 potential causal relationships to investigate.

Respond with JSON:
{{
    "hypotheses": [
        {{
            "source": "variable name",
            "target": "variable name",
            "expected_lag": "e.g., 24-48 hours",
            "expected_direction": "positive/negative",
            "rationale": "why this relationship might exist",
            "priority": "high/medium/low",
            "literature_support": "relevant papers/mechanisms"
        }}
    ]
}}"""

        response = await self._generate(prompt, system)
        
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            data = json.loads(json_str)
            return data.get("hypotheses", [])
        except json.JSONDecodeError:
            return []

    # ============== DISCOVERY SUMMARY ==============
    
    async def summarize_discoveries(
        self,
        relationships: List[Dict[str, Any]],
        domain: str,
        dataset_summary: str = "",
    ) -> str:
        """
        Generate executive summary of causal discovery results.
        """
        system = """You are a scientific report writer summarizing causal discovery findings.

Write clear, actionable summaries that:
1. Highlight the most important discoveries
2. Explain practical implications
3. Note limitations and caveats
4. Suggest next steps for investigation

Use professional scientific language accessible to domain experts."""

        prompt = f"""Summarize these causal discovery findings:

Domain: {domain}
Dataset: {dataset_summary}

Discovered Relationships:
{json.dumps(relationships, indent=2)}

Write an executive summary covering:
1. Key findings (most important relationships)
2. Novel discoveries (unexpected patterns)
3. Validation status (physics-consistent vs. needs verification)
4. Practical implications
5. Recommended next steps"""

        return await self._generate(prompt, system)


# Singleton instance
_llm_service: Optional[OllamaLLMService] = None

def get_llm_service() -> OllamaLLMService:
    """Get or create LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = OllamaLLMService()
    return _llm_service


# Quick test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        service = OllamaLLMService()
        
        print("Checking Ollama availability...")
        available = await service.check_availability()
        print(f"Ollama available: {available}")
        
        if available:
            print("\nTesting data interpretation...")
            result = await service.interpret_dataset(
                columns_info=[
                    {"name": "timestamp", "dtype": "datetime64", "samples": ["2024-01-01", "2024-01-02"]},
                    {"name": "sea_level_anomaly", "dtype": "float64", "samples": [0.15, -0.23, 0.08]},
                    {"name": "wind_speed", "dtype": "float64", "samples": [12.5, 8.3, 15.2]},
                    {"name": "pressure", "dtype": "float64", "samples": [1013.2, 1008.5, 1015.8]},
                ],
                filename="storm_surge_data.csv"
            )
            print(f"Domain: {result.domain}")
            print(f"Temporal column: {result.temporal_column}")
            print(f"Summary: {result.summary}")
    
    asyncio.run(test())
