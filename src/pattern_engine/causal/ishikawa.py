"""
🦴 Ishikawa (Fishbone) Diagram Generator
========================================

Generates cause-and-effect (Ishikawa/Fishbone) diagrams from causal analysis results.
Exports to multiple formats: JSON, SVG, Mermaid, and PDF.

The Ishikawa diagram organizes causes into categories:
- For Manufacturing: Man, Machine, Material, Method, Measurement, Environment
- For Climate/Ocean: Atmospheric, Oceanic, Cryospheric, Anthropogenic, Teleconnection, Local

Usage:
    from src.pattern_engine.causal.ishikawa import IshikawaDiagram
    
    diagram = IshikawaDiagram(
        effect="Lake Maggiore Flood Oct 2000",
        causes=pcmci_result.significant_links
    )
    
    svg = diagram.to_svg()
    mermaid = diagram.to_mermaid()
    diagram.save("flood_causes.svg")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json
from datetime import datetime
import math


class CauseCategory(Enum):
    """Categories for cause classification."""
    # Climate/Ocean categories
    ATMOSPHERIC = "Atmospheric"
    OCEANIC = "Oceanic"
    CRYOSPHERIC = "Cryospheric"
    TELECONNECTION = "Teleconnection"
    ANTHROPOGENIC = "Anthropogenic"
    LOCAL = "Local"
    
    # Manufacturing categories (for completeness)
    MAN = "Man"
    MACHINE = "Machine"
    MATERIAL = "Material"
    METHOD = "Method"
    MEASUREMENT = "Measurement"
    ENVIRONMENT = "Environment"


# Keyword-based category classification
CLIMATE_CATEGORY_KEYWORDS = {
    CauseCategory.ATMOSPHERIC: [
        "wind", "pressure", "precipitation", "humidity", "cloud",
        "temperature", "air", "storm", "cyclone", "anticyclone",
        "jet", "front", "trough", "ridge", "blocking",
    ],
    CauseCategory.OCEANIC: [
        "sst", "sla", "current", "salinity", "wave", "tide",
        "upwelling", "downwelling", "eddy", "gyre", "transport",
        "heat_content", "stratification", "mixing", "ssh",
    ],
    CauseCategory.CRYOSPHERIC: [
        "ice", "snow", "glacier", "permafrost", "arctic", "antarctic",
        "sea_ice", "ice_extent", "freshwater",
    ],
    CauseCategory.TELECONNECTION: [
        "nao", "enso", "amo", "pdo", "ao", "mjo", "qbo",
        "el_nino", "la_nina", "atlantic", "pacific", "teleconnection",
        "oscillation", "index",
    ],
    CauseCategory.ANTHROPOGENIC: [
        "emission", "co2", "greenhouse", "pollution", "urban",
        "land_use", "deforestation", "aerosol", "dam", "reservoir",
    ],
    CauseCategory.LOCAL: [
        "topography", "river", "lake", "catchment", "drainage",
        "soil", "vegetation", "flood", "discharge", "runoff",
    ],
}


@dataclass
class Cause:
    """A cause in the Ishikawa diagram."""
    name: str
    category: CauseCategory
    score: float  # Confidence score (0-1)
    lag: int = 0  # Time lag (0 = direct)
    sub_causes: List["Cause"] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "score": self.score,
            "lag": self.lag,
            "sub_causes": [c.to_dict() for c in self.sub_causes],
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass
class IshikawaDiagram:
    """
    Ishikawa (Fishbone) diagram for cause-effect analysis.
    """
    effect: str
    causes: List[Cause] = field(default_factory=list)
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_causal_links(
        cls,
        effect: str,
        links: List[Dict],
        auto_categorize: bool = True,
    ) -> "IshikawaDiagram":
        """
        Create Ishikawa diagram from causal links (e.g., from PCMCI).
        
        Args:
            effect: The effect/outcome being analyzed
            links: List of causal links with source, target, score, lag
            auto_categorize: Automatically categorize causes
        """
        causes = []
        
        for link in links:
            # Get link info
            source = link.get("source", "Unknown")
            score = link.get("score", link.get("confidence", 0.5))
            lag = link.get("lag", 0)
            
            # Auto-categorize
            if auto_categorize:
                category = cls._categorize_variable(source)
            else:
                category = CauseCategory.LOCAL
            
            # Create cause
            cause = Cause(
                name=source,
                category=category,
                score=score,
                lag=lag,
                metadata={
                    "p_value": link.get("p_value"),
                    "strength": link.get("strength"),
                }
            )
            causes.append(cause)
        
        return cls(effect=effect, causes=causes)
    
    @staticmethod
    def _categorize_variable(var_name: str) -> CauseCategory:
        """Categorize a variable based on its name."""
        var_lower = var_name.lower().replace(" ", "_")
        
        for category, keywords in CLIMATE_CATEGORY_KEYWORDS.items():
            if any(kw in var_lower for kw in keywords):
                return category
        
        return CauseCategory.LOCAL
    
    def add_cause(
        self,
        name: str,
        category: CauseCategory,
        score: float,
        lag: int = 0,
        evidence: List[str] = None,
    ) -> Cause:
        """Add a cause to the diagram."""
        cause = Cause(
            name=name,
            category=category,
            score=score,
            lag=lag,
            evidence=evidence or [],
        )
        self.causes.append(cause)
        return cause
    
    def get_causes_by_category(self) -> Dict[CauseCategory, List[Cause]]:
        """Group causes by category."""
        grouped = {}
        for cause in self.causes:
            if cause.category not in grouped:
                grouped[cause.category] = []
            grouped[cause.category].append(cause)
        
        # Sort by score within each category
        for cat in grouped:
            grouped[cat].sort(key=lambda c: c.score, reverse=True)
        
        return grouped
    
    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            "effect": self.effect,
            "causes": [c.to_dict() for c in self.causes],
            "description": self.description,
            "timestamp": self.timestamp,
            "by_category": {
                cat.value: [c.to_dict() for c in causes]
                for cat, causes in self.get_causes_by_category().items()
            },
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_mermaid(self) -> str:
        """
        Export to Mermaid diagram format.
        
        Can be rendered with Mermaid.js or used in Markdown.
        """
        lines = ["```mermaid", "flowchart LR"]
        
        # Effect node (fish head)
        effect_id = "EFFECT"
        lines.append(f'    {effect_id}["{self.effect}"]')
        lines.append(f'    style {effect_id} fill:#f44336,color:white,stroke:#c62828')
        
        # Group by category
        by_category = self.get_causes_by_category()
        
        # Add categories as main branches
        for i, (category, causes) in enumerate(by_category.items()):
            cat_id = f"CAT_{i}"
            lines.append(f'    {cat_id}["{category.value}"]')
            lines.append(f'    {cat_id} --> {effect_id}')
            
            # Category styling
            colors = {
                CauseCategory.ATMOSPHERIC: "#2196F3",
                CauseCategory.OCEANIC: "#00BCD4",
                CauseCategory.CRYOSPHERIC: "#9C27B0",
                CauseCategory.TELECONNECTION: "#FF9800",
                CauseCategory.ANTHROPOGENIC: "#795548",
                CauseCategory.LOCAL: "#4CAF50",
            }
            color = colors.get(category, "#666666")
            lines.append(f'    style {cat_id} fill:{color},color:white')
            
            # Add causes as sub-branches
            for j, cause in enumerate(causes[:5]):  # Limit to top 5
                cause_id = f"C_{i}_{j}"
                score_pct = int(cause.score * 100)
                lag_str = f" [lag:{cause.lag}]" if cause.lag > 0 else ""
                lines.append(f'    {cause_id}("{cause.name}{lag_str}<br/>{score_pct}%")')
                lines.append(f'    {cause_id} --> {cat_id}')
                
                # Score-based opacity
                opacity = 0.3 + cause.score * 0.7
                lines.append(f'    style {cause_id} fill:{color},opacity:{opacity}')
        
        lines.append("```")
        return "\n".join(lines)
    
    def to_svg(
        self,
        width: int = 1200,
        height: int = 600,
    ) -> str:
        """
        Generate SVG representation of the Ishikawa diagram.
        """
        # Basic layout
        margin = 50
        spine_y = height // 2
        head_x = width - margin - 100
        tail_x = margin + 50
        
        # Colors by category
        colors = {
            CauseCategory.ATMOSPHERIC: "#2196F3",
            CauseCategory.OCEANIC: "#00BCD4",
            CauseCategory.CRYOSPHERIC: "#9C27B0",
            CauseCategory.TELECONNECTION: "#FF9800",
            CauseCategory.ANTHROPOGENIC: "#795548",
            CauseCategory.LOCAL: "#4CAF50",
        }
        
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            '<defs>',
            '  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            '    <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>',
            '  </marker>',
            '</defs>',
            
            # Background
            '<rect width="100%" height="100%" fill="#fafafa"/>',
            
            # Title
            f'<text x="{width//2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">Ishikawa Diagram</text>',
            
            # Main spine (backbone)
            f'<line x1="{tail_x}" y1="{spine_y}" x2="{head_x}" y2="{spine_y}" stroke="#333" stroke-width="3" marker-end="url(#arrowhead)"/>',
            
            # Fish head (effect)
            f'<ellipse cx="{head_x + 50}" cy="{spine_y}" rx="60" ry="40" fill="#f44336"/>',
            f'<text x="{head_x + 50}" y="{spine_y + 5}" text-anchor="middle" fill="white" font-size="11" font-weight="bold">',
            f'  <tspan x="{head_x + 50}" dy="0">{self.effect[:20]}</tspan>',
        ]
        
        # Wrap effect text if needed
        if len(self.effect) > 20:
            svg_parts.append(f'  <tspan x="{head_x + 50}" dy="14">{self.effect[20:40]}</tspan>')
        svg_parts.append('</text>')
        
        # Group by category
        by_category = self.get_causes_by_category()
        n_categories = len(by_category)
        
        # Calculate bone positions
        spine_length = head_x - tail_x - 100
        bone_spacing = spine_length / max(n_categories, 1)
        
        for i, (category, causes) in enumerate(by_category.items()):
            color = colors.get(category, "#666666")
            bone_x = tail_x + 50 + i * bone_spacing
            
            # Alternate top/bottom
            direction = 1 if i % 2 == 0 else -1
            bone_end_y = spine_y + direction * 150
            
            # Main bone (to category)
            svg_parts.append(
                f'<line x1="{bone_x}" y1="{spine_y}" x2="{bone_x + 50}" y2="{bone_end_y}" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            
            # Category label
            label_y = bone_end_y + direction * 20
            svg_parts.append(
                f'<text x="{bone_x + 50}" y="{label_y}" text-anchor="middle" '
                f'fill="{color}" font-weight="bold" font-size="12">{category.value}</text>'
            )
            
            # Sub-bones (causes)
            n_causes = min(len(causes), 4)  # Max 4 per category
            for j, cause in enumerate(causes[:n_causes]):
                sub_x = bone_x + 15 + j * 25
                progress = j / max(n_causes - 1, 1)
                sub_y_start = spine_y + direction * (30 + progress * 100)
                sub_y_end = sub_y_start + direction * 40
                
                # Score affects line thickness
                thickness = 1 + cause.score * 2
                
                svg_parts.append(
                    f'<line x1="{sub_x}" y1="{sub_y_start}" x2="{sub_x + 20}" y2="{sub_y_end}" '
                    f'stroke="{color}" stroke-width="{thickness}" opacity="{0.5 + cause.score * 0.5}"/>'
                )
                
                # Cause label
                label_offset = 10 if direction > 0 else -5
                score_pct = int(cause.score * 100)
                svg_parts.append(
                    f'<text x="{sub_x + 20}" y="{sub_y_end + label_offset}" '
                    f'font-size="9" fill="#333">{cause.name[:15]} ({score_pct}%)</text>'
                )
        
        # Legend
        legend_y = height - 70
        svg_parts.append(f'<text x="50" y="{legend_y}" font-size="11" font-weight="bold">Legend:</text>')
        
        for i, (cat, color) in enumerate(list(colors.items())[:6]):
            lx = 50 + (i % 3) * 200
            ly = legend_y + 15 + (i // 3) * 20
            svg_parts.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{color}"/>')
            svg_parts.append(f'<text x="{lx + 18}" y="{ly + 10}" font-size="10">{cat.value}</text>')
        
        svg_parts.append('</svg>')
        return "\n".join(svg_parts)
    
    def save(self, path: str, format: str = None) -> None:
        """
        Save diagram to file.
        
        Args:
            path: Output file path
            format: Format (auto-detected from extension if None)
        """
        if format is None:
            if path.endswith(".svg"):
                format = "svg"
            elif path.endswith(".json"):
                format = "json"
            elif path.endswith(".md"):
                format = "mermaid"
            else:
                format = "json"
        
        if format == "svg":
            content = self.to_svg()
        elif format == "mermaid":
            content = self.to_mermaid()
        else:
            content = self.to_json()
        
        with open(path, "w") as f:
            f.write(content)


def create_ishikawa_from_pcmci(
    pcmci_result: Any,
    effect: str,
) -> IshikawaDiagram:
    """
    Create Ishikawa diagram from PCMCI result.
    
    Args:
        pcmci_result: PCMCIResult from pcmci_engine
        effect: Name of the effect being analyzed
        
    Returns:
        IshikawaDiagram
    """
    # Convert PCMCI links to diagram causes
    links = [
        {
            "source": link.source,
            "target": link.target,
            "score": link.score,
            "lag": link.lag,
            "p_value": link.p_value,
            "strength": link.strength,
        }
        for link in pcmci_result.significant_links
        if link.target == effect or pcmci_result.var_names[-1] == link.target
    ]
    
    return IshikawaDiagram.from_causal_links(effect, links)


# CLI test
if __name__ == "__main__":
    print("=== Ishikawa Diagram Generator Test ===\n")
    
    # Create sample diagram
    diagram = IshikawaDiagram(
        effect="Lake Maggiore Flood Oct 2000",
        description="Causal analysis of severe flooding event"
    )
    
    # Add causes
    diagram.add_cause("NAO Index", CauseCategory.TELECONNECTION, 0.92, lag=7)
    diagram.add_cause("SST Anomaly", CauseCategory.OCEANIC, 0.85, lag=5)
    diagram.add_cause("Wind Stress", CauseCategory.ATMOSPHERIC, 0.78, lag=3)
    diagram.add_cause("Precipitation", CauseCategory.ATMOSPHERIC, 0.95, lag=1)
    diagram.add_cause("Pressure Gradient", CauseCategory.ATMOSPHERIC, 0.72, lag=2)
    diagram.add_cause("Sea Level Anomaly", CauseCategory.OCEANIC, 0.68, lag=2)
    diagram.add_cause("River Discharge", CauseCategory.LOCAL, 0.88, lag=0)
    diagram.add_cause("Catchment Saturation", CauseCategory.LOCAL, 0.75, lag=1)
    
    # Print summary
    print("Effect:", diagram.effect)
    print(f"\nCauses by category:")
    for cat, causes in diagram.get_causes_by_category().items():
        print(f"\n  {cat.value}:")
        for c in causes:
            print(f"    - {c.name}: score={c.score:.2f}, lag={c.lag}")
    
    # Generate outputs
    print("\n--- Mermaid Format ---")
    print(diagram.to_mermaid())
    
    print("\n--- SVG saved to: ishikawa_test.svg ---")
    diagram.save("ishikawa_test.svg")
    
    print("\n--- JSON saved to: ishikawa_test.json ---")
    diagram.save("ishikawa_test.json")
