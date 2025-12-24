"""
Pattern reporter for generating various output formats.

Supports:
- JSON reports
- HTML reports with visualizations
- CSV exports
- Dashboard data
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import logging
from datetime import datetime

import pandas as pd

from ..core.config import OutputConfig, OutputFormat
from ..core.pattern import Pattern, PatternDatabase
from ..detection.base import DetectionResult
from ..causal.discovery import CausalGraph

logger = logging.getLogger(__name__)


class PatternReporter:
    """
    Generate reports from detected patterns.
    
    Example:
        reporter = PatternReporter(OutputConfig(
            formats=[OutputFormat.JSON, OutputFormat.HTML],
            output_dir=Path("./reports")
        ))
        
        reporter.generate(
            detection_result,
            causal_graph=graph,
            name="manufacturing_analysis"
        )
    """
    
    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        result: DetectionResult,
        causal_graph: Optional[CausalGraph] = None,
        name: str = "pattern_report",
        data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Path]:
        """
        Generate reports in configured formats.
        
        Args:
            result: Detection result from supervised/unsupervised detector
            causal_graph: Optional causal graph
            name: Report name
            data: Original data (for visualizations)
            
        Returns:
            Dictionary of format -> output file path
        """
        outputs = {}
        
        for fmt in self.config.formats:
            if fmt == OutputFormat.JSON:
                path = self._generate_json(result, causal_graph, name)
            elif fmt == OutputFormat.HTML:
                path = self._generate_html(result, causal_graph, name, data)
            elif fmt == OutputFormat.CSV:
                path = self._generate_csv(result, name)
            elif fmt == OutputFormat.PARQUET:
                path = self._generate_parquet(result, name)
            else:
                continue
            
            outputs[fmt.value] = path
            logger.info(f"Generated {fmt.value} report: {path}")
        
        return outputs
    
    def _generate_json(
        self,
        result: DetectionResult,
        causal_graph: Optional[CausalGraph],
        name: str,
    ) -> Path:
        """Generate JSON report."""
        report = {
            "name": name,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_patterns": len(result.patterns),
                "n_features": result.metadata.get("n_features", 0),
                "n_samples": result.metadata.get("n_samples", 0),
            },
            "metrics": result.metrics,
            "feature_importance": result.feature_importance,
            "patterns": [p.to_dict() for p in result.patterns],
        }
        
        if causal_graph:
            report["causal_graph"] = causal_graph.to_dict()
        
        path = self.config.output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return path
    
    def _generate_html(
        self,
        result: DetectionResult,
        causal_graph: Optional[CausalGraph],
        name: str,
        data: Optional[pd.DataFrame],
    ) -> Path:
        """Generate HTML report with visualizations."""
        
        # Pattern cards
        pattern_cards = []
        for i, pattern in enumerate(result.patterns[:20], 1):  # Top 20
            conditions_html = "<br>".join(
                f"• {c.feature} {c.operator.value} {c.value}"
                for c in pattern.conditions
            )
            
            outcome_html = ""
            if pattern.outcome:
                outcome_html = f"""
                <div class="outcome">
                    → <strong>{pattern.outcome.name}</strong> = {pattern.outcome.value}
                    {f"(probability: {pattern.outcome.probability:.2%})" if pattern.outcome.probability else ""}
                    {f"(lag: {pattern.outcome.lag})" if pattern.outcome.lag else ""}
                </div>
                """
            
            card = f"""
            <div class="pattern-card">
                <div class="pattern-header">
                    <span class="pattern-type">{pattern.pattern_type.value.upper()}</span>
                    <span class="pattern-id">#{i}</span>
                </div>
                <div class="conditions">
                    {conditions_html}
                </div>
                {outcome_html}
                <div class="metrics">
                    <span>Confidence: {pattern.confidence:.2%}</span>
                    <span>Support: {pattern.support:.2%}</span>
                    {f"<span>Causal strength: {pattern.causal_strength:.3f}</span>" if pattern.causal_strength else ""}
                </div>
                <div class="description">{pattern.description}</div>
            </div>
            """
            pattern_cards.append(card)
        
        # Feature importance chart (simple bar)
        importance_html = ""
        if result.feature_importance:
            sorted_importance = sorted(
                result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            bars = []
            max_imp = max(imp for _, imp in sorted_importance) if sorted_importance else 1
            for feat, imp in sorted_importance:
                width = (imp / max_imp) * 100
                bars.append(f"""
                <div class="importance-bar">
                    <span class="feature-name">{feat}</span>
                    <div class="bar-container">
                        <div class="bar" style="width: {width}%"></div>
                    </div>
                    <span class="importance-value">{imp:.3f}</span>
                </div>
                """)
            importance_html = f"""
            <div class="section">
                <h2>Feature Importance</h2>
                <div class="importance-chart">
                    {"".join(bars)}
                </div>
            </div>
            """
        
        # Metrics summary
        metrics_html = ""
        if result.metrics:
            metrics_items = "".join(
                f"<div class='metric'><span class='metric-name'>{k}</span><span class='metric-value'>{v:.3f}</span></div>"
                for k, v in result.metrics.items()
            )
            metrics_html = f"""
            <div class="section">
                <h2>Model Metrics</h2>
                <div class="metrics-grid">{metrics_items}</div>
            </div>
            """
        
        # Causal graph section
        causal_html = ""
        if causal_graph and causal_graph.edges:
            edges_list = "".join(
                f"<li>{e.source} → {e.target} (lag={e.lag}, strength={e.strength:.3f})</li>"
                for e in sorted(causal_graph.edges, key=lambda e: abs(e.strength), reverse=True)[:10]
            )
            causal_html = f"""
            <div class="section">
                <h2>Causal Relationships</h2>
                <ul class="causal-edges">{edges_list}</ul>
            </div>
            """
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Pattern Detection Report: {name}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5;
            color: #333;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; margin-bottom: 10px; }}
        .timestamp {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 30px; }}
        
        .summary {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{ margin: 0 0 10px 0; color: #7f8c8d; font-size: 0.9em; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        
        .section {{ 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{ margin-top: 0; color: #2c3e50; }}
        
        .pattern-card {{
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .pattern-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        .pattern-type {{
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }}
        .pattern-id {{ color: #7f8c8d; }}
        .conditions {{ margin: 10px 0; font-family: monospace; }}
        .outcome {{ 
            background: #e8f5e9;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .metrics {{ 
            display: flex; 
            gap: 20px; 
            font-size: 0.9em; 
            color: #7f8c8d;
            margin: 10px 0;
        }}
        .description {{ font-style: italic; color: #666; }}
        
        .importance-chart {{ max-width: 600px; }}
        .importance-bar {{ 
            display: flex; 
            align-items: center; 
            margin: 8px 0; 
        }}
        .feature-name {{ width: 150px; font-size: 0.9em; }}
        .bar-container {{ 
            flex: 1; 
            background: #e0e0e0; 
            height: 20px; 
            border-radius: 4px; 
            overflow: hidden;
        }}
        .bar {{ background: #3498db; height: 100%; }}
        .importance-value {{ width: 60px; text-align: right; font-size: 0.9em; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #fafafa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-name {{ display: block; color: #7f8c8d; font-size: 0.9em; }}
        .metric-value {{ display: block; font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
        
        .causal-edges {{ list-style: none; padding: 0; }}
        .causal-edges li {{ 
            padding: 10px;
            background: #fafafa;
            margin: 5px 0;
            border-radius: 4px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Pattern Detection Report: {name}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Patterns Discovered</h3>
                <div class="value">{len(result.patterns)}</div>
            </div>
            <div class="summary-card">
                <h3>Features Analyzed</h3>
                <div class="value">{result.metadata.get('n_features', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>Samples</h3>
                <div class="value">{result.metadata.get('n_samples', 0):,}</div>
            </div>
        </div>
        
        {metrics_html}
        {importance_html}
        {causal_html}
        
        <div class="section">
            <h2>Discovered Patterns</h2>
            {"".join(pattern_cards)}
        </div>
    </div>
</body>
</html>
        """
        
        path = self.config.output_dir / f"{name}.html"
        with open(path, "w") as f:
            f.write(html)
        
        return path
    
    def _generate_csv(
        self,
        result: DetectionResult,
        name: str,
    ) -> Path:
        """Generate CSV export of patterns."""
        rows = []
        for pattern in result.patterns:
            row = {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type.value,
                "conditions": " AND ".join(str(c) for c in pattern.conditions),
                "outcome_name": pattern.outcome.name if pattern.outcome else None,
                "outcome_value": pattern.outcome.value if pattern.outcome else None,
                "outcome_probability": pattern.outcome.probability if pattern.outcome else None,
                "confidence": pattern.confidence,
                "support": pattern.support,
                "lift": pattern.lift,
                "causal_strength": pattern.causal_strength,
                "causal_lag": pattern.causal_lag,
                "description": pattern.description,
                "sample_size": pattern.sample_size,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        path = self.config.output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        
        return path
    
    def _generate_parquet(
        self,
        result: DetectionResult,
        name: str,
    ) -> Path:
        """Generate Parquet export of patterns."""
        rows = [p.to_dict() for p in result.patterns]
        df = pd.DataFrame(rows)
        path = self.config.output_dir / f"{name}.parquet"
        df.to_parquet(path)
        
        return path
    
    def generate_alert(
        self,
        pattern: Pattern,
        matched_data: Dict[str, Any],
        severity: str = "warning",
    ) -> Dict[str, Any]:
        """
        Generate an alert for a matched pattern.
        
        Can be sent to webhook, logged, etc.
        """
        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type.value,
            "description": pattern.description,
            "conditions_met": [str(c) for c in pattern.conditions],
            "predicted_outcome": pattern.outcome.to_dict() if pattern.outcome else None,
            "matched_data": matched_data,
            "confidence": pattern.confidence,
        }
        
        # Send to webhook if configured
        if self.config.alert_webhook:
            self._send_webhook(alert)
        
        return alert
    
    def _send_webhook(self, alert: Dict[str, Any]) -> None:
        """Send alert to webhook."""
        try:
            import requests
            response = requests.post(
                self.config.alert_webhook,
                json=alert,
                timeout=10,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
