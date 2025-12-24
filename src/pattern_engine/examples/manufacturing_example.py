"""
Example: Manufacturing Batch Failure Prediction

This example demonstrates how to use the Pattern Detection Engine
to identify conditions that lead to manufacturing failures.

Scenario:
- Hose manufacturing process
- Batch data includes: extrusion temperature, speed, diameter, humidity
- Outcome: burst test pass/fail

Pattern discovered:
"When temperature >= 25°C during extrusion AND speed > 150,
 there's a 78% probability of burst failure"
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pattern_engine import (
    PatternEngineConfig,
    DataSourceConfig,
    PreprocessingConfig,
    SupervisedConfig,
    UnsupervisedConfig,
    CausalConfig,
    OutputConfig,
    DetectionMethod,
    OutputFormat,
)
from pattern_engine.preprocessing import DataIngestor, DataTidier
from pattern_engine.detection import SupervisedDetector, UnsupervisedDetector
from pattern_engine.causal import CausalDiscovery
from pattern_engine.output import PatternReporter


def generate_synthetic_manufacturing_data(n_batches: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic manufacturing data for demonstration.
    
    The data simulates:
    - Batch ID
    - Production timestamp
    - Extrusion temperature (ambient + process)
    - Extrusion speed
    - Inner liner diameter
    - Humidity level
    - Burst test result (pass/fail)
    
    Hidden pattern: High temperature + high speed → failure
    """
    np.random.seed(42)
    
    # Time range
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i*8) for i in range(n_batches)]
    
    # Base features
    data = {
        "batch_id": [f"BATCH_{i:05d}" for i in range(n_batches)],
        "timestamp": timestamps,
        "extrusion_temp": np.random.normal(22, 3, n_batches),  # °C
        "extrusion_speed": np.random.normal(140, 15, n_batches),  # mm/s
        "inner_diameter": np.random.normal(25.0, 0.5, n_batches),  # mm
        "humidity": np.random.normal(50, 10, n_batches),  # %
        "material_batch": np.random.choice(["A", "B", "C"], n_batches),
        "operator_shift": np.random.choice([1, 2, 3], n_batches),
    }
    
    df = pd.DataFrame(data)
    
    # Add seasonal variation (summer = higher temp)
    df["month"] = df["timestamp"].dt.month
    df["extrusion_temp"] += (df["month"] - 6).abs() * -0.5  # Cooler in summer months
    
    # THE HIDDEN PATTERN:
    # High temperature (>=25°C) AND high speed (>150) → failure
    # Also: humidity > 65% increases failure risk
    
    failure_prob = np.zeros(n_batches)
    
    # Temperature effect
    high_temp_mask = df["extrusion_temp"] >= 25
    failure_prob[high_temp_mask] += 0.4
    
    # Speed effect
    high_speed_mask = df["extrusion_speed"] > 150
    failure_prob[high_speed_mask] += 0.3
    
    # Combined effect (synergistic)
    combined_mask = high_temp_mask & high_speed_mask
    failure_prob[combined_mask] += 0.2
    
    # Humidity effect
    high_humidity_mask = df["humidity"] > 65
    failure_prob[high_humidity_mask] += 0.15
    
    # Material B has higher failure rate
    material_b_mask = df["material_batch"] == "B"
    failure_prob[material_b_mask] += 0.1
    
    # Cap probability
    failure_prob = np.clip(failure_prob, 0, 0.95)
    
    # Generate outcomes
    df["burst_test_failed"] = np.random.binomial(1, failure_prob)
    
    # Add some measurement noise
    df["pressure_at_burst"] = np.where(
        df["burst_test_failed"] == 1,
        np.random.normal(180, 20, n_batches),  # Failed: lower pressure
        np.random.normal(250, 15, n_batches)   # Passed: higher pressure
    )
    
    return df


def run_manufacturing_example():
    """
    Complete example of pattern detection for manufacturing.
    """
    print("=" * 60)
    print("Manufacturing Batch Failure Pattern Detection")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic manufacturing data...")
    df = generate_synthetic_manufacturing_data(n_batches=1000)
    print(f"   Generated {len(df)} batches")
    print(f"   Failure rate: {df['burst_test_failed'].mean():.1%}")
    
    # 2. Configure the pattern engine
    print("\n2. Configuring pattern engine...")
    config = PatternEngineConfig(
        name="manufacturing_quality",
        description="Detect patterns leading to burst test failures",
        preprocessing=PreprocessingConfig(
            missing_strategy="infer",
            normalize_numeric=True,
            create_time_features=True,
        ),
        supervised=SupervisedConfig(
            methods=[DetectionMethod.RANDOM_FOREST, DetectionMethod.DECISION_TREE],
            cv_folds=5,
            handle_imbalance=True,
        ),
        unsupervised=UnsupervisedConfig(
            methods=[DetectionMethod.ISOLATION_FOREST, DetectionMethod.KMEANS],
            contamination=0.1,
        ),
        causal=CausalConfig(
            enabled=True,
            max_lag=5,
            alpha_level=0.05,
        ),
        output=OutputConfig(
            formats=[OutputFormat.JSON, OutputFormat.HTML],
            output_dir=Path("./pattern_output/manufacturing"),
        ),
    )
    
    # 3. Preprocess data
    print("\n3. Preprocessing data...")
    tidier = DataTidier(config.preprocessing)
    
    # Prepare features (exclude non-predictive columns)
    feature_cols = [
        "extrusion_temp", "extrusion_speed", "inner_diameter",
        "humidity", "operator_shift"
    ]
    
    X = df[feature_cols].copy()
    y = df["burst_test_failed"]
    
    X_clean, report = tidier.tidy(
        X,
        target_column=None,
        timestamp_column=None,
    )
    print(f"   {report.summary()}")
    
    # 4. Supervised Pattern Detection
    print("\n4. Running supervised pattern detection...")
    supervised = SupervisedDetector(config.supervised)
    sup_result = supervised.fit_detect(X_clean, y)
    
    print(f"\n   Discovered {len(sup_result.patterns)} patterns")
    print(f"   Best model: {sup_result.metadata.get('best_model')}")
    print(f"   Metrics: {sup_result.metrics}")
    
    print("\n   Top 3 patterns:")
    for i, pattern in enumerate(sup_result.top_patterns[:3], 1):
        print(f"   {i}. {pattern}")
    
    # 5. Feature Importance
    print("\n5. Feature importance:")
    for feat, imp in sorted(
        sup_result.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"   - {feat}: {imp:.3f}")
    
    # 6. Unsupervised Pattern Detection (Anomaly Detection)
    print("\n6. Running unsupervised pattern detection...")
    unsupervised = UnsupervisedDetector(config.unsupervised)
    unsup_result = unsupervised.fit_detect(X_clean)
    
    print(f"   Found {unsup_result.metrics.get('n_anomalies', 0)} anomalies")
    print(f"   Found {unsup_result.metrics.get('n_clusters', 0)} clusters")
    
    # 7. Causal Discovery
    print("\n7. Running causal discovery...")
    causal = CausalDiscovery(config.causal)
    
    # Add target to feature data for causal analysis
    causal_df = X_clean.copy()
    causal_df["failure"] = y.values
    
    causal_patterns = causal.discover_patterns(causal_df, target="failure")
    
    print(f"   Found {len(causal_patterns)} causal patterns")
    for pattern in causal_patterns[:3]:
        print(f"   - {pattern.conditions[0].feature} → failure")
        print(f"     Strength: {pattern.causal_strength:.3f}, Lag: {pattern.causal_lag}")
    
    # 8. Generate Reports
    print("\n8. Generating reports...")
    reporter = PatternReporter(config.output)
    
    outputs = reporter.generate(
        result=sup_result,
        causal_graph=causal._graph,
        name="manufacturing_batch_analysis",
        data=df,
    )
    
    for fmt, path in outputs.items():
        print(f"   {fmt}: {path}")
    
    # 9. Example: Match new data against patterns
    print("\n9. Testing pattern matching on new batch...")
    new_batch = {
        "extrusion_temp": 26.5,  # High!
        "extrusion_speed": 155,  # High!
        "inner_diameter": 25.1,
        "humidity": 48,
        "operator_shift": 1,
    }
    
    # Find matching patterns
    matches = sup_result.database.match_data(new_batch)
    
    print(f"   New batch conditions: {new_batch}")
    print(f"   Matched {len(matches)} patterns")
    
    if matches:
        print(f"   ⚠️  WARNING: High risk pattern detected!")
        print(f"   Predicted outcome: {matches[0].predicted_outcome}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    
    return sup_result, unsup_result, causal_patterns


if __name__ == "__main__":
    run_manufacturing_example()
