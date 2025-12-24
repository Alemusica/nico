"""
Example: Load Forecasting Pattern Detection

This example demonstrates how to use the Pattern Detection Engine
for electricity load forecasting - identifying patterns that predict
high/low demand periods.

Scenario:
- Hourly electricity load data
- Features: temperature, humidity, hour, day of week, holidays
- Outcome: peak demand events

Patterns discovered:
- "High temperature (>30°C) during afternoon (14:00-17:00) on weekdays → peak demand"
- "Monday morning after holiday → demand spike"
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
    PreprocessingConfig,
    SupervisedConfig,
    UnsupervisedConfig,
    CausalConfig,
    OutputConfig,
    DetectionMethod,
    OutputFormat,
)
from pattern_engine.preprocessing import DataTidier
from pattern_engine.detection import SupervisedDetector, UnsupervisedDetector
from pattern_engine.causal import CausalDiscovery
from pattern_engine.output import PatternReporter


def generate_synthetic_load_data(n_hours: int = 8760) -> pd.DataFrame:
    """
    Generate synthetic electricity load data (1 year hourly).
    
    Simulates realistic patterns:
    - Daily pattern (peak in afternoon)
    - Weekly pattern (lower on weekends)
    - Seasonal pattern (AC in summer, heating in winter)
    - Temperature correlation
    """
    np.random.seed(42)
    
    # Time range (1 year hourly)
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_hours)]
    
    df = pd.DataFrame({"timestamp": timestamps})
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"] >= 5
    
    # Holidays (simplified)
    holidays = [
        datetime(2024, 1, 1),   # New Year
        datetime(2024, 7, 4),   # Independence Day
        datetime(2024, 12, 25), # Christmas
    ]
    df["is_holiday"] = df["timestamp"].dt.date.isin([h.date() for h in holidays])
    
    # Temperature (seasonal pattern + daily variation)
    base_temp = 15 + 10 * np.sin(2 * np.pi * (df["month"] - 4) / 12)  # Seasonal
    daily_temp = 5 * np.sin(2 * np.pi * (df["hour"] - 6) / 24)  # Daily
    df["temperature"] = base_temp + daily_temp + np.random.normal(0, 3, n_hours)
    
    # Humidity
    df["humidity"] = 50 + 20 * np.sin(2 * np.pi * df["month"] / 12) + np.random.normal(0, 10, n_hours)
    df["humidity"] = df["humidity"].clip(20, 95)
    
    # Cloud cover
    df["cloud_cover"] = np.random.beta(2, 5, n_hours) * 100
    
    # Base load (daily pattern)
    daily_pattern = np.array([
        0.6, 0.55, 0.5, 0.5, 0.55, 0.65,  # 0-5 (night/early morning)
        0.75, 0.85, 0.95, 1.0, 1.0, 0.95,  # 6-11 (morning)
        0.9, 0.95, 1.0, 1.05, 1.1, 1.05,   # 12-17 (afternoon peak)
        0.95, 0.9, 0.85, 0.8, 0.7, 0.65,   # 18-23 (evening)
    ])
    
    base_load = 1000 * daily_pattern[df["hour"]]
    
    # Weekend reduction
    base_load = np.where(df["is_weekend"], base_load * 0.8, base_load)
    
    # Holiday reduction
    base_load = np.where(df["is_holiday"], base_load * 0.6, base_load)
    
    # Temperature effect (AC in summer, heating in winter)
    temp_effect = np.where(
        df["temperature"] > 25,
        (df["temperature"] - 25) * 30,  # AC
        np.where(
            df["temperature"] < 10,
            (10 - df["temperature"]) * 25,  # Heating
            0
        )
    )
    
    # Final load with noise
    df["load_mw"] = base_load + temp_effect + np.random.normal(0, 50, n_hours)
    df["load_mw"] = df["load_mw"].clip(400, 2000)
    
    # Peak demand flag (top 10%)
    threshold = df["load_mw"].quantile(0.9)
    df["is_peak"] = df["load_mw"] >= threshold
    
    # Demand category
    df["demand_level"] = pd.cut(
        df["load_mw"],
        bins=[0, 600, 900, 1200, 2000],
        labels=["low", "normal", "high", "peak"]
    )
    
    return df


def run_load_forecasting_example():
    """
    Complete example of pattern detection for load forecasting.
    """
    print("=" * 60)
    print("Electricity Load Forecasting Pattern Detection")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic load data (1 year hourly)...")
    df = generate_synthetic_load_data(n_hours=8760)
    print(f"   Generated {len(df)} hourly records")
    print(f"   Peak demand rate: {df['is_peak'].mean():.1%}")
    print(f"   Load range: {df['load_mw'].min():.0f} - {df['load_mw'].max():.0f} MW")
    
    # 2. Configure the pattern engine
    print("\n2. Configuring pattern engine...")
    config = PatternEngineConfig(
        name="load_forecasting",
        description="Detect patterns for electricity demand forecasting",
        preprocessing=PreprocessingConfig(
            normalize_numeric=True,
            create_lag_features=True,
            lag_periods=[1, 24, 168],  # 1 hour, 1 day, 1 week ago
        ),
        supervised=SupervisedConfig(
            methods=[DetectionMethod.RANDOM_FOREST, DetectionMethod.XGBOOST],
            cv_folds=5,
        ),
        causal=CausalConfig(
            enabled=True,
            max_lag=24,  # Look for effects up to 24 hours
            alpha_level=0.01,
        ),
        output=OutputConfig(
            formats=[OutputFormat.JSON, OutputFormat.HTML],
            output_dir=Path("./pattern_output/load_forecasting"),
        ),
    )
    
    # 3. Preprocess data
    print("\n3. Preprocessing data...")
    
    # Feature selection
    feature_cols = [
        "hour", "day_of_week", "month",
        "is_weekend", "is_holiday",
        "temperature", "humidity", "cloud_cover"
    ]
    
    X = df[feature_cols].copy()
    X["is_weekend"] = X["is_weekend"].astype(int)
    X["is_holiday"] = X["is_holiday"].astype(int)
    
    y = df["is_peak"].astype(int)
    
    tidier = DataTidier(config.preprocessing)
    X_clean, report = tidier.tidy(X)
    print(f"   Shape after preprocessing: {X_clean.shape}")
    
    # 4. Supervised Pattern Detection
    print("\n4. Running supervised pattern detection...")
    supervised = SupervisedDetector(config.supervised)
    sup_result = supervised.fit_detect(X_clean, y)
    
    print(f"\n   Discovered {len(sup_result.patterns)} patterns")
    print(f"   Best model: {sup_result.metadata.get('best_model')}")
    print(f"   Accuracy: {sup_result.metrics.get('accuracy', 0):.2%}")
    print(f"   F1 Score: {sup_result.metrics.get('f1', 0):.3f}")
    
    print("\n   Top patterns for peak demand:")
    for i, pattern in enumerate(sup_result.top_patterns[:5], 1):
        if pattern.outcome and pattern.outcome.value == 1:
            print(f"   {i}. {pattern}")
    
    # 5. Feature Importance
    print("\n5. Feature importance for peak prediction:")
    for feat, imp in sorted(
        sup_result.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:8]:
        print(f"   - {feat}: {imp:.3f}")
    
    # 6. Causal Discovery
    print("\n6. Running causal discovery...")
    causal = CausalDiscovery(config.causal)
    
    # Prepare time-series data
    causal_df = df[["temperature", "humidity", "load_mw"]].copy()
    
    graph = causal.discover(causal_df, target="load_mw", method="granger")
    
    print(f"   Found {len(graph.edges)} causal relationships")
    
    causes_of_load = graph.get_causes("load_mw")
    for edge in causes_of_load:
        print(f"   - {edge.source} → load_mw (lag={edge.lag}, strength={edge.strength:.3f})")
    
    # 7. Generate Reports
    print("\n7. Generating reports...")
    reporter = PatternReporter(config.output)
    
    outputs = reporter.generate(
        result=sup_result,
        causal_graph=graph,
        name="load_forecasting_analysis",
        data=df,
    )
    
    for fmt, path in outputs.items():
        print(f"   {fmt}: {path}")
    
    # 8. Demand Profile Analysis
    print("\n8. Demand profile analysis...")
    
    # Analyze patterns by hour
    hourly_peak = df.groupby("hour")["is_peak"].mean()
    peak_hours = hourly_peak[hourly_peak > 0.15].index.tolist()
    print(f"   Peak-prone hours: {peak_hours}")
    
    # Analyze patterns by day
    daily_peak = df.groupby("day_of_week")["is_peak"].mean()
    peak_days = daily_peak[daily_peak > 0.1].index.tolist()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"   Peak-prone days: {[day_names[d] for d in peak_days]}")
    
    # Temperature threshold
    hot_days = df[df["temperature"] > 28]
    print(f"   Peak rate on hot days (>28°C): {hot_days['is_peak'].mean():.1%}")
    
    # 9. Forecasting simulation
    print("\n9. Testing prediction on new data points...")
    
    test_cases = [
        {"hour": 15, "day_of_week": 2, "month": 7, "is_weekend": 0, "is_holiday": 0,
         "temperature": 32, "humidity": 60, "cloud_cover": 10},
        {"hour": 3, "day_of_week": 6, "month": 1, "is_weekend": 1, "is_holiday": 0,
         "temperature": 5, "humidity": 70, "cloud_cover": 80},
    ]
    
    for i, case in enumerate(test_cases, 1):
        # Prepare for prediction
        case_df = pd.DataFrame([case])
        case_df, _ = tidier.transform(case_df)
        
        # Ensure columns match
        for col in X_clean.columns:
            if col not in case_df.columns:
                case_df[col] = 0
        case_df = case_df[X_clean.columns]
        
        pred = supervised.predict(case_df)
        proba = supervised.predict_proba(case_df)
        
        print(f"\n   Test case {i}:")
        print(f"   - Hour: {case['hour']}, Temp: {case['temperature']}°C")
        print(f"   - Predicted: {'PEAK' if pred.iloc[0] == 1 else 'Normal'}")
        print(f"   - Peak probability: {proba.iloc[0, 1]:.1%}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    
    return sup_result, graph


if __name__ == "__main__":
    run_load_forecasting_example()
