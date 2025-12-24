# Pattern Detection Engine

A domain-agnostic framework for detecting patterns in heterogeneous data that correlate input conditions with outcomes.

## Overview

The Pattern Detection Engine identifies relationships between:
- **Conditions** (input features like temperature, speed, humidity)
- **Outcomes** (results like failures, peak demand, anomalies)

It works across multiple domains:
- **Manufacturing**: Batch conditions → quality failures
- **Energy**: Weather/time patterns → load forecasting
- **Finance**: Market conditions → risk events
- **IoT**: Sensor readings → equipment failures

## Key Features

### 1. Heterogeneous Data Handling
- Ingests CSV, JSON, Parquet, Excel, databases, APIs
- Auto-infers schemas and data types
- Handles missing values and outliers
- Creates time-based and lag features

### 2. Supervised Pattern Detection
Uses labeled outcomes to discover predictive rules:
```
When temperature >= 25°C AND speed > 150 → failure (78% confidence)
```

Supported methods:
- Random Forest
- XGBoost
- Decision Trees
- Logistic Regression
- SVM
- Neural Networks

### 3. Unsupervised Pattern Detection
Discovers patterns without labels:
- **Anomaly Detection**: Isolation Forest, LOF
- **Clustering**: K-Means, DBSCAN
- **Association Rules**: Frequent itemset mining

### 4. Causal Discovery
Identifies time-lagged causal relationships:
```
Temperature at T-24h causes failure at T
```

Methods:
- PCMCI (Tigramite)
- Granger Causality
- Correlation Analysis

### 5. Flexible Output
- JSON reports
- Interactive HTML reports
- CSV/Parquet exports
- Alert webhooks

## Installation

```bash
# Core dependencies
pip install pandas numpy scikit-learn

# Optional: for XGBoost
pip install xgboost

# Optional: for causal discovery
pip install tigramite statsmodels

# Optional: for imbalanced data
pip install imbalanced-learn

# Optional: for association rules
pip install mlxtend
```

## Quick Start

```python
from pattern_engine import (
    PatternEngineConfig,
    DataTidier,
    SupervisedDetector,
    PatternReporter,
)

# 1. Load your data
import pandas as pd
df = pd.read_csv("manufacturing_data.csv")

# 2. Prepare features and target
X = df[["temperature", "speed", "humidity"]]
y = df["failure"]

# 3. Preprocess
tidier = DataTidier()
X_clean, report = tidier.tidy(X)

# 4. Detect patterns
detector = SupervisedDetector()
result = detector.fit_detect(X_clean, y)

# 5. View patterns
for pattern in result.top_patterns[:5]:
    print(pattern)
    # [causation] temperature >= 25.0 AND speed > 150.0 → failure=True (conf=0.78)

# 6. Generate report
reporter = PatternReporter()
reporter.generate(result, name="quality_analysis")
```

## Example: Manufacturing Batch Failure

```python
from pattern_engine.examples import run_manufacturing_example

# Run the full example
result, unsup_result, causal_patterns = run_manufacturing_example()

# Output:
# Discovered patterns like:
# - High temperature (≥25°C) + high speed (>150) → 78% failure rate
# - Material batch B → 15% higher failure rate
# - High humidity (>65%) → 12% higher failure rate
```

## Example: Load Forecasting

```python
from pattern_engine.examples import run_load_forecasting_example

result, causal_graph = run_load_forecasting_example()

# Output:
# Peak demand patterns:
# - Hour 14-17 on weekdays → peak demand
# - Temperature > 28°C → 40% peak probability
# - Monday after holiday → demand spike
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW HETEROGENEOUS DATA                    │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                       │
│  DataIngestor → DataTidier → SchemaInferrer                 │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    PATTERN DETECTION                         │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │    SUPERVISED    │      │   UNSUPERVISED   │            │
│  │  (with labels)   │      │  (no labels)     │            │
│  └──────────────────┘      └──────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    CAUSAL INFERENCE                          │
│  CausalDiscovery (PCMCI, Granger, Correlation)              │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                              │
│  PatternReporter → JSON, HTML, CSV, Alerts                  │
└─────────────────────────────────────────────────────────────┘
```

## Pattern Data Structure

```python
Pattern(
    pattern_type=PatternType.CAUSATION,
    conditions=[
        Condition("temperature", ConditionOperator.GREATER_EQUAL, 25.0),
        Condition("speed", ConditionOperator.GREATER, 150.0),
    ],
    outcome=Outcome(
        name="failure",
        value=True,
        probability=0.78,
        lag=24,  # hours
    ),
    confidence=0.78,
    support=0.15,  # 15% of cases
    causal_strength=0.45,
    description="High temp + speed causes failure with 24h lag"
)
```

## Configuration

```python
from pattern_engine import (
    PatternEngineConfig,
    PreprocessingConfig,
    SupervisedConfig,
    CausalConfig,
    OutputConfig,
    DetectionMethod,
)

config = PatternEngineConfig(
    name="my_analysis",
    preprocessing=PreprocessingConfig(
        missing_strategy="interpolate",
        normalize_numeric=True,
        create_lag_features=True,
        lag_periods=[1, 6, 24],
    ),
    supervised=SupervisedConfig(
        methods=[DetectionMethod.RANDOM_FOREST, DetectionMethod.XGBOOST],
        cv_folds=5,
        handle_imbalance=True,
    ),
    causal=CausalConfig(
        max_lag=24,
        alpha_level=0.05,
    ),
    output=OutputConfig(
        formats=["json", "html"],
        output_dir="./reports",
    ),
)
```

## Module Structure

```
pattern_engine/
├── __init__.py           # Main exports
├── core/
│   ├── config.py         # Configuration dataclasses
│   └── pattern.py        # Pattern, Condition, Outcome classes
├── preprocessing/
│   ├── ingestor.py       # Data ingestion from various sources
│   ├── tidier.py         # Data cleaning and transformation
│   └── schema.py         # Schema inference and validation
├── detection/
│   ├── base.py           # Base detector class
│   ├── supervised.py     # Supervised pattern detection
│   └── unsupervised.py   # Unsupervised pattern detection
├── causal/
│   └── discovery.py      # Causal relationship discovery
├── output/
│   └── reporter.py       # Report generation
└── examples/
    ├── manufacturing_example.py
    └── load_forecasting_example.py
```

## API Reference

### DataIngestor
```python
ingestor = DataIngestor()
data = ingestor.ingest(DataSourceConfig(
    name="batches",
    source_type="csv",
    path_or_uri="data/batches.csv"
))
```

### DataTidier
```python
tidier = DataTidier(PreprocessingConfig())
X_clean, report = tidier.tidy(X, target_column="failure")
```

### SupervisedDetector
```python
detector = SupervisedDetector(SupervisedConfig())
result = detector.fit_detect(X, y)
predictions = detector.predict(X_new)
```

### UnsupervisedDetector
```python
detector = UnsupervisedDetector(UnsupervisedConfig())
result = detector.fit_detect(X)
# result.patterns contains anomalies and clusters
```

### CausalDiscovery
```python
discovery = CausalDiscovery(CausalConfig())
graph = discovery.discover(df, target="outcome")
patterns = discovery.discover_patterns(df, target="outcome")
```

### PatternReporter
```python
reporter = PatternReporter(OutputConfig())
outputs = reporter.generate(result, name="my_report")
```

## License

MIT License
