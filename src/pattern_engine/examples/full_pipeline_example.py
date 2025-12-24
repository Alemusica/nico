"""
Full Pipeline Example - Manufacturing Pattern Detection
=======================================================

Complete example showing:
1. Data loading
2. TSFresh feature extraction
3. Supervised detection (RandomForest)
4. Association rules mining (mlxtend)
5. Physics validation
6. Gray zone detection
7. Report generation

Run this script:
    python -m src.pattern_engine.examples.full_pipeline_example
"""

import os
import sys
from pathlib import Path
import logging
import warnings

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def main():
    """Run the full pattern detection pipeline."""
    
    print("=" * 70)
    print("PATTERN DETECTION ENGINE - Full Pipeline Demo")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n📂 STEP 1: Loading data...")
    
    data_path = Path(__file__).parent / "data" / "manufacturing_batches.csv"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Creating sample data inline...")
        df = create_sample_data()
    else:
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Batches: {df['batch_id'].nunique()}")
    print(f"   Failures: {df.groupby('batch_id')['failure'].max().sum()} / {df['batch_id'].nunique()}")
    
    # =========================================================================
    # STEP 2: Feature Extraction (tsfresh)
    # =========================================================================
    print("\n🔬 STEP 2: Extracting features with tsfresh...")
    
    try:
        from src.pattern_engine.preprocessing import TSFreshExtractor, FeatureExtractionConfig
        
        extractor = TSFreshExtractor(FeatureExtractionConfig(
            mode='minimal',  # Use 'efficient' for more features
            n_jobs=1,
            disable_progressbar=True,
        ))
        
        value_cols = ['temperature', 'humidity', 'extrusion_speed', 'pressure']
        
        features_df = extractor.extract_from_wide(
            df=df,
            id_column='batch_id',
            value_columns=value_cols,
            timestamp_column='timestamp',
        )
        
        print(f"✅ Extracted {features_df.shape[1]} features")
        print(f"   Sample features: {list(features_df.columns[:5])}")
        
    except ImportError:
        print("⚠️  tsfresh not available, using basic feature extraction")
        features_df = extract_basic_features(df)
    
    # Add target variable
    target = df.groupby('batch_id')['failure'].max()
    features_df = features_df.join(target, how='left')
    
    # =========================================================================
    # STEP 3: Supervised Detection
    # =========================================================================
    print("\n🎯 STEP 3: Running supervised detection...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X = features_df.drop(columns=['failure']).fillna(0)
        y = features_df['failure']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        accuracy = rf.score(X_test, y_test)
        print(f"✅ RandomForest accuracy: {accuracy:.2%}")
        
        # Feature importance
        importance = pd.Series(rf.feature_importances_, index=X.columns)
        importance = importance.sort_values(ascending=False)
        
        print(f"   Top features:")
        for feat, imp in importance.head(5).items():
            print(f"     - {feat}: {imp:.3f}")
        
    except ImportError:
        print("⚠️  sklearn not available, skipping supervised detection")
        importance = None
    
    # =========================================================================
    # STEP 4: Association Rules Mining
    # =========================================================================
    print("\n🔗 STEP 4: Mining association rules with mlxtend...")
    
    try:
        from src.pattern_engine.detection import AssociationRuleDetector, AssociationRuleConfig
        
        # Aggregate to batch level for rule mining
        batch_df = df.groupby('batch_id').agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'extrusion_speed': 'mean',
            'pressure': 'mean',
            'failure': 'max',
        }).reset_index()
        
        detector = AssociationRuleDetector(AssociationRuleConfig(
            min_support=0.15,
            min_confidence=0.5,
            min_lift=1.0,
            algorithm='fpgrowth',
        ))
        
        rules = detector.fit_from_binned(
            df=batch_df,
            target_column='failure',
            bin_columns=['temperature', 'humidity', 'extrusion_speed'],
            n_bins=3,
        )
        
        print(f"✅ Found {len(rules)} association rules")
        
        # Filter rules predicting failure
        failure_rules = detector.get_rules_for_consequent('failure', top_k=5)
        print(f"   Rules predicting failure:")
        for rule in failure_rules[:3]:
            print(f"     {rule}")
        
    except ImportError:
        print("⚠️  mlxtend not available, skipping association rules")
        rules = []
    
    # =========================================================================
    # STEP 5: Physics Validation
    # =========================================================================
    print("\n⚗️  STEP 5: Applying physics validation...")
    
    from src.pattern_engine.physics import PhysicsValidator, create_manufacturing_validator
    
    validator = create_manufacturing_validator()
    print(f"   Available rules: {validator.available_rules()}")
    
    # Apply physics to batch data
    validated_df = validator.apply(batch_df)
    
    if 'physics_score_combined' in validated_df.columns:
        print(f"✅ Physics validation applied")
        print(f"   Mean physics score: {validated_df['physics_score_combined'].mean():.2f}")
        
        # Show physics by failure status
        failed_physics = validated_df[validated_df['failure'] == 1]['physics_score_combined'].mean()
        ok_physics = validated_df[validated_df['failure'] == 0]['physics_score_combined'].mean()
        print(f"   Failed batches physics: {failed_physics:.2f}")
        print(f"   OK batches physics: {ok_physics:.2f}")
    else:
        print("⚠️  No physics rules could be applied (check column names)")
    
    # =========================================================================
    # STEP 6: Gray Zone Analysis
    # =========================================================================
    print("\n🌫️  STEP 6: Gray zone analysis...")
    
    from src.pattern_engine.output import GrayZoneDetector, GrayZoneConfig
    from src.pattern_engine.core.pattern import Pattern, Condition, ConditionOperator, Outcome
    
    # Create sample patterns from association rules
    detected_patterns = []
    if rules:
        for i, rule in enumerate(rules[:5]):
            conditions = []
            for ant in rule.antecedents:
                # Parse "feature_bin" format
                parts = str(ant).rsplit('_', 1)
                if len(parts) == 2:
                    feat, level = parts
                    if level == 'high':
                        conditions.append(Condition(feat, ConditionOperator.GREATER_EQUAL, 25))
                    elif level == 'low':
                        conditions.append(Condition(feat, ConditionOperator.LESS_EQUAL, 22))
            
            if conditions:
                pattern = Pattern(
                    pattern_id=f"rule_{i}",
                    conditions=conditions,
                    outcome=Outcome("failure", 1),
                    confidence=rule.confidence,
                    support=rule.support,
                )
                detected_patterns.append(pattern)
    
    if detected_patterns:
        gray_detector = GrayZoneDetector(
            physics_validator=validator,
            config=GrayZoneConfig(
                statistical_threshold=0.4,
                physics_threshold=0.4,
            )
        )
        
        gray_patterns = gray_detector.analyze(detected_patterns, batch_df)
        
        summary = gray_detector.summary()
        print(f"✅ Gray zone analysis complete")
        print(f"   Patterns analyzed: {summary['total_analyzed']}")
        print(f"   Auto-approved: {summary['approved']}")
        print(f"   Gray zone (needs review): {summary['gray_zone']}")
        print(f"   Auto-rejected: {summary['rejected']}")
        
        if gray_patterns:
            print("\n   ⚠️  Patterns needing review:")
            for gp in gray_patterns[:2]:
                print(f"     - {gp.reason[:80]}...")
    else:
        print("⚠️  No patterns to analyze for gray zone")
    
    # =========================================================================
    # STEP 7: Summary Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 PIPELINE SUMMARY")
    print("=" * 70)
    
    print(f"""
Data:
  - {df['batch_id'].nunique()} batches analyzed
  - {features_df.shape[1] - 1} features extracted
  - Failure rate: {(df.groupby('batch_id')['failure'].max().mean() * 100):.1f}%

Detection:
  - Supervised: {'✅ RandomForest trained' if importance is not None else '❌ Not available'}
  - Association Rules: {f'✅ {len(rules)} rules found' if rules else '❌ Not available'}

Validation:
  - Physics rules applied: {len(validator.available_rules())}
  - Gray zone patterns: {len(gray_patterns) if 'gray_patterns' in dir() else 0}

Key Findings:
""")
    
    if importance is not None:
        print("  Top predictive features:")
        for feat, imp in importance.head(3).items():
            print(f"    - {feat}: {imp:.3f}")
    
    if rules:
        print("\n  Top association rules:")
        for rule in rules[:2]:
            print(f"    - {rule}")
    
    print("\n" + "=" * 70)
    print("✅ Pipeline complete!")
    print("=" * 70)


def create_sample_data():
    """Create sample manufacturing data."""
    np.random.seed(42)
    
    data = []
    for batch_id in range(1, 21):
        # Determine if this batch fails (based on temperature and speed)
        base_temp = np.random.uniform(21, 30)
        base_speed = np.random.uniform(100, 150)
        fails = (base_temp > 26 and base_speed > 120) or np.random.random() < 0.1
        
        for hour in range(4):
            ts = pd.Timestamp(f'2024-01-{batch_id:02d} {8+hour}:00:00')
            data.append({
                'batch_id': batch_id,
                'timestamp': ts,
                'temperature': base_temp + np.random.normal(0, 0.5),
                'humidity': np.random.uniform(50, 75),
                'extrusion_speed': base_speed + np.random.normal(0, 5),
                'pressure': np.random.uniform(1.1, 1.5),
                'failure': int(fails),
            })
    
    return pd.DataFrame(data)


def extract_basic_features(df):
    """Basic feature extraction fallback."""
    agg_funcs = {
        'temperature': ['mean', 'std', 'min', 'max'],
        'humidity': ['mean', 'std', 'min', 'max'],
        'extrusion_speed': ['mean', 'std', 'min', 'max'],
        'pressure': ['mean', 'std', 'min', 'max'],
    }
    
    features = df.groupby('batch_id').agg(agg_funcs)
    features.columns = ['_'.join(col) for col in features.columns]
    return features


if __name__ == "__main__":
    main()
