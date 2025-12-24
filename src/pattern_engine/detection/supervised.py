"""
Supervised pattern detection.

Uses labeled outcomes to discover conditions that lead to specific results.
E.g., "When temperature >= 25°C and speed > 150, failure probability = 0.78"
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import LabelEncoder

from .base import BaseDetector, DetectionResult
from ..core.config import SupervisedConfig, DetectionMethod
from ..core.pattern import (
    Pattern, PatternType, PatternDatabase,
    Condition, ConditionOperator, Outcome
)

logger = logging.getLogger(__name__)


class SupervisedDetector(BaseDetector):
    """
    Supervised pattern detection using labeled data.
    
    Discovers rules/patterns that predict outcomes based on input conditions.
    
    Example:
        detector = SupervisedDetector(SupervisedConfig(
            methods=[DetectionMethod.RANDOM_FOREST, DetectionMethod.XGBOOST]
        ))
        
        result = detector.fit_detect(X_train, y_train)
        
        # Get discovered patterns
        for pattern in result.patterns:
            print(pattern)
            # [causation] temperature >= 25.0 AND speed > 150 → failure=True (conf=0.78)
    """
    
    def __init__(
        self,
        config: Optional[SupervisedConfig] = None,
        random_seed: int = 42,
    ):
        super().__init__(config, random_seed)
        self.config = config or SupervisedConfig()
        self._models: Dict[str, Any] = {}
        self._label_encoder: Optional[LabelEncoder] = None
        self._best_model_name: Optional[str] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "SupervisedDetector":
        """
        Fit supervised models to data.
        
        Args:
            X: Feature matrix
            y: Target variable (outcome to predict)
        """
        self._feature_names = list(X.columns)
        
        # Handle class imbalance
        X_train, y_train = self._handle_imbalance(X, y)
        
        # Encode target if categorical
        if y_train.dtype == "object" or isinstance(y_train.iloc[0], str):
            self._label_encoder = LabelEncoder()
            y_train = pd.Series(
                self._label_encoder.fit_transform(y_train),
                index=y_train.index
            )
        
        # Train models
        best_score = -1
        for method in self.config.methods:
            model = self._create_model(method)
            if model is None:
                continue
            
            try:
                model.fit(X_train, y_train)
                self._models[method.value] = model
                
                # Cross-validation score
                scores = cross_val_score(
                    model, X_train, y_train,
                    cv=min(self.config.cv_folds, len(y_train)),
                    scoring="f1_weighted"
                )
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    self._best_model_name = method.value
                
                logger.info(f"{method.value}: CV F1={avg_score:.3f} (+/- {scores.std():.3f})")
                
            except Exception as e:
                logger.warning(f"Failed to train {method.value}: {e}")
        
        self._fitted = True
        return self
    
    def detect(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> DetectionResult:
        """
        Detect patterns from fitted models.
        
        Extracts interpretable rules from tree-based models.
        """
        self._check_fitted()
        
        patterns = []
        feature_importance = {}
        
        # Get feature importance from best model
        best_model = self._models.get(self._best_model_name)
        if best_model and hasattr(best_model, "feature_importances_"):
            importance = best_model.feature_importances_
            feature_importance = dict(zip(self._feature_names, importance))
        
        # Extract rules from tree-based models
        for model_name, model in self._models.items():
            if hasattr(model, "estimators_"):  # Ensemble models
                rules = self._extract_rules_from_forest(model, X, y)
                patterns.extend(rules)
            elif hasattr(model, "tree_"):  # Single tree
                rules = self._extract_rules_from_tree(model, X, y)
                patterns.extend(rules)
        
        # If no tree-based patterns, create threshold-based patterns
        if not patterns:
            patterns = self._create_threshold_patterns(X, y, feature_importance)
        
        # Calculate metrics
        y_encoded = y
        if self._label_encoder:
            y_encoded = pd.Series(self._label_encoder.transform(y), index=y.index)
        
        y_pred = self.predict(X)
        metrics = self._calculate_metrics(y_encoded, y_pred)
        
        # Create database
        database = PatternDatabase(name="supervised_patterns")
        for pattern in patterns:
            database.add_pattern(pattern)
        
        return DetectionResult(
            patterns=patterns,
            database=database,
            model=best_model,
            feature_importance=feature_importance,
            metrics=metrics,
            metadata={
                "best_model": self._best_model_name,
                "n_features": len(self._feature_names),
                "n_samples": len(X),
            }
        )
    
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Predict outcomes for new data."""
        self._check_fitted()
        
        model = self._models.get(self._best_model_name)
        if model is None:
            raise RuntimeError("No trained model available")
        
        predictions = model.predict(X)
        
        if self._label_encoder:
            predictions = self._label_encoder.inverse_transform(predictions)
        
        return pd.Series(predictions, index=X.index)
    
    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict outcome probabilities."""
        self._check_fitted()
        
        model = self._models.get(self._best_model_name)
        if model is None or not hasattr(model, "predict_proba"):
            raise RuntimeError("Model does not support probability prediction")
        
        proba = model.predict_proba(X)
        
        if self._label_encoder:
            columns = self._label_encoder.classes_
        else:
            columns = [f"class_{i}" for i in range(proba.shape[1])]
        
        return pd.DataFrame(proba, index=X.index, columns=columns)
    
    def _create_model(self, method: DetectionMethod) -> Optional[Any]:
        """Create a model instance for the given method."""
        params = self.config.hyperparameters.get(method.value, {})
        params["random_state"] = self.random_seed
        
        if method == DetectionMethod.DECISION_TREE:
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(**params)
        
        elif method == DetectionMethod.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier
            params.setdefault("n_estimators", 100)
            params.setdefault("n_jobs", -1)
            return RandomForestClassifier(**params)
        
        elif method == DetectionMethod.XGBOOST:
            try:
                from xgboost import XGBClassifier
                params.setdefault("n_estimators", 100)
                params.setdefault("use_label_encoder", False)
                params.setdefault("eval_metric", "logloss")
                return XGBClassifier(**params)
            except ImportError:
                logger.warning("XGBoost not installed, skipping")
                return None
        
        elif method == DetectionMethod.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression
            params.setdefault("max_iter", 1000)
            return LogisticRegression(**params)
        
        elif method == DetectionMethod.SVM:
            from sklearn.svm import SVC
            params.setdefault("probability", True)
            return SVC(**params)
        
        elif method == DetectionMethod.NEURAL_NETWORK:
            from sklearn.neural_network import MLPClassifier
            params.setdefault("hidden_layer_sizes", (100, 50))
            params.setdefault("max_iter", 500)
            return MLPClassifier(**params)
        
        return None
    
    def _handle_imbalance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance in training data."""
        if not self.config.handle_imbalance:
            return X, y
        
        method = self.config.imbalance_method
        
        if method == "smote":
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_seed)
                X_res, y_res = smote.fit_resample(X, y)
                return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
            except ImportError:
                logger.warning("imbalanced-learn not installed, skipping SMOTE")
        
        elif method == "undersample":
            try:
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(random_state=self.random_seed)
                X_res, y_res = rus.fit_resample(X, y)
                return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
            except ImportError:
                logger.warning("imbalanced-learn not installed, skipping undersampling")
        
        return X, y
    
    def _extract_rules_from_forest(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Pattern]:
        """Extract rules from random forest or ensemble models."""
        patterns = []
        
        # Use top trees by feature importance
        n_trees = min(5, len(model.estimators_))
        
        for tree in model.estimators_[:n_trees]:
            tree_patterns = self._extract_rules_from_tree(tree, X, y)
            patterns.extend(tree_patterns)
        
        # Deduplicate and aggregate patterns
        return self._aggregate_patterns(patterns)
    
    def _extract_rules_from_tree(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> List[Pattern]:
        """Extract rules from a decision tree."""
        patterns = []
        
        tree = model.tree_ if hasattr(model, "tree_") else model
        feature_names = self._feature_names
        
        def traverse_tree(node_id: int, conditions: List[Condition], depth: int = 0):
            if depth > 5:  # Limit depth to keep rules interpretable
                return
            
            # Leaf node
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Get class distribution at leaf
                values = tree.value[node_id].flatten()
                total = values.sum()
                if total == 0:
                    return
                
                predicted_class = values.argmax()
                confidence = values[predicted_class] / total
                support = total / len(X)
                
                # Only keep high-confidence patterns
                if confidence >= 0.6 and len(conditions) > 0:
                    outcome_value = predicted_class
                    if self._label_encoder:
                        outcome_value = self._label_encoder.inverse_transform([predicted_class])[0]
                    
                    pattern = Pattern(
                        pattern_type=PatternType.CAUSATION,
                        conditions=conditions.copy(),
                        outcome=Outcome(
                            name=y.name or "target",
                            value=outcome_value,
                            probability=confidence,
                        ),
                        confidence=confidence,
                        support=support,
                        sample_size=int(total),
                    )
                    patterns.append(pattern)
                return
            
            # Internal node - traverse children
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]
            
            # Left child (<=)
            left_conditions = conditions + [
                Condition(feature_name, ConditionOperator.LESS_EQUAL, round(threshold, 3))
            ]
            traverse_tree(tree.children_left[node_id], left_conditions, depth + 1)
            
            # Right child (>)
            right_conditions = conditions + [
                Condition(feature_name, ConditionOperator.GREATER, round(threshold, 3))
            ]
            traverse_tree(tree.children_right[node_id], right_conditions, depth + 1)
        
        traverse_tree(0, [])
        return patterns
    
    def _create_threshold_patterns(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_importance: Dict[str, float],
    ) -> List[Pattern]:
        """Create patterns based on feature thresholds for non-tree models."""
        patterns = []
        
        # Get top important features
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # For each class
        classes = y.unique()
        
        for class_val in classes:
            class_mask = y == class_val
            class_data = X[class_mask]
            other_data = X[~class_mask]
            
            conditions = []
            
            for feature, importance in top_features:
                if importance < self.config.min_feature_importance:
                    continue
                
                # Find threshold that separates classes
                class_mean = class_data[feature].mean()
                other_mean = other_data[feature].mean()
                
                if class_mean > other_mean:
                    threshold = (class_mean + other_mean) / 2
                    conditions.append(
                        Condition(feature, ConditionOperator.GREATER, round(threshold, 3))
                    )
                else:
                    threshold = (class_mean + other_mean) / 2
                    conditions.append(
                        Condition(feature, ConditionOperator.LESS_EQUAL, round(threshold, 3))
                    )
            
            if conditions:
                # Calculate confidence
                confidence = class_mask.sum() / len(y)
                
                pattern = Pattern(
                    pattern_type=PatternType.CORRELATION,
                    conditions=conditions[:3],  # Limit conditions
                    outcome=Outcome(
                        name=y.name or "target",
                        value=class_val,
                        probability=confidence,
                    ),
                    confidence=confidence,
                    support=class_mask.sum() / len(y),
                    feature_importance={f: importance for f, importance in top_features[:3]},
                )
                patterns.append(pattern)
        
        return patterns
    
    def _aggregate_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        """Aggregate similar patterns and keep best ones."""
        # Simple deduplication by pattern ID
        unique_patterns = {}
        for pattern in patterns:
            pid = pattern.pattern_id
            if pid not in unique_patterns or pattern.confidence > unique_patterns[pid].confidence:
                unique_patterns[pid] = pattern
        
        # Sort by confidence and return top patterns
        return sorted(
            unique_patterns.values(),
            key=lambda p: (p.confidence, p.support),
            reverse=True
        )[:20]  # Keep top 20 patterns
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        # Encode predictions if needed
        if self._label_encoder and y_pred.dtype == "object":
            y_pred = pd.Series(self._label_encoder.transform(y_pred))
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        # AUC for binary classification
        if len(y_true.unique()) == 2:
            try:
                model = self._models.get(self._best_model_name)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(self._last_X)[:, 1]
                    metrics["auc"] = roc_auc_score(y_true, y_proba)
            except Exception:
                pass
        
        return metrics
