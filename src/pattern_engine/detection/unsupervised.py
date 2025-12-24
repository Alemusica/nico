"""
Unsupervised pattern detection.

Discovers patterns without labeled outcomes:
- Anomaly detection
- Clustering
- Association rules
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

from .base import BaseDetector, DetectionResult
from ..core.config import UnsupervisedConfig, DetectionMethod
from ..core.pattern import (
    Pattern, PatternType, PatternDatabase,
    Condition, ConditionOperator, Outcome
)

logger = logging.getLogger(__name__)


class UnsupervisedDetector(BaseDetector):
    """
    Unsupervised pattern detection for anomaly and cluster discovery.
    
    Finds patterns without requiring labeled outcomes.
    
    Example:
        detector = UnsupervisedDetector(UnsupervisedConfig(
            methods=[DetectionMethod.ISOLATION_FOREST, DetectionMethod.KMEANS],
            contamination=0.1
        ))
        
        result = detector.fit_detect(X)
        
        # Get anomalies
        anomalies = result.patterns[result.patterns.pattern_type == PatternType.ANOMALY]
    """
    
    def __init__(
        self,
        config: Optional[UnsupervisedConfig] = None,
        random_seed: int = 42,
    ):
        super().__init__(config, random_seed)
        self.config = config or UnsupervisedConfig()
        self._models: Dict[str, Any] = {}
        self._cluster_centers: Optional[np.ndarray] = None
        self._anomaly_scores: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "UnsupervisedDetector":
        """
        Fit unsupervised models to data.
        
        Args:
            X: Feature matrix
            y: Ignored (for API compatibility)
        """
        self._feature_names = list(X.columns)
        
        # Reduce dimensions if configured
        if self.config.reduce_dimensions and len(self._feature_names) > self.config.n_components:
            X_reduced, self._reducer = self._reduce_dimensions(X)
        else:
            X_reduced = X
            self._reducer = None
        
        # Train models
        for method in self.config.methods:
            model = self._create_model(method, X_reduced)
            if model is None:
                continue
            
            try:
                model.fit(X_reduced)
                self._models[method.value] = model
                logger.info(f"Fitted {method.value}")
            except Exception as e:
                logger.warning(f"Failed to train {method.value}: {e}")
        
        self._fitted = True
        self._X_fitted = X
        return self
    
    def detect(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> DetectionResult:
        """
        Detect patterns (anomalies, clusters) from fitted models.
        """
        self._check_fitted()
        
        patterns = []
        feature_importance = {}
        
        # Reduce dimensions if fitted with reduction
        if self._reducer:
            X_reduced = pd.DataFrame(
                self._reducer.transform(X),
                index=X.index
            )
        else:
            X_reduced = X
        
        # Detect anomalies
        anomaly_patterns = self._detect_anomalies(X, X_reduced)
        patterns.extend(anomaly_patterns)
        
        # Detect clusters
        cluster_patterns = self._detect_clusters(X, X_reduced)
        patterns.extend(cluster_patterns)
        
        # Association rules
        if DetectionMethod.ASSOCIATION_RULES in self.config.methods:
            assoc_patterns = self._detect_associations(X)
            patterns.extend(assoc_patterns)
        
        # Calculate feature importance from anomaly detection
        if "isolation_forest" in self._models:
            feature_importance = self._calculate_anomaly_importance(X)
        
        # Create database
        database = PatternDatabase(name="unsupervised_patterns")
        for pattern in patterns:
            database.add_pattern(pattern)
        
        return DetectionResult(
            patterns=patterns,
            database=database,
            model=self._models,
            feature_importance=feature_importance,
            metrics={
                "n_anomalies": len([p for p in patterns if p.pattern_type == PatternType.ANOMALY]),
                "n_clusters": len([p for p in patterns if p.pattern_type == PatternType.CLUSTER]),
                "n_associations": len([p for p in patterns if p.pattern_type == PatternType.ASSOCIATION]),
            },
            metadata={
                "n_features": len(self._feature_names),
                "n_samples": len(X),
            }
        )
    
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict cluster assignments or anomaly labels for new data.
        """
        self._check_fitted()
        
        results = {}
        
        # Reduce dimensions if needed
        if self._reducer:
            X_reduced = pd.DataFrame(
                self._reducer.transform(X),
                index=X.index
            )
        else:
            X_reduced = X
        
        # Anomaly predictions
        if "isolation_forest" in self._models:
            model = self._models["isolation_forest"]
            results["is_anomaly"] = model.predict(X_reduced) == -1
            results["anomaly_score"] = -model.decision_function(X_reduced)
        
        if "lof" in self._models:
            model = self._models["lof"]
            results["lof_score"] = -model.score_samples(X_reduced)
        
        # Cluster predictions
        if "kmeans" in self._models:
            model = self._models["kmeans"]
            results["cluster"] = model.predict(X_reduced)
            results["cluster_distance"] = model.transform(X_reduced).min(axis=1)
        
        if "dbscan" in self._models:
            # DBSCAN doesn't have predict, use nearest cluster
            model = self._models["dbscan"]
            # Approximate by fitting on combined data
            results["dbscan_label"] = model.fit_predict(X_reduced)
        
        return pd.DataFrame(results, index=X.index)
    
    def _create_model(self, method: DetectionMethod, X: pd.DataFrame) -> Optional[Any]:
        """Create a model instance for the given method."""
        if method == DetectionMethod.ISOLATION_FOREST:
            from sklearn.ensemble import IsolationForest
            return IsolationForest(
                contamination=self.config.contamination,
                random_state=self.random_seed,
                n_jobs=-1,
            )
        
        elif method == DetectionMethod.LOCAL_OUTLIER_FACTOR:
            from sklearn.neighbors import LocalOutlierFactor
            return LocalOutlierFactor(
                contamination=self.config.contamination,
                novelty=True,  # Enable predict
                n_jobs=-1,
            )
        
        elif method == DetectionMethod.KMEANS:
            from sklearn.cluster import KMeans
            n_clusters = self.config.n_clusters
            if n_clusters is None:
                n_clusters = self._estimate_n_clusters(X)
            return KMeans(
                n_clusters=n_clusters,
                random_state=self.random_seed,
                n_init=10,
            )
        
        elif method == DetectionMethod.DBSCAN:
            from sklearn.cluster import DBSCAN
            return DBSCAN(
                min_samples=self.config.min_cluster_size,
            )
        
        elif method == DetectionMethod.PCA:
            from sklearn.decomposition import PCA
            return PCA(
                n_components=self.config.n_components,
                random_state=self.random_seed,
            )
        
        elif method == DetectionMethod.UMAP:
            try:
                from umap import UMAP
                return UMAP(
                    n_components=self.config.n_components,
                    random_state=self.random_seed,
                )
            except ImportError:
                logger.warning("UMAP not installed, skipping")
                return None
        
        return None
    
    def _reduce_dimensions(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """Reduce dimensionality of features."""
        from sklearn.decomposition import PCA
        
        reducer = PCA(
            n_components=min(self.config.n_components, len(X.columns), len(X)),
            random_state=self.random_seed,
        )
        
        X_reduced = pd.DataFrame(
            reducer.fit_transform(X),
            index=X.index,
            columns=[f"PC{i+1}" for i in range(reducer.n_components_)],
        )
        
        return X_reduced, reducer
    
    def _estimate_n_clusters(self, X: pd.DataFrame) -> int:
        """Estimate optimal number of clusters using elbow method."""
        from sklearn.cluster import KMeans
        
        max_clusters = min(10, len(X) // 10)
        if max_clusters < 2:
            return 2
        
        inertias = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        diffs = np.diff(inertias)
        elbow = np.argmin(diffs) + 2  # +2 because we started at k=2
        
        return max(2, min(elbow, max_clusters))
    
    def _detect_anomalies(
        self,
        X: pd.DataFrame,
        X_reduced: pd.DataFrame,
    ) -> List[Pattern]:
        """Detect anomaly patterns."""
        patterns = []
        
        # Isolation Forest anomalies
        if "isolation_forest" in self._models:
            model = self._models["isolation_forest"]
            predictions = model.predict(X_reduced)
            scores = -model.decision_function(X_reduced)
            
            anomaly_mask = predictions == -1
            n_anomalies = anomaly_mask.sum()
            
            if n_anomalies > 0:
                # Characterize anomalies
                anomaly_data = X[anomaly_mask]
                normal_data = X[~anomaly_mask]
                
                # Find features that distinguish anomalies
                conditions = self._characterize_anomalies(anomaly_data, normal_data)
                
                pattern = Pattern(
                    pattern_type=PatternType.ANOMALY,
                    conditions=conditions,
                    outcome=Outcome(
                        name="anomaly",
                        value=True,
                        probability=n_anomalies / len(X),
                    ),
                    confidence=1.0 - self.config.contamination,
                    support=n_anomalies / len(X),
                    sample_size=n_anomalies,
                    description=f"Detected {n_anomalies} anomalies using Isolation Forest",
                    metadata={
                        "method": "isolation_forest",
                        "anomaly_indices": X.index[anomaly_mask].tolist(),
                        "mean_anomaly_score": float(scores[anomaly_mask].mean()),
                    }
                )
                patterns.append(pattern)
        
        # LOF anomalies
        if "lof" in self._models:
            model = self._models["lof"]
            predictions = model.predict(X_reduced)
            
            anomaly_mask = predictions == -1
            n_anomalies = anomaly_mask.sum()
            
            if n_anomalies > 0:
                pattern = Pattern(
                    pattern_type=PatternType.ANOMALY,
                    conditions=[],
                    outcome=Outcome(
                        name="anomaly_lof",
                        value=True,
                        probability=n_anomalies / len(X),
                    ),
                    confidence=1.0 - self.config.contamination,
                    support=n_anomalies / len(X),
                    sample_size=n_anomalies,
                    description=f"Detected {n_anomalies} anomalies using LOF",
                    metadata={
                        "method": "lof",
                        "anomaly_indices": X.index[anomaly_mask].tolist(),
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _characterize_anomalies(
        self,
        anomaly_data: pd.DataFrame,
        normal_data: pd.DataFrame,
    ) -> List[Condition]:
        """Find conditions that characterize anomalies."""
        conditions = []
        
        numeric_cols = anomaly_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Top 5 features
            anomaly_mean = anomaly_data[col].mean()
            normal_mean = normal_data[col].mean()
            normal_std = normal_data[col].std()
            
            if normal_std == 0:
                continue
            
            # Check if anomalies are significantly different
            z_score = abs(anomaly_mean - normal_mean) / normal_std
            
            if z_score > 2:  # Significant difference
                if anomaly_mean > normal_mean:
                    threshold = normal_mean + 2 * normal_std
                    conditions.append(
                        Condition(col, ConditionOperator.GREATER, round(threshold, 3))
                    )
                else:
                    threshold = normal_mean - 2 * normal_std
                    conditions.append(
                        Condition(col, ConditionOperator.LESS, round(threshold, 3))
                    )
        
        return conditions[:3]  # Limit to 3 conditions
    
    def _detect_clusters(
        self,
        X: pd.DataFrame,
        X_reduced: pd.DataFrame,
    ) -> List[Pattern]:
        """Detect cluster patterns."""
        patterns = []
        
        # KMeans clusters
        if "kmeans" in self._models:
            model = self._models["kmeans"]
            labels = model.predict(X_reduced)
            centers = model.cluster_centers_
            
            for cluster_id in range(model.n_clusters):
                cluster_mask = labels == cluster_id
                cluster_size = cluster_mask.sum()
                
                if cluster_size < self.config.min_cluster_size:
                    continue
                
                # Characterize cluster
                cluster_data = X[cluster_mask]
                conditions = self._characterize_cluster(cluster_data, X)
                
                pattern = Pattern(
                    pattern_type=PatternType.CLUSTER,
                    conditions=conditions,
                    outcome=Outcome(
                        name="cluster",
                        value=cluster_id,
                        probability=cluster_size / len(X),
                    ),
                    confidence=1.0,
                    support=cluster_size / len(X),
                    sample_size=cluster_size,
                    description=f"Cluster {cluster_id} with {cluster_size} samples",
                    metadata={
                        "method": "kmeans",
                        "cluster_id": cluster_id,
                        "cluster_indices": X.index[cluster_mask].tolist(),
                    }
                )
                patterns.append(pattern)
        
        # DBSCAN clusters
        if "dbscan" in self._models:
            model = self._models["dbscan"]
            labels = model.labels_
            
            unique_labels = set(labels) - {-1}  # Exclude noise
            
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                cluster_size = cluster_mask.sum()
                
                if cluster_size < self.config.min_cluster_size:
                    continue
                
                cluster_data = X[cluster_mask]
                conditions = self._characterize_cluster(cluster_data, X)
                
                pattern = Pattern(
                    pattern_type=PatternType.CLUSTER,
                    conditions=conditions,
                    outcome=Outcome(
                        name="cluster_dbscan",
                        value=int(cluster_id),
                        probability=cluster_size / len(X),
                    ),
                    confidence=1.0,
                    support=cluster_size / len(X),
                    sample_size=cluster_size,
                    description=f"DBSCAN Cluster {cluster_id} with {cluster_size} samples",
                    metadata={
                        "method": "dbscan",
                        "cluster_id": int(cluster_id),
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    def _characterize_cluster(
        self,
        cluster_data: pd.DataFrame,
        full_data: pd.DataFrame,
    ) -> List[Condition]:
        """Find conditions that characterize a cluster."""
        conditions = []
        
        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:
            cluster_mean = cluster_data[col].mean()
            cluster_std = cluster_data[col].std()
            full_mean = full_data[col].mean()
            full_std = full_data[col].std()
            
            if full_std == 0:
                continue
            
            # Check if cluster is significantly different
            z_score = abs(cluster_mean - full_mean) / full_std
            
            if z_score > 1:  # Notable difference
                # Create range condition
                low = cluster_mean - cluster_std
                high = cluster_mean + cluster_std
                conditions.append(
                    Condition(
                        col,
                        ConditionOperator.IN_RANGE,
                        (round(low, 3), round(high, 3))
                    )
                )
        
        return conditions[:3]
    
    def _detect_associations(self, X: pd.DataFrame) -> List[Pattern]:
        """Detect association rule patterns."""
        patterns = []
        
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            
            # Binarize data for association rules
            X_binary = X.copy()
            for col in X.columns:
                if X[col].dtype in [np.float64, np.int64]:
                    median = X[col].median()
                    X_binary[f"{col}_high"] = X[col] > median
                    X_binary[f"{col}_low"] = X[col] <= median
                    X_binary = X_binary.drop(columns=[col])
            
            # Find frequent itemsets
            frequent = apriori(
                X_binary.astype(bool),
                min_support=self.config.min_support,
                use_colnames=True,
            )
            
            if len(frequent) == 0:
                return patterns
            
            # Generate rules
            rules = association_rules(
                frequent,
                metric="confidence",
                min_threshold=self.config.min_confidence,
            )
            
            # Filter by lift
            rules = rules[rules["lift"] >= self.config.min_lift]
            
            # Convert to patterns
            for _, rule in rules.head(10).iterrows():
                antecedents = list(rule["antecedents"])
                consequents = list(rule["consequents"])
                
                conditions = []
                for item in antecedents:
                    if "_high" in item:
                        feature = item.replace("_high", "")
                        conditions.append(
                            Condition(feature, ConditionOperator.GREATER, "median")
                        )
                    elif "_low" in item:
                        feature = item.replace("_low", "")
                        conditions.append(
                            Condition(feature, ConditionOperator.LESS_EQUAL, "median")
                        )
                
                pattern = Pattern(
                    pattern_type=PatternType.ASSOCIATION,
                    conditions=conditions,
                    outcome=Outcome(
                        name="association",
                        value=consequents,
                        probability=rule["confidence"],
                    ),
                    confidence=rule["confidence"],
                    support=rule["support"],
                    lift=rule["lift"],
                    description=f"{antecedents} → {consequents}",
                    metadata={
                        "antecedents": antecedents,
                        "consequents": consequents,
                        "conviction": rule.get("conviction", None),
                    }
                )
                patterns.append(pattern)
                
        except ImportError:
            logger.warning("mlxtend not installed, skipping association rules")
        except Exception as e:
            logger.warning(f"Association rule mining failed: {e}")
        
        return patterns
    
    def _calculate_anomaly_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance for anomaly detection."""
        if "isolation_forest" not in self._models:
            return {}
        
        model = self._models["isolation_forest"]
        
        # Use permutation importance
        try:
            from sklearn.inspection import permutation_importance
            
            # Anomaly scores as target
            scores = -model.decision_function(X)
            
            result = permutation_importance(
                model, X, scores,
                n_repeats=5,
                random_state=self.random_seed,
                n_jobs=-1,
            )
            
            importance = dict(zip(self._feature_names, result.importances_mean))
            return importance
            
        except Exception:
            return {}
