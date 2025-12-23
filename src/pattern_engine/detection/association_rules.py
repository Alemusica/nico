"""
Association Rules Detection with mlxtend
========================================

Extract association rules from categorical/binned features.
Uses Apriori and FP-Growth algorithms.

Example rules:
- IF temperature >= 25 AND humidity > 60 THEN failure
- IF wind_speed > 15 AND direction = 'NW' THEN surge

Reference: http://rasbt.github.io/mlxtend/
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import mlxtend
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    logger.warning("mlxtend not installed. Install with: pip install mlxtend")


@dataclass
class AssociationRuleConfig:
    """
    Configuration for association rule mining.
    
    Attributes:
        algorithm: 'apriori' or 'fpgrowth' (fpgrowth is faster)
        min_support: Minimum support threshold (0-1)
        min_confidence: Minimum confidence threshold (0-1)
        min_lift: Minimum lift (>1 means positive association)
        max_antecedent_len: Max number of conditions in IF part
        metric: Metric for rule selection ('confidence', 'lift', 'leverage')
        use_colnames: Use column names instead of indices
    """
    algorithm: str = "fpgrowth"  # fpgrowth is faster
    min_support: float = 0.1  # At least 10% of transactions
    min_confidence: float = 0.6  # At least 60% confidence
    min_lift: float = 1.0  # Positive association
    max_antecedent_len: Optional[int] = 3
    metric: str = "lift"
    metric_threshold: float = 1.0
    use_colnames: bool = True


@dataclass
class AssociationRule:
    """A single association rule."""
    antecedents: frozenset
    consequents: frozenset
    support: float
    confidence: float
    lift: float
    leverage: float = 0.0
    conviction: float = 0.0
    
    def __str__(self) -> str:
        ant = " AND ".join(sorted(self.antecedents))
        cons = " AND ".join(sorted(self.consequents))
        return f"IF {ant} THEN {cons} [support={self.support:.3f}, conf={self.confidence:.3f}, lift={self.lift:.2f}]"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "antecedents": list(self.antecedents),
            "consequents": list(self.consequents),
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
            "leverage": self.leverage,
            "conviction": self.conviction,
        }


class AssociationRuleDetector:
    """
    Detect association rules using mlxtend.
    
    Example:
        detector = AssociationRuleDetector(AssociationRuleConfig(
            min_support=0.1,
            min_confidence=0.7,
            min_lift=1.2
        ))
        
        # From boolean matrix
        rules = detector.fit_from_boolean(boolean_df)
        
        # From binned continuous features
        rules = detector.fit_from_binned(
            df=data_df,
            target_column='failure',
            bin_columns=['temperature', 'humidity'],
            n_bins=3
        )
    """
    
    def __init__(self, config: Optional[AssociationRuleConfig] = None):
        self.config = config or AssociationRuleConfig()
        
        if not MLXTEND_AVAILABLE:
            raise ImportError(
                "mlxtend is not installed. Install with: pip install mlxtend>=0.23.1"
            )
        
        self._frequent_itemsets: Optional[pd.DataFrame] = None
        self._rules_df: Optional[pd.DataFrame] = None
        self._rules: List[AssociationRule] = []
    
    def fit_from_boolean(
        self,
        boolean_df: pd.DataFrame,
    ) -> List[AssociationRule]:
        """
        Mine association rules from boolean (one-hot encoded) DataFrame.
        
        Args:
            boolean_df: DataFrame with boolean columns (True/False)
            
        Returns:
            List of AssociationRule objects
        """
        if boolean_df.empty:
            logger.warning("Empty DataFrame provided")
            return []
        
        # Ensure boolean types
        df = boolean_df.astype(bool)
        
        logger.info(f"Mining rules from {df.shape[0]} rows, {df.shape[1]} features")
        
        # Find frequent itemsets
        algorithm = self.config.algorithm
        if algorithm == "apriori":
            self._frequent_itemsets = apriori(
                df,
                min_support=self.config.min_support,
                use_colnames=self.config.use_colnames,
                max_len=self.config.max_antecedent_len + 1 if self.config.max_antecedent_len else None,
            )
        else:  # fpgrowth
            self._frequent_itemsets = fpgrowth(
                df,
                min_support=self.config.min_support,
                use_colnames=self.config.use_colnames,
                max_len=self.config.max_antecedent_len + 1 if self.config.max_antecedent_len else None,
            )
        
        if self._frequent_itemsets.empty:
            logger.warning(f"No frequent itemsets found with min_support={self.config.min_support}")
            return []
        
        logger.info(f"Found {len(self._frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        self._rules_df = association_rules(
            self._frequent_itemsets,
            metric=self.config.metric,
            min_threshold=self.config.metric_threshold,
        )
        
        # Filter by confidence and lift
        self._rules_df = self._rules_df[
            (self._rules_df['confidence'] >= self.config.min_confidence) &
            (self._rules_df['lift'] >= self.config.min_lift)
        ]
        
        # Filter by antecedent length
        if self.config.max_antecedent_len:
            self._rules_df = self._rules_df[
                self._rules_df['antecedents'].apply(len) <= self.config.max_antecedent_len
            ]
        
        # Convert to AssociationRule objects
        self._rules = []
        for _, row in self._rules_df.iterrows():
            rule = AssociationRule(
                antecedents=row['antecedents'],
                consequents=row['consequents'],
                support=row['support'],
                confidence=row['confidence'],
                lift=row['lift'],
                leverage=row.get('leverage', 0.0),
                conviction=row.get('conviction', 0.0) if not np.isinf(row.get('conviction', 0.0)) else 999.0,
            )
            self._rules.append(rule)
        
        logger.info(f"Found {len(self._rules)} rules")
        return self._rules
    
    def fit_from_binned(
        self,
        df: pd.DataFrame,
        target_column: str,
        bin_columns: Optional[List[str]] = None,
        n_bins: int = 3,
        bin_labels: Optional[Dict[str, List[str]]] = None,
    ) -> List[AssociationRule]:
        """
        Mine rules from continuous features by binning them first.
        
        Args:
            df: DataFrame with continuous features
            target_column: Column to predict (will be included in rules)
            bin_columns: Columns to bin (default: all numeric except target)
            n_bins: Number of bins per column
            bin_labels: Custom labels, e.g., {'temperature': ['low', 'medium', 'high']}
            
        Returns:
            List of AssociationRule objects
        """
        if df.empty:
            return []
        
        result = df.copy()
        
        # Get columns to bin
        if bin_columns is None:
            bin_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != target_column
            ]
        
        # Bin continuous columns
        binned_columns = []
        for col in bin_columns:
            if col not in result.columns:
                continue
            
            # Custom labels or default
            if bin_labels and col in bin_labels:
                labels = bin_labels[col]
            else:
                labels = [f"{col}_low", f"{col}_med", f"{col}_high"][:n_bins]
            
            try:
                binned = pd.cut(
                    result[col],
                    bins=n_bins,
                    labels=labels,
                    include_lowest=True
                )
                result[f"{col}_binned"] = binned
                binned_columns.append(f"{col}_binned")
            except Exception as e:
                logger.warning(f"Could not bin {col}: {e}")
        
        # Bin target if numeric
        if target_column in result.columns:
            if result[target_column].dtype in [np.number, 'float64', 'int64']:
                # Binary classification: assume > 0 is positive
                if result[target_column].nunique() <= 2:
                    result[f"{target_column}_yes"] = result[target_column] > 0
                    binned_columns.append(f"{target_column}_yes")
                else:
                    binned = pd.cut(
                        result[target_column],
                        bins=n_bins,
                        labels=[f"{target_column}_low", f"{target_column}_med", f"{target_column}_high"][:n_bins],
                        include_lowest=True
                    )
                    result[f"{target_column}_binned"] = binned
                    binned_columns.append(f"{target_column}_binned")
            else:
                binned_columns.append(target_column)
        
        # One-hot encode binned columns
        boolean_df = pd.get_dummies(result[binned_columns], prefix='', prefix_sep='')
        
        # Convert to boolean
        boolean_df = boolean_df.astype(bool)
        
        return self.fit_from_boolean(boolean_df)
    
    def fit_from_transactions(
        self,
        transactions: List[List[str]],
    ) -> List[AssociationRule]:
        """
        Mine rules from transaction data (market basket format).
        
        Args:
            transactions: List of transactions, each transaction is a list of items
            
        Returns:
            List of AssociationRule objects
        """
        # Encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        
        return self.fit_from_boolean(df)
    
    def get_rules_for_consequent(
        self,
        consequent: str,
        top_k: Optional[int] = None,
    ) -> List[AssociationRule]:
        """
        Get rules that predict a specific consequent.
        
        Args:
            consequent: The outcome to predict
            top_k: Return only top K rules by lift
            
        Returns:
            Filtered rules
        """
        matching = [
            r for r in self._rules
            if consequent in r.consequents or any(consequent in str(c) for c in r.consequents)
        ]
        
        # Sort by lift
        matching.sort(key=lambda r: r.lift, reverse=True)
        
        if top_k:
            matching = matching[:top_k]
        
        return matching
    
    def rules_to_dataframe(self) -> pd.DataFrame:
        """Convert rules to DataFrame."""
        if not self._rules:
            return pd.DataFrame()
        
        return pd.DataFrame([r.to_dict() for r in self._rules])
    
    def print_rules(self, top_k: int = 10):
        """Print top rules."""
        sorted_rules = sorted(self._rules, key=lambda r: r.lift, reverse=True)
        print(f"\n=== Top {min(top_k, len(sorted_rules))} Association Rules ===\n")
        for i, rule in enumerate(sorted_rules[:top_k], 1):
            print(f"{i}. {rule}")


def quick_association_rules(
    df: pd.DataFrame,
    target_column: str,
    min_support: float = 0.1,
    min_confidence: float = 0.6,
    min_lift: float = 1.0,
    n_bins: int = 3,
) -> List[AssociationRule]:
    """
    Quick helper to find association rules from a DataFrame.
    
    Example:
        rules = quick_association_rules(
            batch_data,
            target_column='failure',
            min_confidence=0.7
        )
        for rule in rules[:5]:
            print(rule)
    """
    if not MLXTEND_AVAILABLE:
        raise ImportError("mlxtend not installed")
    
    detector = AssociationRuleDetector(AssociationRuleConfig(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
    ))
    
    return detector.fit_from_binned(df, target_column, n_bins=n_bins)
