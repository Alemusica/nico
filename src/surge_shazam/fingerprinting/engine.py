"""
🎵 Fingerprint Engine - MiniRocket-based Pattern Extraction
============================================================

Estrae fingerprint da serie temporali multivariate usando MiniRocket (SOTA 2024-2025).
Come Shazam riconosce canzoni da fingerprint audio, noi riconosciamo pattern precursori
di eventi estremi (alluvioni, storm surge) da fingerprint di serie temporali.

Key Features:
    - MiniRocket embedding (fixed-length vector)
    - Multi-source data support (ERA5, CMEMS, GPM, GRACE, etc.)
    - Uncertainty propagation
    - Cosine similarity for pattern matching
    - Physics-aware variable weighting

Data Sources Supportate:
    - ERA5: pressure, temp, wind, precip, soil moisture
    - CMEMS: sea level, SST, currents
    - GPM: precipitation near-RT
    - GRACE: terrestrial water storage
    - CYGNSS: wind speed
    - Sentinel: SAR wind, ocean color
    - Aircraft: AMDAR, Mode-S (upper air)
    - Tide Gauges: ground truth sea level
    - Climate Indices: NAO, ENSO, AMO, PDO

Reference:
    - MiniRocket: arxiv:2408.02760, Dempster et al. (2021)
    - sktime: https://www.sktime.net

Usage:
    from src.surge_shazam.fingerprinting.engine import FingerprintEngine
    
    engine = FingerprintEngine()
    
    # Extract from harmonized DataFrame
    fp = engine.extract(df, "lago_maggiore_2000")
    
    # Compare fingerprints
    similarity = engine.similarity(fp1, fp2)
    
    # Batch extraction
    fingerprints = engine.batch_extract(events)

Requirements:
    pip install sktime numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
import logging
import hashlib
import warnings

logger = logging.getLogger(__name__)

# Check for sktime MiniRocket
try:
    from sktime.transformations.panel.rocket import MiniRocket
    HAS_MINIROCKET = True
except ImportError:
    HAS_MINIROCKET = False
    logger.warning("⚠️ sktime not installed. Install with: pip install sktime")

# Sklearn for dimensionality reduction and similarity
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Fingerprint:
    """
    Pattern fingerprint from multivariate time series.
    
    The embedding is a fixed-length vector that captures the essential
    characteristics of the input time series, enabling fast similarity search.
    """
    event_id: str
    embedding: np.ndarray  # Fixed-length vector (default: 100-dim)
    
    # Time range
    timestamp_start: datetime
    timestamp_end: datetime
    
    # Data provenance
    variables_used: List[str]
    sources_used: List[str]  # ['era5', 'cmems', 'gpm', 'grace', ...]
    
    # Spatial extent
    bbox: Optional[Tuple[float, float, float, float]] = None  # (lat_min, lat_max, lon_min, lon_max)
    center: Optional[Tuple[float, float]] = None  # (lat, lon)
    
    # Quality metrics
    n_samples: int = 0
    n_variables: int = 0
    missing_ratio: float = 0.0
    uncertainty: float = 0.0
    
    # Raw statistics (for interpretability)
    variable_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Metadata
    extraction_method: str = "minirocket"
    extraction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return (
            f"Fingerprint({self.event_id}, "
            f"dim={len(self.embedding)}, "
            f"vars={len(self.variables_used)}, "
            f"sources={self.sources_used})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "embedding": self.embedding.tolist(),
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat(),
            "variables_used": self.variables_used,
            "sources_used": self.sources_used,
            "bbox": self.bbox,
            "center": self.center,
            "n_samples": self.n_samples,
            "n_variables": self.n_variables,
            "missing_ratio": self.missing_ratio,
            "uncertainty": self.uncertainty,
            "variable_stats": self.variable_stats,
            "extraction_method": self.extraction_method,
            "extraction_timestamp": self.extraction_timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fingerprint":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            embedding=np.array(data["embedding"]),
            timestamp_start=datetime.fromisoformat(data["timestamp_start"]),
            timestamp_end=datetime.fromisoformat(data["timestamp_end"]),
            variables_used=data["variables_used"],
            sources_used=data["sources_used"],
            bbox=tuple(data["bbox"]) if data.get("bbox") else None,
            center=tuple(data["center"]) if data.get("center") else None,
            n_samples=data.get("n_samples", 0),
            n_variables=data.get("n_variables", 0),
            missing_ratio=data.get("missing_ratio", 0.0),
            uncertainty=data.get("uncertainty", 0.0),
            variable_stats=data.get("variable_stats", {}),
            extraction_method=data.get("extraction_method", "minirocket"),
            extraction_timestamp=data.get("extraction_timestamp", ""),
            metadata=data.get("metadata", {}),
        )
    
    def hash(self) -> str:
        """Generate hash from embedding for quick lookup."""
        return hashlib.md5(self.embedding.tobytes()).hexdigest()[:16]


@dataclass
class SimilarityResult:
    """Result of fingerprint similarity comparison."""
    query_id: str
    match_id: str
    similarity: float  # Cosine similarity [0, 1]
    distance: float  # Euclidean distance
    rank: int = 0
    
    # Breakdown by variable group (optional)
    group_similarities: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self):
        return f"SimilarityResult({self.query_id} ↔ {self.match_id}, sim={self.similarity:.3f})"


# =============================================================================
# PHYSICS-AWARE VARIABLE CONFIGURATION
# =============================================================================

# Variable importance weights for flood/surge prediction
# Based on literature: NAO, pressure, precip are strongest precursors
VARIABLE_WEIGHTS = {
    # Climate indices (strongest predictors)
    "nao": 1.5,
    "nao_index": 1.5,
    "enso": 1.3,
    "enso_index": 1.3,
    "amo": 1.2,
    "pdo": 1.1,
    
    # Atmospheric pressure (direct forcing)
    "msl": 1.4,
    "pressure": 1.4,
    "pressure_msl": 1.4,
    "slp": 1.4,
    
    # Precipitation (critical for floods)
    "tp": 1.5,
    "precipitation": 1.5,
    "total_precipitation": 1.5,
    "precip": 1.5,
    
    # Soil moisture (flood precursor)
    "swvl1": 1.3,
    "soil_moisture": 1.3,
    "sm": 1.3,
    "tws": 1.3,  # GRACE water storage
    
    # Sea level (direct indicator)
    "sla": 1.4,
    "ssh": 1.4,
    "sea_level": 1.4,
    "adt": 1.3,
    
    # Wind (storm surge driver)
    "u10": 1.2,
    "v10": 1.2,
    "wind_speed": 1.2,
    "ws": 1.2,
    
    # Temperature (indirect)
    "t2m": 0.9,
    "sst": 1.0,
    "temperature": 0.9,
    
    # Wave height
    "swh": 1.1,
    "wave_height": 1.1,
    
    # Default
    "default": 1.0,
}

# Variable groups for interpretability
VARIABLE_GROUPS = {
    "climate_indices": ["nao", "nao_index", "enso", "enso_index", "amo", "pdo", "ao", "pna"],
    "atmospheric": ["msl", "pressure", "pressure_msl", "slp", "t2m", "temperature", "d2m"],
    "precipitation": ["tp", "precipitation", "total_precipitation", "precip", "runoff", "ro"],
    "wind": ["u10", "v10", "wind_speed", "ws", "u_wind", "v_wind"],
    "ocean": ["sla", "ssh", "sea_level", "adt", "sst", "swh", "wave_height"],
    "hydrology": ["swvl1", "soil_moisture", "sm", "tws", "snow_depth", "sd"],
}


# =============================================================================
# FINGERPRINT ENGINE
# =============================================================================

class FingerprintEngine:
    """
    MiniRocket-based fingerprint extraction engine.
    
    Extracts fixed-length embeddings from multivariate time series,
    enabling fast pattern matching and similarity search.
    """
    
    def __init__(
        self,
        n_kernels: int = 10000,
        embedding_dim: int = 100,
        use_physics_weights: bool = True,
        normalize: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize fingerprint engine.
        
        Args:
            n_kernels: Number of MiniRocket kernels (more = more expressive)
            embedding_dim: Final embedding dimension after PCA (0 = no PCA)
            use_physics_weights: Weight variables by physics importance
            normalize: Standardize input data
            random_state: Random seed for reproducibility
        """
        self.n_kernels = n_kernels
        self.embedding_dim = embedding_dim
        self.use_physics_weights = use_physics_weights
        self.normalize = normalize
        self.random_state = random_state
        
        # Components (initialized on first use)
        self._rocket: Optional[Any] = None
        self._pca: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._fitted = False
        
        # Check dependencies
        if not HAS_MINIROCKET:
            logger.warning(
                "⚠️ MiniRocket not available. Using fallback statistical fingerprint. "
                "Install with: pip install sktime"
            )
    
    def extract(
        self,
        df: pd.DataFrame,
        event_id: str,
        sources: List[str] = None,
        bbox: Tuple[float, float, float, float] = None,
        center: Tuple[float, float] = None,
        metadata: Dict[str, Any] = None,
    ) -> Fingerprint:
        """
        Extract fingerprint from harmonized DataFrame.
        
        Args:
            df: DataFrame with time series (rows=time, cols=variables)
                Index should be DatetimeIndex
            event_id: Unique identifier for this event
            sources: List of data sources used (e.g., ['era5', 'cmems'])
            bbox: Spatial bounding box (lat_min, lat_max, lon_min, lon_max)
            center: Center point (lat, lon)
            metadata: Additional metadata
            
        Returns:
            Fingerprint object with embedding and metadata
        """
        # Validate input
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Store original info
        original_vars = list(df.columns)
        n_original_samples = len(df)
        
        # Handle missing data
        df_clean, missing_ratio = self._handle_missing(df)
        
        if df_clean.empty or len(df_clean) < 10:
            raise ValueError(f"Insufficient data after cleaning: {len(df_clean)} samples")
        
        # Compute variable statistics
        var_stats = self._compute_variable_stats(df_clean)
        
        # Apply physics weights if enabled
        if self.use_physics_weights:
            df_weighted = self._apply_physics_weights(df_clean)
        else:
            df_weighted = df_clean
        
        # Normalize
        if self.normalize:
            df_norm = self._normalize(df_weighted)
        else:
            df_norm = df_weighted
        
        # Extract embedding
        if HAS_MINIROCKET:
            embedding = self._extract_minirocket(df_norm)
        else:
            embedding = self._extract_fallback(df_norm)
        
        # Reduce dimension if needed
        if self.embedding_dim > 0 and len(embedding) > self.embedding_dim:
            embedding = self._reduce_dimension(embedding)
        
        # Compute uncertainty
        uncertainty = self._compute_uncertainty(df_clean, missing_ratio)
        
        # Get time range
        if isinstance(df.index, pd.DatetimeIndex):
            ts_start = df.index[0].to_pydatetime()
            ts_end = df.index[-1].to_pydatetime()
        else:
            ts_start = datetime.now()
            ts_end = datetime.now()
        
        return Fingerprint(
            event_id=event_id,
            embedding=embedding.astype(np.float32),
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            variables_used=original_vars,
            sources_used=sources or ["unknown"],
            bbox=bbox,
            center=center,
            n_samples=n_original_samples,
            n_variables=len(original_vars),
            missing_ratio=missing_ratio,
            uncertainty=uncertainty,
            variable_stats=var_stats,
            extraction_method="minirocket" if HAS_MINIROCKET else "statistical",
            metadata=metadata or {},
        )
    
    def similarity(self, fp1: Fingerprint, fp2: Fingerprint) -> float:
        """
        Compute cosine similarity between two fingerprints.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Similarity score in [0, 1]
        """
        # Ensure same dimension
        if len(fp1.embedding) != len(fp2.embedding):
            logger.warning(
                f"Embedding dimension mismatch: {len(fp1.embedding)} vs {len(fp2.embedding)}"
            )
            min_dim = min(len(fp1.embedding), len(fp2.embedding))
            e1 = fp1.embedding[:min_dim]
            e2 = fp2.embedding[:min_dim]
        else:
            e1 = fp1.embedding
            e2 = fp2.embedding
        
        # Cosine similarity
        dot = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        sim = dot / (norm1 * norm2)
        
        # Clip to [0, 1] (cosine can be negative)
        return float(np.clip((sim + 1) / 2, 0, 1))
    
    def compare(self, fp1: Fingerprint, fp2: Fingerprint) -> SimilarityResult:
        """
        Compare two fingerprints with detailed breakdown.
        
        Args:
            fp1: Query fingerprint
            fp2: Match fingerprint
            
        Returns:
            SimilarityResult with similarity score and breakdown
        """
        sim = self.similarity(fp1, fp2)
        
        # Euclidean distance
        dist = float(np.linalg.norm(fp1.embedding - fp2.embedding))
        
        # Group similarities (if we stored per-group embeddings)
        group_sims = {}
        
        return SimilarityResult(
            query_id=fp1.event_id,
            match_id=fp2.event_id,
            similarity=sim,
            distance=dist,
            group_similarities=group_sims,
        )
    
    def batch_extract(
        self,
        events: List[Dict[str, Any]],
        data_loader: callable = None,
    ) -> List[Fingerprint]:
        """
        Extract fingerprints for multiple events.
        
        Args:
            events: List of event dicts with keys:
                - event_id: str
                - df: pd.DataFrame (or use data_loader)
                - sources: List[str] (optional)
                - bbox: tuple (optional)
                - center: tuple (optional)
            data_loader: Optional callable(event) -> pd.DataFrame
            
        Returns:
            List of Fingerprint objects
        """
        fingerprints = []
        
        for event in events:
            try:
                # Get DataFrame
                if "df" in event:
                    df = event["df"]
                elif data_loader:
                    df = data_loader(event)
                else:
                    logger.warning(f"No data for event {event.get('event_id', 'unknown')}")
                    continue
                
                fp = self.extract(
                    df=df,
                    event_id=event["event_id"],
                    sources=event.get("sources"),
                    bbox=event.get("bbox"),
                    center=event.get("center"),
                    metadata=event.get("metadata"),
                )
                fingerprints.append(fp)
                
            except Exception as e:
                logger.error(f"Failed to extract fingerprint for {event.get('event_id')}: {e}")
        
        return fingerprints
    
    def find_similar(
        self,
        query: Fingerprint,
        candidates: List[Fingerprint],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SimilarityResult]:
        """
        Find most similar fingerprints to query.
        
        Args:
            query: Query fingerprint
            candidates: List of candidate fingerprints
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SimilarityResult, sorted by similarity (descending)
        """
        results = []
        
        for candidate in candidates:
            if candidate.event_id == query.event_id:
                continue  # Skip self
            
            result = self.compare(query, candidate)
            
            if result.similarity >= min_similarity:
                results.append(result)
        
        # Sort by similarity
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        # Assign ranks
        for i, r in enumerate(results[:top_k]):
            r.rank = i + 1
        
        return results[:top_k]
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _handle_missing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Handle missing data with forward/backward fill."""
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0.0
        
        # Fill missing values
        df_filled = df.ffill().bfill()
        
        # Drop remaining NaN columns
        df_clean = df_filled.dropna(axis=1, how='all')
        
        # Drop rows with any remaining NaN
        df_clean = df_clean.dropna(axis=0, how='any')
        
        return df_clean, missing_ratio
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data to zero mean, unit variance."""
        if not HAS_SKLEARN:
            # Simple standardization
            return (df - df.mean()) / (df.std() + 1e-10)
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df.values)
        return pd.DataFrame(data_scaled, index=df.index, columns=df.columns)
    
    def _apply_physics_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply physics-based weights to variables."""
        df_weighted = df.copy()
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Find matching weight
            weight = VARIABLE_WEIGHTS.get("default", 1.0)
            for key, w in VARIABLE_WEIGHTS.items():
                if key in col_lower:
                    weight = w
                    break
            
            df_weighted[col] = df[col] * weight
        
        return df_weighted
    
    def _compute_variable_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each variable."""
        stats = {}
        
        for col in df.columns:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "skew": float(df[col].skew()) if len(df) > 3 else 0.0,
            }
        
        return stats
    
    def _extract_minirocket(self, df: pd.DataFrame) -> np.ndarray:
        """Extract embedding using MiniRocket."""
        # MiniRocket expects 3D array: (n_instances, n_dims, n_timepoints)
        # We have one instance with multiple variables
        data = df.values.T[np.newaxis, :, :]  # (1, n_vars, n_time)
        
        # Initialize MiniRocket
        rocket = MiniRocket(
            num_kernels=self.n_kernels,
            random_state=self.random_state,
        )
        
        # Fit and transform
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rocket.fit(data)
            features = rocket.transform(data)
        
        # Flatten to 1D
        embedding = features.flatten()
        
        return embedding
    
    def _extract_fallback(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fallback statistical fingerprint when MiniRocket unavailable.
        
        Extracts statistical features from each variable:
        - Mean, std, min, max
        - Trend (linear regression slope)
        - Autocorrelation at lag 1
        - First difference statistics
        """
        features = []
        
        for col in df.columns:
            series = df[col].values
            
            # Basic statistics
            features.extend([
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                np.percentile(series, 25),
                np.percentile(series, 75),
            ])
            
            # Trend
            x = np.arange(len(series))
            if len(series) > 1:
                slope = np.polyfit(x, series, 1)[0]
            else:
                slope = 0.0
            features.append(slope)
            
            # Autocorrelation lag 1
            if len(series) > 1:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                if np.isnan(autocorr):
                    autocorr = 0.0
            else:
                autocorr = 0.0
            features.append(autocorr)
            
            # First difference statistics
            diff = np.diff(series)
            if len(diff) > 0:
                features.extend([
                    np.mean(diff),
                    np.std(diff),
                ])
            else:
                features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def _reduce_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce embedding dimension using PCA."""
        if not HAS_SKLEARN or self.embedding_dim <= 0:
            return embedding
        
        if len(embedding) <= self.embedding_dim:
            return embedding
        
        # Can't do PCA on single sample, use truncation
        # In practice, fit PCA on corpus and apply to new fingerprints
        return embedding[:self.embedding_dim]
    
    def _compute_uncertainty(
        self,
        df: pd.DataFrame,
        missing_ratio: float,
    ) -> float:
        """
        Compute uncertainty score based on data quality.
        
        Lower uncertainty = better data quality.
        """
        # Base uncertainty from missing data
        uncertainty = missing_ratio * 0.5
        
        # Add uncertainty from short time series
        n_samples = len(df)
        if n_samples < 30:
            uncertainty += 0.3 * (1 - n_samples / 30)
        
        # Add uncertainty from low variance (constant data)
        low_var_count = sum(1 for col in df.columns if df[col].std() < 1e-6)
        uncertainty += 0.2 * (low_var_count / len(df.columns))
        
        return float(np.clip(uncertainty, 0, 1))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def extract_fingerprint(
    df: pd.DataFrame,
    event_id: str,
    sources: List[str] = None,
    **kwargs,
) -> Fingerprint:
    """
    Quick fingerprint extraction.
    
    Args:
        df: DataFrame with time series
        event_id: Event identifier
        sources: Data sources
        **kwargs: Additional arguments for FingerprintEngine
        
    Returns:
        Fingerprint object
    """
    engine = FingerprintEngine(**kwargs)
    return engine.extract(df, event_id, sources=sources)


def compare_fingerprints(
    fp1: Fingerprint,
    fp2: Fingerprint,
) -> SimilarityResult:
    """Quick fingerprint comparison."""
    engine = FingerprintEngine()
    return engine.compare(fp1, fp2)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=== 🎵 Fingerprint Engine Test ===\n")
    
    # Check dependencies
    print(f"MiniRocket available: {HAS_MINIROCKET}")
    print(f"sklearn available: {HAS_SKLEARN}")
    
    # Generate synthetic data (simulating Lago Maggiore 2000)
    print("\n📊 Generating synthetic flood event data...")
    
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range("2000-10-01", periods=n_samples, freq="D")
    
    # Create synthetic variables
    df = pd.DataFrame({
        "precipitation": np.random.exponential(5, n_samples),
        "pressure_msl": 101325 - np.random.exponential(1000, n_samples),
        "temperature_2m": 288 + np.random.normal(0, 3, n_samples),
        "soil_moisture": np.random.uniform(0.2, 0.5, n_samples),
        "nao_index": np.random.normal(-0.5, 1, n_samples),  # Negative NAO
    }, index=dates)
    
    # Add trend to precipitation (increasing before flood)
    df["precipitation"] = df["precipitation"] + np.linspace(0, 20, n_samples)
    
    print(f"Data shape: {df.shape}")
    print(f"Variables: {list(df.columns)}")
    
    # Initialize engine
    engine = FingerprintEngine(
        n_kernels=1000,  # Reduced for test
        embedding_dim=50,
        use_physics_weights=True,
    )
    
    # Extract fingerprint
    print("\n🎵 Extracting fingerprint...")
    
    fp = engine.extract(
        df=df,
        event_id="lago_maggiore_2000_test",
        sources=["era5_synthetic", "noaa_indices_synthetic"],
        bbox=(44.0, 47.0, 7.0, 11.0),
        center=(45.8, 8.7),
    )
    
    print(f"\n✅ Fingerprint extracted:")
    print(f"   Event: {fp.event_id}")
    print(f"   Embedding dim: {len(fp.embedding)}")
    print(f"   Variables: {fp.variables_used}")
    print(f"   Sources: {fp.sources_used}")
    print(f"   Samples: {fp.n_samples}")
    print(f"   Uncertainty: {fp.uncertainty:.3f}")
    print(f"   Method: {fp.extraction_method}")
    
    # Self-similarity test
    print("\n🔍 Self-similarity test...")
    self_sim = engine.similarity(fp, fp)
    print(f"   Self-similarity: {self_sim:.4f} (should be ~1.0)")
    
    # Create similar event (same pattern, small noise)
    print("\n📊 Creating similar event...")
    df_similar = df + np.random.normal(0, 0.1, df.shape)
    fp_similar = engine.extract(
        df=df_similar,
        event_id="similar_event",
        sources=["era5_synthetic"],
    )
    
    sim_score = engine.similarity(fp, fp_similar)
    print(f"   Similarity to similar event: {sim_score:.4f} (should be >0.8)")
    
    # Create different event
    print("\n📊 Creating different event...")
    df_different = pd.DataFrame({
        "precipitation": np.random.exponential(1, n_samples),  # Less rain
        "pressure_msl": 101325 + np.random.exponential(500, n_samples),  # High pressure
        "temperature_2m": 295 + np.random.normal(0, 2, n_samples),  # Warmer
        "soil_moisture": np.random.uniform(0.1, 0.2, n_samples),  # Dry
        "nao_index": np.random.normal(1.0, 0.5, n_samples),  # Positive NAO
    }, index=dates)
    
    fp_different = engine.extract(
        df=df_different,
        event_id="different_event",
        sources=["era5_synthetic"],
    )
    
    diff_score = engine.similarity(fp, fp_different)
    print(f"   Similarity to different event: {diff_score:.4f} (should be <0.7)")
    
    # Find similar
    print("\n🔎 Finding similar events...")
    results = engine.find_similar(
        query=fp,
        candidates=[fp_similar, fp_different],
        top_k=5,
    )
    
    for r in results:
        print(f"   #{r.rank} {r.match_id}: similarity={r.similarity:.4f}")
    
    print("\n✅ Fingerprint Engine test complete!")
