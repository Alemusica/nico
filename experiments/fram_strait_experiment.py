"""
🌊 Fram Strait Evolution Experiment
====================================
Multi-satellite tracking of sea level dynamics in Fram Strait
with Physics Engine + Experience Engine integration.

Fram Strait: Gateway between Arctic Ocean and Atlantic
- Key area for Arctic freshwater export
- Multiple satellite passes (Sentinel-3, Jason, TOPEX)
- Different temporal resolutions and orbits

This experiment:
1. Loads data from multiple satellites (SLCCI, CMEMS, AVISO)
2. Tracks evolution of sea level at Fram Strait
3. Correlates parameters across different timelines
4. Builds "experience" (learned correlations) validated by "physics" (constraints)
5. Discovers cross-satellite lag patterns
"""

import asyncio
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import json

# Fram Strait bounding box
FRAM_STRAIT = {
    "name": "Fram Strait",
    "lat_min": 77.0,
    "lat_max": 81.0,
    "lon_min": -10.0,
    "lon_max": 15.0,
    "description": "Gateway between Arctic Ocean and North Atlantic",
    "importance": "Controls ~50% of Arctic freshwater export"
}

# Norwegian Sea - area with both SLCCI (Jason ~66°N max) and CMEMS coverage
NORWEGIAN_SEA = {
    "name": "Norwegian Sea",
    "lat_min": 62.0,
    "lat_max": 66.0,
    "lon_min": -5.0,
    "lon_max": 10.0,
    "lon_360_min": 355.0,  # For SLCCI (0-360 convention)
    "lon_360_max": 10.0,
    "description": "Transition zone between North Atlantic and Arctic",
    "importance": "Key area for Atlantic water inflow to Arctic"
}

# Use Norwegian Sea for actual data, can switch to Fram for CMEMS-only
STUDY_AREA = NORWEGIAN_SEA


class SatelliteSource(Enum):
    """Available satellite data sources."""
    SLCCI_JASON1 = "slcci_j1"      # Jason-1 altimetry (10-day repeat)
    SLCCI_JASON2 = "slcci_j2"      # Jason-2 altimetry (10-day repeat)
    CMEMS_L3 = "cmems_l3"          # Along-track L3 products
    CMEMS_L4 = "cmems_l4"          # Gridded L4 products (daily)
    AVISO = "aviso"                # AVISO gridded products
    SENTINEL3 = "sentinel3"        # Sentinel-3 (27-day repeat)


@dataclass
class SatellitePass:
    """A single satellite observation over the study area."""
    source: SatelliteSource
    timestamp: datetime
    cycle: Optional[int] = None
    track: Optional[int] = None
    lat: float = 0.0
    lon: float = 0.0
    
    # Measurements
    sla: Optional[float] = None           # Sea Level Anomaly (m)
    adt: Optional[float] = None           # Absolute Dynamic Topography (m)
    mdt: Optional[float] = None           # Mean Dynamic Topography (m)
    sla_error: Optional[float] = None     # Uncertainty (m)
    
    # Derived
    geostrophic_u: Optional[float] = None  # Geostrophic velocity U (m/s)
    geostrophic_v: Optional[float] = None  # Geostrophic velocity V (m/s)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesPoint:
    """A point in the evolution timeline."""
    timestamp: datetime
    source: SatelliteSource
    variable: str
    value: float
    uncertainty: Optional[float] = None
    quality_flag: int = 0


@dataclass
class CrossCorrelation:
    """Correlation between two variables from different sources."""
    source_a: SatelliteSource
    source_b: SatelliteSource
    variable_a: str
    variable_b: str
    lag_days: int
    correlation: float
    p_value: float
    n_samples: int
    physics_valid: bool = False
    physics_score: float = 0.0
    experience_confidence: float = 0.0


class PhysicsEngine:
    """
    Physics Engine: Validates correlations against physical laws.
    
    Physical constraints for oceanographic data:
    1. Geostrophic balance: f*u = -g*(dη/dy)
    2. Conservation: mass, momentum, energy
    3. Causality: cause must precede effect
    4. Propagation speeds: signals travel at finite speed
    """
    
    # Physical constants
    G = 9.81  # Gravity (m/s²)
    OMEGA = 7.2921e-5  # Earth rotation rate (rad/s)
    
    # Fram Strait characteristics
    FRAM_DEPTH_M = 2600  # Average depth
    FRAM_WIDTH_KM = 400  # Width
    
    # Known propagation speeds (m/s)
    BAROTROPIC_SPEED = np.sqrt(G * FRAM_DEPTH_M)  # ~160 m/s
    BAROCLINIC_SPEED = 0.5  # First baroclinic mode ~0.5 m/s
    ROSSBY_SPEED = 0.01  # Rossby wave at 80°N ~0.01 m/s
    
    def __init__(self):
        self.constraints = self._load_constraints()
        
    def _load_constraints(self) -> Dict[str, Any]:
        """Load physical constraints."""
        return {
            "max_sla_change_per_day": 0.1,  # m/day (reasonable limit)
            "max_velocity": 2.0,  # m/s (maximum realistic current)
            "min_lag_barotropic": 0.001,  # days (nearly instantaneous)
            "max_lag_baroclinic": 365,  # days (slow internal waves)
            "temperature_sla_sensitivity": 0.002,  # m/°C (steric effect)
            "salinity_sla_sensitivity": 0.0007,  # m/PSU (haline effect)
        }
    
    def validate_correlation(
        self,
        corr: CrossCorrelation,
        var_a_type: str,
        var_b_type: str
    ) -> Tuple[bool, float, str]:
        """
        Validate a correlation against physics.
        
        Returns: (is_valid, physics_score, explanation)
        """
        reasons = []
        score = 0.0
        
        # 1. Causality check: lag must be positive for cause→effect
        if corr.lag_days < 0:
            # Negative lag means B leads A - check if physically plausible
            if self._can_b_cause_a(var_b_type, var_a_type):
                score += 0.2
                reasons.append("Reverse causality is physically possible")
            else:
                reasons.append("Negative lag suggests spurious correlation")
        else:
            score += 0.3
            reasons.append("Temporal ordering is correct")
        
        # 2. Lag magnitude check: is the lag physically reasonable?
        expected_lag = self._expected_lag(var_a_type, var_b_type, FRAM_STRAIT["lat_min"])
        if expected_lag:
            lag_ratio = abs(corr.lag_days - expected_lag) / max(expected_lag, 1)
            if lag_ratio < 0.5:
                score += 0.3
                reasons.append(f"Lag ({corr.lag_days}d) matches expected ({expected_lag:.0f}d)")
            elif lag_ratio < 1.0:
                score += 0.15
                reasons.append(f"Lag partially matches expected")
            else:
                reasons.append(f"Lag differs from expected ({expected_lag:.0f}d)")
        
        # 3. Correlation sign check: is the sign physically correct?
        expected_sign = self._expected_sign(var_a_type, var_b_type)
        if expected_sign:
            actual_sign = np.sign(corr.correlation)
            if actual_sign == expected_sign:
                score += 0.2
                reasons.append("Correlation sign is physically correct")
            else:
                reasons.append("Correlation sign is unexpected")
        
        # 4. Magnitude check: is correlation strength reasonable?
        if abs(corr.correlation) > 0.95:
            reasons.append("Very high correlation - check for same-source bias")
        elif abs(corr.correlation) > 0.5:
            score += 0.2
            reasons.append("Strong physically meaningful correlation")
        elif abs(corr.correlation) > 0.3:
            score += 0.1
            reasons.append("Moderate correlation")
        
        is_valid = score >= 0.5
        explanation = "; ".join(reasons)
        
        return is_valid, score, explanation
    
    def _expected_lag(self, var_a: str, var_b: str, latitude: float) -> Optional[float]:
        """Calculate expected lag between variables based on physics."""
        # Distance scale (km) and propagation
        distance_km = 500  # Approximate scale for Fram Strait dynamics
        
        # Calculate Coriolis parameter
        f = 2 * self.OMEGA * np.sin(np.radians(latitude))
        
        # Rossby radius (internal)
        rossby_radius = self.BAROCLINIC_SPEED / abs(f) / 1000  # km
        
        lag_days = {
            ("sla", "sla"): 0,  # Same variable, no lag
            ("sla", "adt"): 0,  # Essentially same
            ("sla", "temperature"): 30,  # Steric response ~1 month
            ("sla", "velocity"): 1,  # Geostrophic adjustment ~1 day
            ("nao", "sla"): 60,  # NAO teleconnection ~2 months
            ("amo", "sla"): 180,  # AMO slower ~6 months
            ("ice_extent", "sla"): 90,  # Ice-ocean coupling ~3 months
        }
        
        key = (var_a.lower(), var_b.lower())
        return lag_days.get(key, lag_days.get((var_b.lower(), var_a.lower())))
    
    def _expected_sign(self, var_a: str, var_b: str) -> Optional[int]:
        """Expected sign of correlation based on physics."""
        signs = {
            ("sla", "temperature"): 1,   # Warmer → higher SLA (steric)
            ("sla", "salinity"): -1,     # Saltier → lower SLA
            ("nao", "sla"): 1,           # Positive NAO → higher SLA in Arctic
            ("ice_extent", "sla"): -1,   # More ice → lower SLA (loading)
            ("velocity", "sla_gradient"): 1,  # Geostrophic balance
        }
        key = (var_a.lower(), var_b.lower())
        return signs.get(key, signs.get((var_b.lower(), var_a.lower())))
    
    def _can_b_cause_a(self, var_b: str, var_a: str) -> bool:
        """Check if variable B can physically cause variable A."""
        # Define causal possibilities
        can_cause = {
            "temperature": ["sla", "density"],
            "wind": ["sla", "velocity", "waves"],
            "nao": ["sla", "temperature", "velocity"],
            "pressure": ["sla"],
        }
        return var_a.lower() in can_cause.get(var_b.lower(), [])


class ExperienceEngine:
    """
    Experience Engine: Learns patterns from data.
    
    Unlike the Physics Engine (deductive), the Experience Engine
    is inductive - it learns correlations from historical data
    and builds confidence through repeated observations.
    """
    
    def __init__(self):
        self.learned_patterns: List[Dict[str, Any]] = []
        self.correlation_history: Dict[str, List[float]] = {}
        self.confidence_threshold = 0.7
        
    def record_observation(
        self,
        pattern_id: str,
        correlation: float,
        context: Dict[str, Any]
    ):
        """Record a new observation for a pattern."""
        if pattern_id not in self.correlation_history:
            self.correlation_history[pattern_id] = []
        
        self.correlation_history[pattern_id].append(correlation)
        
    def calculate_confidence(self, pattern_id: str) -> float:
        """
        Calculate confidence in a pattern based on historical observations.
        
        Confidence grows with:
        1. Number of observations
        2. Consistency of correlation values
        3. Variety of conditions (seasons, years)
        """
        if pattern_id not in self.correlation_history:
            return 0.0
        
        obs = self.correlation_history[pattern_id]
        n = len(obs)
        
        if n < 3:
            return 0.1 * n  # Low confidence with few observations
        
        # Consistency: inverse of standard deviation
        std = np.std(obs)
        mean = np.mean(obs)
        consistency = 1.0 / (1.0 + std)
        
        # Sample size factor (saturates at ~30 observations)
        sample_factor = 1 - np.exp(-n / 10)
        
        # Effect size (stronger correlations = more confident)
        effect_factor = min(abs(mean) / 0.5, 1.0)
        
        confidence = 0.4 * consistency + 0.3 * sample_factor + 0.3 * effect_factor
        
        return min(confidence, 1.0)
    
    def find_similar_patterns(
        self,
        new_pattern: CrossCorrelation,
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Find patterns similar to a new observation."""
        similar = []
        
        for pattern in self.learned_patterns:
            similarity = self._pattern_similarity(new_pattern, pattern)
            if similarity >= threshold:
                similar.append({
                    "pattern": pattern,
                    "similarity": similarity
                })
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)
    
    def _pattern_similarity(
        self,
        new: CrossCorrelation,
        existing: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two patterns."""
        score = 0.0
        
        # Same variables
        if new.variable_a == existing.get("variable_a"):
            score += 0.3
        if new.variable_b == existing.get("variable_b"):
            score += 0.3
        
        # Similar lag
        lag_diff = abs(new.lag_days - existing.get("lag_days", 0))
        if lag_diff <= 3:
            score += 0.2
        elif lag_diff <= 7:
            score += 0.1
        
        # Similar correlation
        corr_diff = abs(new.correlation - existing.get("correlation", 0))
        if corr_diff < 0.1:
            score += 0.2
        elif corr_diff < 0.2:
            score += 0.1
        
        return score
    
    def update_learned_patterns(
        self,
        correlation: CrossCorrelation,
        physics_valid: bool,
        physics_score: float
    ):
        """Update learned patterns with new observation."""
        pattern_id = f"{correlation.variable_a}_{correlation.variable_b}_{correlation.lag_days}"
        
        self.record_observation(
            pattern_id,
            correlation.correlation,
            {
                "source_a": correlation.source_a.value,
                "source_b": correlation.source_b.value,
                "physics_valid": physics_valid,
                "physics_score": physics_score,
            }
        )
        
        # Check if pattern should be promoted to "learned"
        confidence = self.calculate_confidence(pattern_id)
        
        if confidence >= self.confidence_threshold and physics_valid:
            existing = next(
                (p for p in self.learned_patterns if p["id"] == pattern_id),
                None
            )
            
            if existing:
                existing["confidence"] = confidence
                existing["n_observations"] = len(self.correlation_history[pattern_id])
            else:
                self.learned_patterns.append({
                    "id": pattern_id,
                    "variable_a": correlation.variable_a,
                    "variable_b": correlation.variable_b,
                    "lag_days": correlation.lag_days,
                    "correlation": np.mean(self.correlation_history[pattern_id]),
                    "confidence": confidence,
                    "physics_score": physics_score,
                    "n_observations": len(self.correlation_history[pattern_id]),
                })


class FramStraitExperiment:
    """
    Main experiment class for Fram Strait evolution tracking.
    
    Workflow:
    1. Load satellite data from multiple sources
    2. Extract observations for Fram Strait region
    3. Build time series for each parameter
    4. Calculate cross-correlations with various lags
    5. Validate with Physics Engine
    6. Learn patterns with Experience Engine
    7. Discover cross-satellite lag patterns
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.physics = PhysicsEngine()
        self.experience = ExperienceEngine()
        
        self.observations: List[SatellitePass] = []
        self.time_series: Dict[str, List[TimeSeriesPoint]] = {}
        self.correlations: List[CrossCorrelation] = []
        
    async def load_slcci_data(self) -> int:
        """Load SLCCI Jason-1 cycle data."""
        slcci_dir = self.data_dir / "slcci"
        n_loaded = 0
        
        for nc_file in sorted(slcci_dir.glob("SLCCI_ALTDB_J1_Cycle*.nc")):
            try:
                ds = xr.open_dataset(nc_file)
                
                # Extract cycle number from filename
                cycle = int(nc_file.stem.split("Cycle")[1].split("_")[0])
                
                # SLCCI uses latitude/longitude coords directly in data vars
                lat = ds['latitude'].values
                lon = ds['longitude'].values  # 0-360 convention
                
                # Convert lon to -180/180 for comparison, or use 0-360 bounds
                # SLCCI uses 0-360, so we need to handle wrap-around
                if "lon_360_min" in STUDY_AREA:
                    # Handle 0-360 coordinate system
                    if STUDY_AREA["lon_360_min"] > STUDY_AREA["lon_360_max"]:
                        # Wraps around 0/360
                        lon_mask = (lon >= STUDY_AREA["lon_360_min"]) | (lon <= STUDY_AREA["lon_360_max"])
                    else:
                        lon_mask = (lon >= STUDY_AREA["lon_360_min"]) & (lon <= STUDY_AREA["lon_360_max"])
                else:
                    # Convert SLCCI lon to -180/180
                    lon_180 = np.where(lon > 180, lon - 360, lon)
                    lon_mask = (lon_180 >= STUDY_AREA["lon_min"]) & (lon_180 <= STUDY_AREA["lon_max"])
                
                lat_mask = (lat >= STUDY_AREA["lat_min"]) & (lat <= STUDY_AREA["lat_max"])
                mask = lat_mask & lon_mask
                
                if mask.any():
                    n_points = mask.sum()
                    
                    # Get time - SLCCI has time coordinate
                    if 'time' in ds.coords:
                        time_vals = ds['time'].values
                    else:
                        # Estimate from cycle
                        base_date = datetime(2002, 1, 15)
                        time_vals = np.array([np.datetime64(base_date + timedelta(days=cycle * 9.9156))])
                    
                    # Get SLA/SSH data - SLCCI has 'corssh' (corrected SSH)
                    sla_var = next(
                        (v for v in ['sla', 'SLA', 'sea_level_anomaly', 'ssha', 'corssh'] if v in ds),
                        None
                    )
                    
                    if sla_var and n_points > 0:
                        sla_data = ds[sla_var].values
                        sla_region = sla_data[mask]
                        
                        # Get mean time in region
                        time_region = time_vals[mask] if len(time_vals) == len(sla_data) else time_vals
                        
                        # Create regional average observation
                        mean_time = pd.to_datetime(time_region[0] if hasattr(time_region, '__len__') else time_region)
                        
                        obs = SatellitePass(
                            source=SatelliteSource.SLCCI_JASON1,
                            timestamp=mean_time.to_pydatetime() if hasattr(mean_time, 'to_pydatetime') else mean_time,
                            cycle=cycle,
                            lat=float(np.mean(lat[mask])),
                            lon=float(np.mean(lon[mask])),
                            sla=float(np.nanmean(sla_region)),
                            metadata={"file": nc_file.name, "n_points": int(n_points)}
                        )
                        self.observations.append(obs)
                        n_loaded += 1
                
                ds.close()
                
            except Exception as e:
                print(f"  Warning: Could not load {nc_file.name}: {e}")
        
        return n_loaded
    
    async def load_cmems_data(self) -> int:
        """Load CMEMS L3 and L4 data."""
        cmems_dir = self.data_dir / "cmems"
        n_loaded = 0
        
        for nc_file in cmems_dir.glob("cmems_*.nc"):
            try:
                ds = xr.open_dataset(nc_file)
                
                # Determine L3 or L4
                is_l4 = "l4" in nc_file.name.lower()
                source = SatelliteSource.CMEMS_L4 if is_l4 else SatelliteSource.CMEMS_L3
                
                # Get coordinates
                lat_var = next((v for v in ['lat', 'latitude'] if v in ds.coords), None)
                lon_var = next((v for v in ['lon', 'longitude'] if v in ds.coords), None)
                
                if lat_var and lon_var:
                    lat = ds[lat_var].values
                    lon = ds[lon_var].values
                    
                    # CMEMS typically uses -180/180 lon
                    lat_mask = (lat >= STUDY_AREA["lat_min"]) & (lat <= STUDY_AREA["lat_max"])
                    lon_mask = (lon >= STUDY_AREA["lon_min"]) & (lon <= STUDY_AREA["lon_max"])
                    
                    if lat_mask.any() and lon_mask.any():
                        # Get time
                        if 'time' in ds.coords:
                            times = pd.to_datetime(ds['time'].values)
                        else:
                            times = pd.DatetimeIndex([datetime(2020, 1, 1)])
                        
                        # Get SLA
                        sla_var = next(
                            (v for v in ['sla', 'adt', 'sea_level_anomaly'] if v in ds),
                            None
                        )
                        
                        if sla_var:
                            for t in times:
                                try:
                                    sla_slice = ds[sla_var].sel(time=t, method='nearest')
                                    # Select region
                                    sla_region = sla_slice.sel(
                                        **{lat_var: lat[lat_mask], lon_var: lon[lon_mask]}
                                    )
                                    sla_mean = float(sla_region.mean())
                                    
                                    if not np.isnan(sla_mean):
                                        obs = SatellitePass(
                                            source=source,
                                            timestamp=t.to_pydatetime(),
                                            lat=float(np.mean(lat[lat_mask])),
                                            lon=float(np.mean(lon[lon_mask])),
                                            sla=sla_mean,
                                            metadata={"file": nc_file.name, "product": "L4" if is_l4 else "L3"}
                                        )
                                        self.observations.append(obs)
                                        n_loaded += 1
                                except Exception as e:
                                    pass  # Skip this time step
                
                ds.close()
                
            except Exception as e:
                print(f"  Warning: Could not load {nc_file.name}: {e}")
        
        return n_loaded
    
    def build_time_series(self):
        """Build time series from observations grouped by source and variable."""
        self.time_series = {}
        
        for obs in self.observations:
            key = f"{obs.source.value}_sla"
            
            if key not in self.time_series:
                self.time_series[key] = []
            
            if obs.sla is not None:
                self.time_series[key].append(TimeSeriesPoint(
                    timestamp=obs.timestamp,
                    source=obs.source,
                    variable="sla",
                    value=obs.sla,
                    uncertainty=obs.sla_error
                ))
        
        # Sort by time
        for key in self.time_series:
            self.time_series[key].sort(key=lambda x: x.timestamp)
    
    def calculate_cross_correlations(
        self,
        max_lag_days: int = 90,
        lag_step_days: int = 7
    ) -> List[CrossCorrelation]:
        """
        Calculate cross-correlations between different satellite sources.
        
        This is where we discover lags between parameters from different
        sensors that don't share the same timeline.
        """
        correlations = []
        sources = list(set(ts.split("_")[0] for ts in self.time_series.keys()))
        
        print(f"\n  Calculating correlations across {len(sources)} sources...")
        
        for i, src_a in enumerate(sources):
            for src_b in sources[i:]:  # Include self-correlation
                ts_a_key = f"{src_a}_sla"
                ts_b_key = f"{src_b}_sla"
                
                if ts_a_key not in self.time_series or ts_b_key not in self.time_series:
                    continue
                
                ts_a = self.time_series[ts_a_key]
                ts_b = self.time_series[ts_b_key]
                
                # Convert to pandas for easier time alignment
                df_a = pd.DataFrame([
                    {"time": p.timestamp, "value": p.value}
                    for p in ts_a
                ]).set_index("time")
                
                df_b = pd.DataFrame([
                    {"time": p.timestamp, "value": p.value}
                    for p in ts_b
                ]).set_index("time")
                
                # Try different lags
                for lag in range(-max_lag_days, max_lag_days + 1, lag_step_days):
                    # Shift series B by lag days
                    df_b_shifted = df_b.shift(periods=lag, freq='D')
                    
                    # Resample to common frequency (weekly) for comparison
                    df_a_weekly = df_a.resample('7D').mean()
                    df_b_weekly = df_b_shifted.resample('7D').mean()
                    
                    # Align and correlate
                    aligned = pd.concat([df_a_weekly, df_b_weekly], axis=1, join='inner')
                    aligned.columns = ['a', 'b']
                    aligned = aligned.dropna()
                    
                    if len(aligned) >= 5:  # Minimum samples
                        corr = aligned['a'].corr(aligned['b'])
                        
                        if not np.isnan(corr):
                            # Calculate p-value (simplified)
                            n = len(aligned)
                            t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2)
                            # Rough p-value approximation
                            p_value = 2 * (1 - min(0.99, abs(t_stat) / 10))
                            
                            cross_corr = CrossCorrelation(
                                source_a=SatelliteSource(src_a),
                                source_b=SatelliteSource(src_b),
                                variable_a="sla",
                                variable_b="sla",
                                lag_days=lag,
                                correlation=corr,
                                p_value=p_value,
                                n_samples=n
                            )
                            correlations.append(cross_corr)
        
        self.correlations = correlations
        return correlations
    
    def validate_and_learn(self) -> Dict[str, Any]:
        """
        Run Physics Engine validation + Experience Engine learning.
        
        This is where the two engines work together:
        1. Physics validates if correlation is physically plausible
        2. Experience learns patterns that are repeatedly observed
        3. Only physics-valid patterns with high experience confidence become "knowledge"
        """
        results = {
            "total_correlations": len(self.correlations),
            "physics_valid": 0,
            "physics_invalid": 0,
            "learned_patterns": 0,
            "best_correlations": [],
        }
        
        print("\n  Validating correlations with Physics + Experience engines...")
        
        for corr in self.correlations:
            # Physics Engine validation
            is_valid, physics_score, explanation = self.physics.validate_correlation(
                corr, 
                corr.variable_a, 
                corr.variable_b
            )
            
            corr.physics_valid = is_valid
            corr.physics_score = physics_score
            
            if is_valid:
                results["physics_valid"] += 1
            else:
                results["physics_invalid"] += 1
            
            # Experience Engine learning
            self.experience.update_learned_patterns(corr, is_valid, physics_score)
            corr.experience_confidence = self.experience.calculate_confidence(
                f"{corr.variable_a}_{corr.variable_b}_{corr.lag_days}"
            )
        
        results["learned_patterns"] = len(self.experience.learned_patterns)
        
        # Find best correlations (high physics score + high experience confidence)
        best = sorted(
            [c for c in self.correlations if c.physics_valid],
            key=lambda x: x.physics_score * (1 + x.experience_confidence),
            reverse=True
        )[:10]
        
        results["best_correlations"] = [
            {
                "source_a": c.source_a.value,
                "source_b": c.source_b.value,
                "lag_days": c.lag_days,
                "correlation": round(c.correlation, 3),
                "physics_score": round(c.physics_score, 2),
                "experience_confidence": round(c.experience_confidence, 2),
            }
            for c in best
        ]
        
        return results
    
    async def run(self) -> Dict[str, Any]:
        """Run the complete experiment."""
        print("╔══════════════════════════════════════════════════╗")
        print("║      MULTI-SATELLITE EVOLUTION EXPERIMENT        ║")
        print("║  Physics + Experience Engine validation          ║")
        print("╚══════════════════════════════════════════════════╝")
        
        print(f"\n📍 Study Area: {STUDY_AREA['name']}")
        print(f"   Lat: {STUDY_AREA['lat_min']}° - {STUDY_AREA['lat_max']}°")
        print(f"   Lon: {STUDY_AREA['lon_min']}° - {STUDY_AREA['lon_max']}°")
        print(f"   {STUDY_AREA['importance']}")
        
        # Step 1: Load data
        print("\n📡 STEP 1: Loading satellite data...")
        
        n_slcci = await self.load_slcci_data()
        print(f"   SLCCI (Jason-1): {n_slcci} observations")
        
        n_cmems = await self.load_cmems_data()
        print(f"   CMEMS (L3/L4): {n_cmems} observations")
        
        print(f"   Total: {len(self.observations)} observations")
        
        if len(self.observations) == 0:
            print("\n⚠️  No data found in Fram Strait region.")
            print("   This demo will show the concept with simulated data...")
            self._generate_demo_data()
        
        # Step 2: Build time series
        print("\n📊 STEP 2: Building time series...")
        self.build_time_series()
        
        for key, ts in self.time_series.items():
            if len(ts) > 0:
                t_min = min(p.timestamp for p in ts)
                t_max = max(p.timestamp for p in ts)
                print(f"   {key}: {len(ts)} points ({t_min.date()} to {t_max.date()})")
        
        # Step 3: Cross-correlations
        print("\n🔗 STEP 3: Calculating cross-correlations...")
        correlations = self.calculate_cross_correlations()
        print(f"   Calculated {len(correlations)} lag-correlation pairs")
        
        # Step 4: Physics + Experience validation
        print("\n⚡ STEP 4: Physics + Experience Engine validation...")
        results = self.validate_and_learn()
        
        print(f"\n   Physics validation:")
        print(f"     ✓ Valid: {results['physics_valid']}")
        print(f"     ✗ Invalid: {results['physics_invalid']}")
        
        print(f"\n   Experience learning:")
        print(f"     📚 Learned patterns: {results['learned_patterns']}")
        
        # Step 5: Report best correlations
        print("\n🏆 STEP 5: Best cross-satellite correlations:")
        print("   " + "-" * 60)
        
        for i, c in enumerate(results["best_correlations"][:5], 1):
            print(f"   {i}. {c['source_a']} ↔ {c['source_b']}")
            print(f"      Lag: {c['lag_days']} days, r={c['correlation']:.3f}")
            print(f"      Physics: {c['physics_score']:.2f}, Experience: {c['experience_confidence']:.2f}")
        
        # Step 6: Knowledge graph integration summary
        print("\n📈 STEP 6: Knowledge ready for graph storage:")
        print("   " + "-" * 60)
        print("   Patterns to store in Neo4j/SurrealDB:")
        
        for pattern in self.experience.learned_patterns[:3]:
            print(f"   • {pattern['variable_a']} → {pattern['variable_b']}")
            print(f"     Lag: {pattern['lag_days']}d, Confidence: {pattern['confidence']:.2f}")
        
        return {
            "study_area": STUDY_AREA,
            "observations": len(self.observations),
            "time_series": {k: len(v) for k, v in self.time_series.items()},
            "correlations": len(self.correlations),
            "results": results,
            "learned_patterns": self.experience.learned_patterns,
        }
    
    def _generate_demo_data(self):
        """Generate demo data when no real data available in region."""
        print("   Generating synthetic observations for demonstration...")
        
        base_date = datetime(2020, 1, 1)
        
        # Jason-1 style (10-day repeat)
        for cycle in range(1, 37):  # ~1 year
            t = base_date + timedelta(days=cycle * 9.9156)
            # Seasonal signal + noise
            seasonal = 0.1 * np.sin(2 * np.pi * cycle * 9.9156 / 365)
            noise = np.random.normal(0, 0.02)
            
            self.observations.append(SatellitePass(
                source=SatelliteSource.SLCCI_JASON1,
                timestamp=t,
                cycle=cycle,
                lat=79.0,
                lon=0.0,
                sla=seasonal + noise,
            ))
        
        # CMEMS L4 style (daily, but with 5-day lag from Jason)
        lag_days = 5
        for day in range(0, 365, 7):  # Weekly
            t = base_date + timedelta(days=day)
            # Same seasonal but lagged + different noise
            seasonal = 0.1 * np.sin(2 * np.pi * (day - lag_days) / 365)
            noise = np.random.normal(0, 0.03)
            
            self.observations.append(SatellitePass(
                source=SatelliteSource.CMEMS_L4,
                timestamp=t,
                lat=79.0,
                lon=0.0,
                sla=seasonal + noise + 0.01,  # Small offset
            ))
        
        print(f"   Generated {len(self.observations)} synthetic observations")


async def main():
    """Run the Fram Strait experiment."""
    experiment = FramStraitExperiment(data_dir="data")
    results = await experiment.run()
    
    # Save results
    output_file = Path("experiments/fram_strait_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "learned_patterns"},
            f,
            indent=2,
            default=str
        )
    
    print(f"\n💾 Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
