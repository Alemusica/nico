#!/usr/bin/env python3
"""
🔮 Historical Episode Analysis Experiment
=========================================

Searches for PRECURSOR SIGNALS before well-documented historical events.
This is the key to PREDICTION: identifying patterns that consistently
appear BEFORE significant oceanographic events.

The goal is to find:
1. What satellite signals (SSH, SLA, SST) appear N days BEFORE the event
2. How reliable are these precursors (correlation, lag stability)
3. Can we use them to predict future similar events

This experiment uses the Physics + Experience engines to validate findings.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Self-contained Physics Validator
# =============================================================================

class PhysicsValidator:
    """
    Validates causal relationships against physical oceanography.
    Checks if observed time lags are consistent with known current speeds.
    """
    
    def __init__(self):
        # Ocean current speeds in m/s for different regions
        self.currents = {
            "wsc": {"name": "West Spitsbergen Current", "speed_range": (0.1, 0.3)},
            "ncc": {"name": "Norwegian Coastal Current", "speed_range": (0.05, 0.15)},
            "eic": {"name": "East Iceland Current", "speed_range": (0.02, 0.08)},
            "egc": {"name": "East Greenland Current", "speed_range": (0.05, 0.2)},
            "nac": {"name": "North Atlantic Current", "speed_range": (0.1, 0.4)},
        }
        
        # Typical distances between regions (km)
        self.distances = {
            ("norwegian_sea", "fram_strait"): 1500,
            ("north_atlantic", "norwegian_sea"): 2000,
            ("barents_sea", "central_arctic"): 800,
            ("norwegian_sea", "barents_sea"): 1000,
        }
    
    def validate_lag(
        self, 
        source_region: str, 
        target_region: str, 
        observed_lag_days: int,
        current_id: str = None
    ) -> Tuple[bool, float, str]:
        """
        Validate if an observed lag is physically plausible.
        
        Returns:
            (is_valid, physics_score, explanation)
        """
        # Determine current to use
        if current_id is None:
            current_id = self._infer_current(source_region, target_region)
        
        if current_id not in self.currents:
            return True, 0.5, "Unknown current system"
        
        current = self.currents[current_id]
        v_min, v_max = current["speed_range"]
        
        # Get distance
        distance_km = self._get_distance(source_region, target_region)
        
        # Calculate expected travel time range
        # Convert m/s to km/day: multiply by 86.4
        min_days = distance_km / (v_max * 86.4)
        max_days = distance_km / (v_min * 86.4)
        
        # Check if observed lag is within reasonable range (with 50% tolerance)
        is_valid = (min_days * 0.5) <= observed_lag_days <= (max_days * 2.0)
        
        # Calculate physics score (1.0 = perfect match)
        expected_mid = (min_days + max_days) / 2
        deviation = abs(observed_lag_days - expected_mid) / expected_mid
        physics_score = max(0, 1 - deviation)
        
        explanation = (f"{current['name']}: expected {min_days:.0f}-{max_days:.0f} days, "
                      f"observed {observed_lag_days} days")
        
        return is_valid, physics_score, explanation
    
    def _infer_current(self, source: str, target: str) -> str:
        """Infer which current system connects two regions."""
        s, t = source.lower(), target.lower()
        
        if "norwegian" in s or "atlantic" in s:
            if "fram" in t or "arctic" in t:
                return "wsc"
        if "barents" in s or "barents" in t:
            return "ncc"
        if "greenland" in s:
            return "egc"
        return "nac"
    
    def _get_distance(self, source: str, target: str) -> float:
        """Get approximate distance between regions."""
        s = source.lower().replace(" ", "_")
        t = target.lower().replace(" ", "_")
        
        for (r1, r2), dist in self.distances.items():
            if (r1 in s or s in r1) and (r2 in t or t in r2):
                return dist
            if (r1 in t or t in r1) and (r2 in s or s in r2):
                return dist
        
        return 1500  # Default distance


class ExperienceEngine:
    """
    Learns from repeated observations to improve predictions.
    Stores patterns that have been validated multiple times.
    """
    
    def __init__(self):
        self.observations = {}  # pattern_id -> list of observations
        self.validated_patterns = {}  # pattern_id -> confidence
        self.confidence_threshold = 0.7
    
    def store_observation(
        self,
        pattern_id: str,
        lag: int,
        correlation: float,
        source: str,
        physics_score: float = 1.0
    ):
        """Store an observation of a pattern."""
        if pattern_id not in self.observations:
            self.observations[pattern_id] = []
        
        self.observations[pattern_id].append({
            "lag": lag,
            "correlation": correlation,
            "source": source,
            "physics_score": physics_score,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update confidence
        obs = self.observations[pattern_id]
        if len(obs) >= 2:
            # Average correlation * physics score
            avg_corr = np.mean([o["correlation"] for o in obs])
            avg_physics = np.mean([o["physics_score"] for o in obs])
            confidence = avg_corr * avg_physics
            
            if confidence >= self.confidence_threshold:
                self.validated_patterns[pattern_id] = confidence
    
    def get_confidence(self, pattern_id: str) -> float:
        """Get confidence for a pattern."""
        return self.validated_patterns.get(pattern_id, 0.0)
    
    def get_validated_patterns(self) -> Dict[str, float]:
        """Get all validated patterns."""
        return self.validated_patterns.copy()


@dataclass
class HistoricalEpisode:
    """A well-documented historical event to analyze."""
    id: str
    name: str
    event_type: str
    start_date: str
    end_date: str
    description: str
    precursor_window_days: int  # How far back to look for signals
    region: Dict[str, Tuple[float, float]]  # lat, lon bounds
    known_precursors: List[str]
    references: List[str]


@dataclass
class PrecursorSignal:
    """A discovered precursor signal."""
    variable: str
    source_region: str
    lag_days: int
    correlation: float
    p_value: float
    physics_validated: bool
    mechanism: str
    confidence: float = 0.0


@dataclass
class EpisodeAnalysisResult:
    """Results from analyzing a historical episode."""
    episode: HistoricalEpisode
    precursors: List[PrecursorSignal]
    overall_confidence: float
    max_lead_time: int
    validated_precursors: int
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    

# =============================================================================
# Historical Episodes Database
# =============================================================================

EPISODES = [
    HistoricalEpisode(
        id="arctic_ice_2007",
        name="2007 Arctic Sea Ice Record Minimum",
        event_type="ice_extent_minimum",
        start_date="2007-09-01",
        end_date="2007-09-21",  # Date of minimum extent
        description="Record low Arctic sea ice extent, 4.3 million km², was 23% below "
                   "the previous record (2005). Strong anomalous atmospheric circulation.",
        precursor_window_days=120,
        region={"lat": (70, 85), "lon": (-180, 180)},
        known_precursors=[
            "NAO negative phase (winter)",
            "Warm SST anomalies in Nordic Seas",
            "Anticyclonic circulation over Arctic",
            "Fram Strait ice export anomaly"
        ],
        references=["Stroeve et al., 2008", "Comiso et al., 2008", "Serreze et al., 2007"]
    ),
    
    HistoricalEpisode(
        id="atlantic_intrusion_2015",
        name="2015-16 Atlantic Water Intrusion",
        event_type="heat_transport_anomaly",
        start_date="2015-10-01",
        end_date="2016-03-31",
        description="Anomalous warm Atlantic water intrusion into Arctic via Fram Strait, "
                   "contributed to unprecedented winter sea ice melt.",
        precursor_window_days=150,
        region={"lat": (76, 82), "lon": (-10, 15)},
        known_precursors=[
            "Norwegian Sea warm anomaly (spring)",
            "West Spitsbergen Current strengthening",
            "NAO phase shift",
            "SSH gradient increase"
        ],
        references=["Polyakov et al., 2017", "Årthun et al., 2017", "Smedsrud et al., 2013"]
    ),
    
    HistoricalEpisode(
        id="fram_export_2012",
        name="2012 Fram Strait Ice Export Event",
        event_type="ice_transport",
        start_date="2012-01-01",
        end_date="2012-04-30",
        description="Enhanced sea ice export through Fram Strait driven by strong "
                   "atmospheric pressure gradients.",
        precursor_window_days=90,
        region={"lat": (76, 82), "lon": (-10, 10)},
        known_precursors=[
            "Strong AO positive phase",
            "Central Arctic SSH anomaly",
            "Wind stress pattern change",
            "Transpolar drift acceleration"
        ],
        references=["Kwok et al., 2013", "Smedsrud et al., 2017"]
    ),
]


class HistoricalEpisodeAnalyzer:
    """
    Analyzes historical episodes to find precursor signals.
    
    Strategy:
    1. Load satellite data for time period BEFORE the event
    2. Load data for the event itself (or proxy variable)
    3. Compute cross-correlations at various lags
    4. Validate findings against physics (ocean current speeds)
    5. Store validated patterns in Experience Engine
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / "data"
        self.physics = PhysicsValidator()
        self.experience = ExperienceEngine()
        
        # Current speed estimates for different regions (m/s)
        self.current_speeds = {
            "wsc": (0.1, 0.3),  # West Spitsbergen Current
            "ncc": (0.05, 0.15),  # Norwegian Coastal Current
            "eic": (0.02, 0.08),  # East Iceland Current
            "egc": (0.05, 0.2),   # East Greenland Current
        }
        
        print("🔮 Historical Episode Analyzer initialized")
        print(f"   📁 Data directory: {self.data_dir}")
        print(f"   📊 {len(EPISODES)} historical episodes available")
    
    def list_episodes(self) -> List[Dict]:
        """List all available historical episodes."""
        return [
            {
                "id": ep.id,
                "name": ep.name,
                "type": ep.event_type,
                "date": ep.start_date,
                "precursor_window": ep.precursor_window_days
            }
            for ep in EPISODES
        ]
    
    def analyze_episode(
        self,
        episode_id: str,
        use_synthetic: bool = True
    ) -> EpisodeAnalysisResult:
        """
        Analyze a historical episode to find precursor signals.
        
        Args:
            episode_id: ID of the episode to analyze
            use_synthetic: If True, generate synthetic data for demonstration
            
        Returns:
            Analysis results including discovered precursors
        """
        # Find the episode
        episode = next((ep for ep in EPISODES if ep.id == episode_id), None)
        if not episode:
            raise ValueError(f"Episode '{episode_id}' not found")
        
        print(f"\n{'='*70}")
        print(f"🎯 Analyzing: {episode.name}")
        print(f"{'='*70}")
        print(f"📅 Event period: {episode.start_date} to {episode.end_date}")
        print(f"🔍 Looking for precursors in {episode.precursor_window_days}-day window")
        print(f"📍 Region: Lat {episode.region['lat']}, Lon {episode.region['lon']}")
        print()
        
        # Load or generate data
        if use_synthetic:
            event_data, precursor_data = self._generate_synthetic_data(episode)
        else:
            event_data, precursor_data = self._load_real_data(episode)
        
        # Search for precursor signals
        precursors = self._find_precursors(episode, event_data, precursor_data)
        
        # Validate with physics
        validated_precursors = self._validate_precursors(episode, precursors)
        
        # Store in experience engine
        self._store_patterns(episode, validated_precursors)
        
        # Compute overall confidence
        if validated_precursors:
            overall_confidence = np.mean([p.confidence for p in validated_precursors])
            max_lead_time = max(p.lag_days for p in validated_precursors)
            validated_count = sum(1 for p in validated_precursors if p.physics_validated)
        else:
            overall_confidence = 0.0
            max_lead_time = 0
            validated_count = 0
        
        result = EpisodeAnalysisResult(
            episode=episode,
            precursors=validated_precursors,
            overall_confidence=overall_confidence,
            max_lead_time=max_lead_time,
            validated_precursors=validated_count
        )
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _generate_synthetic_data(
        self,
        episode: HistoricalEpisode
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate synthetic data that mimics the expected signal structure.
        Used for demonstration when real data is unavailable.
        """
        print("📊 Generating synthetic data for demonstration...")
        
        # Create time axis
        event_start = datetime.fromisoformat(episode.start_date)
        precursor_start = event_start - timedelta(days=episode.precursor_window_days)
        
        n_days = episode.precursor_window_days + 30  # Include some of the event
        time = np.arange(n_days)
        
        # Generate event signal (e.g., ice extent index, heat content)
        # The event shows a clear anomaly near the end
        event_signal = np.zeros(n_days)
        event_signal += 0.2 * np.random.randn(n_days)  # Noise
        event_signal[-30:] += np.linspace(0, -2, 30)  # Anomaly
        
        # Generate precursor signals with known relationships
        precursor_signals = {}
        
        # Precursor 1: SSH signal with ~60-90 day lead
        lead_1 = 75
        ssh_signal = np.roll(event_signal, lead_1) + 0.1 * np.random.randn(n_days)
        precursor_signals["Norwegian Sea SSH"] = ssh_signal * 0.8
        
        # Precursor 2: SST with ~30-45 day lead
        lead_2 = 35
        sst_signal = np.roll(event_signal, lead_2) + 0.15 * np.random.randn(n_days)
        precursor_signals["Barents Sea SST"] = sst_signal * 0.7
        
        # Precursor 3: Wind-related (shorter lead)
        lead_3 = 14
        wind_signal = np.roll(event_signal, lead_3) + 0.2 * np.random.randn(n_days)
        precursor_signals["Wind Stress Curl"] = wind_signal * 0.6
        
        # Precursor 4: Atlantic signal (longer lead)
        lead_4 = 100
        atlantic_signal = np.roll(event_signal, lead_4) + 0.12 * np.random.randn(n_days)
        precursor_signals["Atlantic Water Temperature"] = atlantic_signal * 0.85
        
        # Red herring (no real relationship)
        precursor_signals["Random Index"] = 0.3 * np.random.randn(n_days)
        
        return event_signal, precursor_signals
    
    def _load_real_data(
        self,
        episode: HistoricalEpisode
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Load real satellite data for the episode.
        """
        print("📡 Loading real satellite data...")
        
        # Try to load SLCCI data
        slcci_path = self.data_dir / "slcci"
        cmems_path = self.data_dir / "cmems"
        
        # This would load actual NetCDF files
        # For now, fall back to synthetic if files not available
        print("   ⚠️ Full historical data not available, using synthetic")
        return self._generate_synthetic_data(episode)
    
    def _find_precursors(
        self,
        episode: HistoricalEpisode,
        event_data: np.ndarray,
        precursor_data: Dict[str, np.ndarray]
    ) -> List[PrecursorSignal]:
        """
        Search for precursor signals by computing cross-correlations.
        """
        print("\n🔬 Computing cross-correlations...")
        
        precursors = []
        max_lag = episode.precursor_window_days
        
        for name, signal in precursor_data.items():
            print(f"   📈 Analyzing: {name}")
            
            # Compute cross-correlation at various lags
            best_lag = 0
            best_corr = 0
            best_p = 1.0
            
            for lag in range(0, max_lag, 7):  # Test weekly increments
                # Shift and correlate
                if lag > 0:
                    shifted_precursor = signal[:-lag]
                    event_segment = event_data[lag:]
                else:
                    shifted_precursor = signal
                    event_segment = event_data
                
                # Ensure same length
                min_len = min(len(shifted_precursor), len(event_segment))
                if min_len < 20:
                    continue
                
                shifted_precursor = shifted_precursor[:min_len]
                event_segment = event_segment[:min_len]
                
                # Compute correlation
                corr = np.corrcoef(shifted_precursor, event_segment)[0, 1]
                
                # Simple p-value approximation
                n = min_len
                t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + 1e-10))
                p_value = 2 * (1 - min(0.999, abs(t_stat) / 10))  # Simplified
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
                    best_p = p_value
            
            # Create precursor signal if significant
            if abs(best_corr) > 0.3:  # Threshold
                precursor = PrecursorSignal(
                    variable=name,
                    source_region=self._infer_region(name),
                    lag_days=best_lag,
                    correlation=best_corr,
                    p_value=best_p,
                    physics_validated=False,
                    mechanism="TBD",
                    confidence=abs(best_corr)
                )
                precursors.append(precursor)
                print(f"      ✓ Found signal: lag={best_lag}d, r={best_corr:.3f}")
            else:
                print(f"      ✗ No significant signal")
        
        return precursors
    
    def _validate_precursors(
        self,
        episode: HistoricalEpisode,
        precursors: List[PrecursorSignal]
    ) -> List[PrecursorSignal]:
        """
        Validate precursors against physical oceanography.
        """
        print("\n⚙️ Validating against physics...")
        
        validated = []
        
        for p in precursors:
            # Determine appropriate current system
            if "norwegian" in p.variable.lower() or "atlantic" in p.variable.lower():
                current = "wsc"
                mechanism = "Signal propagates via West Spitsbergen Current"
            elif "barents" in p.variable.lower():
                current = "ncc"
                mechanism = "Heat advection through Nordic Seas"
            elif "wind" in p.variable.lower():
                current = None
                mechanism = "Direct atmospheric forcing"
                p.physics_validated = True  # Wind is fast
            elif "greenland" in p.variable.lower():
                current = "egc"
                mechanism = "East Greenland Current signal"
            else:
                current = None
                mechanism = "Unknown mechanism"
            
            # Validate against current speed
            if current:
                v_min, v_max = self.current_speeds[current]
                
                # Calculate expected travel time
                # Rough distance from Norwegian Sea to Fram Strait: ~1500 km
                distance_km = 1500
                min_days = distance_km / (v_max * 86.4)  # 86.4 = km/day per m/s
                max_days = distance_km / (v_min * 86.4)
                
                # Check if observed lag is physically plausible
                if min_days * 0.5 <= p.lag_days <= max_days * 2:
                    p.physics_validated = True
                    p.mechanism = f"{mechanism} ({min_days:.0f}-{max_days:.0f} day transit)"
                    print(f"   ✓ {p.variable}: Lag {p.lag_days}d matches {current.upper()} propagation")
                else:
                    p.physics_validated = False
                    p.mechanism = f"{mechanism} (lag outside expected range)"
                    print(f"   ⚠ {p.variable}: Lag {p.lag_days}d outside expected {min_days:.0f}-{max_days:.0f}d")
            else:
                if "random" in p.variable.lower():
                    p.physics_validated = False
                    p.mechanism = "No physical mechanism identified"
                    print(f"   ✗ {p.variable}: No physical basis")
            
            # Update confidence based on validation
            if p.physics_validated:
                p.confidence = min(1.0, abs(p.correlation) * 1.2)
            else:
                p.confidence = abs(p.correlation) * 0.5
            
            validated.append(p)
        
        return validated
    
    def _store_patterns(
        self,
        episode: HistoricalEpisode,
        precursors: List[PrecursorSignal]
    ):
        """
        Store validated patterns in the Experience Engine for future use.
        """
        print("\n💾 Storing patterns in Experience Engine...")
        
        for p in precursors:
            if p.physics_validated:
                pattern_id = f"{episode.id}_{p.variable.lower().replace(' ', '_')}"
                
                try:
                    self.experience.store_observation(
                        pattern_id=pattern_id,
                        lag=p.lag_days,
                        correlation=p.correlation,
                        source=f"historical_analysis:{episode.id}",
                        physics_score=1.0 if p.physics_validated else 0.5
                    )
                    print(f"   ✓ Stored: {pattern_id}")
                except Exception as e:
                    print(f"   ⚠ Could not store: {e}")
    
    def _infer_region(self, variable_name: str) -> str:
        """Infer the source region from variable name."""
        name_lower = variable_name.lower()
        if "norwegian" in name_lower:
            return "Norwegian Sea"
        elif "barents" in name_lower:
            return "Barents Sea"
        elif "fram" in name_lower:
            return "Fram Strait"
        elif "atlantic" in name_lower:
            return "North Atlantic"
        elif "greenland" in name_lower:
            return "Greenland Sea"
        else:
            return "Unknown"
    
    def _print_summary(self, result: EpisodeAnalysisResult):
        """Print analysis summary."""
        print(f"\n{'='*70}")
        print(f"📊 ANALYSIS SUMMARY: {result.episode.name}")
        print(f"{'='*70}")
        print(f"🎯 Overall Confidence: {result.overall_confidence*100:.1f}%")
        print(f"⏱️  Maximum Lead Time: {result.max_lead_time} days")
        print(f"✅ Validated Precursors: {result.validated_precursors}/{len(result.precursors)}")
        
        print(f"\n📈 Discovered Precursor Signals:")
        print("-" * 70)
        
        # Sort by confidence
        sorted_precursors = sorted(result.precursors, key=lambda x: -x.confidence)
        
        for i, p in enumerate(sorted_precursors, 1):
            status = "✓" if p.physics_validated else "⚠"
            print(f"\n{i}. {p.variable}")
            print(f"   {status} Lag: {p.lag_days} days | Correlation: {p.correlation:.3f} | Confidence: {p.confidence:.2f}")
            print(f"   📍 Region: {p.source_region}")
            print(f"   🔬 Mechanism: {p.mechanism}")
        
        # Predictive insight
        valid_precursors = [p for p in result.precursors if p.physics_validated and p.confidence > 0.5]
        if valid_precursors:
            print(f"\n🔮 PREDICTIVE INSIGHT:")
            print("-" * 70)
            print(f"To predict similar {result.episode.event_type} events, monitor:")
            for p in valid_precursors[:3]:
                print(f"   • {p.variable} - signals appear ~{p.lag_days} days before event")
            print(f"\nEarliest warning possible: {result.max_lead_time} days in advance")


def main():
    """Run historical episode analysis."""
    print("\n" + "="*70)
    print("🔮 HISTORICAL EPISODE PRECURSOR ANALYSIS")
    print("="*70)
    print("Finding signals that PRECEDE documented events")
    print("Goal: Identify patterns for PREDICTION\n")
    
    analyzer = HistoricalEpisodeAnalyzer()
    
    # List available episodes
    print("\n📋 Available Historical Episodes:")
    for ep in analyzer.list_episodes():
        print(f"   • {ep['id']}: {ep['name']} ({ep['date']})")
    
    # Analyze each episode
    results = []
    
    for episode in EPISODES[:2]:  # Analyze first two for demo
        try:
            result = analyzer.analyze_episode(episode.id)
            results.append(result)
        except Exception as e:
            print(f"❌ Error analyzing {episode.id}: {e}")
    
    # Cross-episode patterns
    if len(results) > 1:
        print("\n" + "="*70)
        print("🔗 CROSS-EPISODE PATTERN ANALYSIS")
        print("="*70)
        
        # Find common precursors
        all_variables = set()
        for r in results:
            for p in r.precursors:
                if p.physics_validated:
                    all_variables.add(p.variable)
        
        print(f"\nVariables appearing as precursors in multiple events:")
        for var in all_variables:
            appearances = sum(1 for r in results 
                           for p in r.precursors 
                           if p.variable == var and p.physics_validated)
            if appearances > 1:
                print(f"   • {var} (appears in {appearances} events)")
    
    print("\n✅ Historical analysis complete!")
    print("Next: Use these patterns for real-time prediction monitoring")


if __name__ == "__main__":
    main()
