"""
Fingerprint Visualizer - See the Storm Pattern
===============================================

Visualizza il fingerprint di un ciclone/tornado come Shazam visualizza le canzoni:
- Spettrogramma (tempo × frequenza × intensità)
- Picchi locali (la "costellazione")
- Pattern geometrico unico

Come il riconoscimento facciale ha layer che estraggono features,
qui estraiamo features spettrali dalla pressione atmosferica.

Usage:
    from surge_shazam.fingerprinting.visualizer import StormFingerprint

    # From pressure time series
    fp = StormFingerprint.from_pressure(pressure_data, times)
    fp.plot()
    fp.plot_constellation()

    # Compare two storms
    similarity = fp1.compare(fp2)
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Check dependencies
try:
    from scipy import signal
    from scipy.ndimage import maximum_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed - install with: pip install scipy")

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed - install with: pip install matplotlib")


@dataclass
class ConstellationPoint:
    """Un picco nella mappa delle costellazioni."""
    time_idx: int
    freq_idx: int
    time_sec: float
    frequency: float
    amplitude: float

    def __repr__(self):
        return f"Peak(t={self.time_sec:.1f}s, f={self.frequency:.4f}Hz, A={self.amplitude:.2f})"


@dataclass
class StormFingerprint:
    """
    Il fingerprint spettrale di una tempesta.

    Come Shazam identifica canzoni dai picchi spettrali,
    identifichiamo pattern di tempesta dai picchi di pressione/vento.
    """

    # Core data
    spectrogram: np.ndarray          # Time × Frequency power matrix
    frequencies: np.ndarray          # Frequency axis (Hz)
    times: np.ndarray                # Time axis (seconds)

    # Constellation (peaks)
    peaks: List[ConstellationPoint] = field(default_factory=list)

    # Metadata
    event_name: str = "unknown"
    source_variable: str = "pressure"
    dt_seconds: float = 3600.0       # Original time step
    n_original_samples: int = 0

    # Spectral characteristics
    spectral_slope: float = 0.0      # β in 1/f^β
    dominant_frequency: float = 0.0
    total_energy: float = 0.0

    @classmethod
    def from_pressure(
        cls,
        pressure: np.ndarray,
        times: Optional[np.ndarray] = None,
        dt_seconds: float = 3600.0,
        event_name: str = "storm",
        nperseg: int = 64,
        noverlap: Optional[int] = None,
        peak_threshold: float = 0.5,
        neighborhood_size: int = 10
    ) -> "StormFingerprint":
        """
        Crea fingerprint da serie temporale di pressione.

        Args:
            pressure: Pressure time series (Pa or hPa)
            times: Optional time array
            dt_seconds: Time step in seconds (default: 1 hour)
            event_name: Name for this event
            nperseg: Samples per STFT segment
            noverlap: Overlap between segments
            peak_threshold: Threshold for peak detection (0-1, relative to max)
            neighborhood_size: Size for local maximum detection

        Returns:
            StormFingerprint with spectrogram and constellation
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required: pip install scipy")

        # Clean data
        pressure_clean = np.nan_to_num(pressure, nan=np.nanmean(pressure))

        # Compute time array if not provided
        if times is None:
            times = np.arange(len(pressure_clean)) * dt_seconds

        # Compute sampling frequency
        fs = 1.0 / dt_seconds  # Hz

        # Adjust nperseg if too large
        if nperseg > len(pressure_clean) // 2:
            nperseg = len(pressure_clean) // 4
            logger.info(f"Adjusted nperseg to {nperseg}")

        if noverlap is None:
            noverlap = nperseg // 2

        # Compute spectrogram (STFT)
        f, t, Sxx = signal.spectrogram(
            pressure_clean,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='spectrum'
        )

        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Find peaks (local maxima in spectrogram)
        peaks = cls._find_peaks(
            Sxx_db, f, t,
            threshold=peak_threshold,
            neighborhood_size=neighborhood_size
        )

        # Compute spectral characteristics
        spectral_slope = cls._compute_spectral_slope(f, Sxx)

        # Find dominant frequency
        mean_power = np.mean(Sxx, axis=1)
        if len(f) > 0 and len(mean_power) > 0:
            dominant_freq = f[np.argmax(mean_power)]
        else:
            dominant_freq = 0.0

        return cls(
            spectrogram=Sxx_db,
            frequencies=f,
            times=t,
            peaks=peaks,
            event_name=event_name,
            source_variable="pressure",
            dt_seconds=dt_seconds,
            n_original_samples=len(pressure),
            spectral_slope=spectral_slope,
            dominant_frequency=dominant_freq,
            total_energy=np.sum(Sxx)
        )

    @staticmethod
    def _find_peaks(
        Sxx_db: np.ndarray,
        freqs: np.ndarray,
        times: np.ndarray,
        threshold: float = 0.5,
        neighborhood_size: int = 10
    ) -> List[ConstellationPoint]:
        """
        Trova i picchi locali nello spettrogramma.

        Come Shazam: cerca i massimi locali che si stagliano sopra il rumore.
        """
        from scipy.ndimage import maximum_filter

        # Local maximum filter
        local_max = maximum_filter(Sxx_db, size=neighborhood_size)

        # Points that are local maxima
        is_peak = (Sxx_db == local_max)

        # Threshold relative to max
        max_val = np.max(Sxx_db)
        min_val = np.min(Sxx_db)
        thresh_val = min_val + threshold * (max_val - min_val)

        is_above_threshold = Sxx_db > thresh_val

        # Combine conditions
        peak_mask = is_peak & is_above_threshold

        # Extract peak coordinates
        freq_indices, time_indices = np.where(peak_mask)

        peaks = []
        for fi, ti in zip(freq_indices, time_indices):
            peaks.append(ConstellationPoint(
                time_idx=ti,
                freq_idx=fi,
                time_sec=times[ti] if ti < len(times) else 0,
                frequency=freqs[fi] if fi < len(freqs) else 0,
                amplitude=Sxx_db[fi, ti]
            ))

        # Sort by amplitude (strongest first)
        peaks.sort(key=lambda p: -p.amplitude)

        logger.info(f"Found {len(peaks)} peaks in spectrogram")
        return peaks

    @staticmethod
    def _compute_spectral_slope(freqs: np.ndarray, Sxx: np.ndarray) -> float:
        """Compute β in 1/f^β power law."""
        # Average power spectrum
        mean_psd = np.mean(Sxx, axis=1)

        # Only use positive frequencies with positive power
        valid = (freqs > 0) & (mean_psd > 0)
        if np.sum(valid) < 2:
            return 0.0

        log_f = np.log10(freqs[valid])
        log_psd = np.log10(mean_psd[valid])

        # Linear fit
        try:
            slope, _ = np.polyfit(log_f, log_psd, 1)
            return -slope  # β = -slope
        except:
            return 0.0

    def plot(
        self,
        figsize: Tuple[int, int] = (14, 10),
        show_peaks: bool = True,
        cmap: str = "magma",
        save_path: Optional[str] = None
    ):
        """
        Visualizza il fingerprint completo.

        4 pannelli:
        1. Spettrogramma con picchi
        2. Costellazione (solo picchi)
        3. Spettro medio (1/f pattern)
        4. Serie originale ricostruita
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required: pip install matplotlib")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"Storm Fingerprint: {self.event_name}", fontsize=14, fontweight='bold')

        # 1. Spectrogram with peaks
        ax1 = axes[0, 0]
        im = ax1.pcolormesh(
            self.times / 3600,  # Convert to hours
            self.frequencies * 3600,  # Convert to cycles/hour
            self.spectrogram,
            shading='auto',
            cmap=cmap
        )
        if show_peaks and self.peaks:
            peak_t = [p.time_sec / 3600 for p in self.peaks[:50]]  # Top 50 peaks
            peak_f = [p.frequency * 3600 for p in self.peaks[:50]]
            ax1.scatter(peak_t, peak_f, c='cyan', s=30, marker='o',
                       edgecolors='white', linewidths=0.5, alpha=0.8)
        ax1.set_xlabel("Time (hours)")
        ax1.set_ylabel("Frequency (cycles/hour)")
        ax1.set_title("Spectrogram + Peaks")
        plt.colorbar(im, ax=ax1, label="Power (dB)")

        # 2. Constellation map (peaks only)
        ax2 = axes[0, 1]
        ax2.set_facecolor('black')
        if self.peaks:
            peak_t = [p.time_sec / 3600 for p in self.peaks]
            peak_f = [p.frequency * 3600 for p in self.peaks]
            peak_a = [p.amplitude for p in self.peaks]

            # Normalize amplitude for sizing
            a_min, a_max = min(peak_a), max(peak_a)
            if a_max > a_min:
                sizes = [20 + 80 * (a - a_min) / (a_max - a_min) for a in peak_a]
            else:
                sizes = [50] * len(peak_a)

            ax2.scatter(peak_t, peak_f, c=peak_a, s=sizes,
                       cmap='plasma', alpha=0.9)

            # Draw constellation lines (connect nearby peaks)
            self._draw_constellation_lines(ax2, self.peaks[:30])

        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Frequency (cycles/hour)")
        ax2.set_title(f"Constellation Map ({len(self.peaks)} peaks)")
        ax2.set_xlim(self.times[0]/3600, self.times[-1]/3600)
        ax2.set_ylim(self.frequencies[0]*3600, self.frequencies[-1]*3600)

        # 3. Mean power spectrum (shows 1/f pattern)
        ax3 = axes[1, 0]
        mean_psd = np.mean(10**(self.spectrogram/10), axis=1)  # Convert back from dB
        valid = self.frequencies > 0
        ax3.loglog(self.frequencies[valid] * 3600, mean_psd[valid], 'b-', linewidth=2)
        ax3.set_xlabel("Frequency (cycles/hour)")
        ax3.set_ylabel("Power Spectral Density")
        ax3.set_title(f"Mean Power Spectrum (β ≈ {self.spectral_slope:.2f})")
        ax3.grid(True, alpha=0.3)

        # Add reference lines for noise colors
        if len(self.frequencies[valid]) > 0:
            f_ref = self.frequencies[valid]
            p_ref = mean_psd[valid][0]
            f0 = f_ref[0]
            ax3.loglog(f_ref * 3600, p_ref * (f0/f_ref)**1, 'g--', alpha=0.5, label='Pink (β=1)')
            ax3.loglog(f_ref * 3600, p_ref * (f0/f_ref)**2, 'r--', alpha=0.5, label='Brown (β=2)')
            ax3.loglog(f_ref * 3600, p_ref * (f0/f_ref)**(5/3), 'm--', alpha=0.5, label='Kolmogorov (β=5/3)')
            ax3.legend(fontsize=8)

        # 4. Summary info
        ax4 = axes[1, 1]
        ax4.axis('off')

        info_text = f"""
        STORM FINGERPRINT SUMMARY
        ═════════════════════════

        Event: {self.event_name}

        SPECTRAL CHARACTERISTICS:
        • Spectral slope β = {self.spectral_slope:.2f}
        • Dominant frequency = {self.dominant_frequency*3600:.3f} cycles/hour
        • Total energy = {self.total_energy:.2e}

        CONSTELLATION:
        • Total peaks: {len(self.peaks)}
        • Top peak: {self.peaks[0] if self.peaks else 'N/A'}

        NOISE COLOR:
        • β ≈ 0: White noise
        • β ≈ 1: Pink noise
        • β ≈ 5/3: Kolmogorov turbulence
        • β ≈ 2: Brown noise

        This storm: {"White" if self.spectral_slope < 0.5 else
                    "Pink" if self.spectral_slope < 1.5 else
                    "Kolmogorov" if self.spectral_slope < 1.85 else
                    "Brown" if self.spectral_slope < 2.5 else "Black"} noise character
        """
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes,
                fontsize=10, fontfamily='monospace', verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved fingerprint to {save_path}")

        plt.show()
        return fig

    def _draw_constellation_lines(self, ax, peaks: List[ConstellationPoint], max_connections: int = 50):
        """Draw lines connecting nearby peaks (like star constellations)."""
        if len(peaks) < 2:
            return

        connections = 0
        for i, p1 in enumerate(peaks):
            for p2 in peaks[i+1:i+4]:  # Connect to next 3 peaks
                t1, f1 = p1.time_sec / 3600, p1.frequency * 3600
                t2, f2 = p2.time_sec / 3600, p2.frequency * 3600
                ax.plot([t1, t2], [f1, f2], 'w-', alpha=0.3, linewidth=0.5)
                connections += 1
                if connections >= max_connections:
                    return

    def plot_constellation(
        self,
        figsize: Tuple[int, int] = (12, 8),
        top_n: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Plot only the constellation map (like Shazam's fingerprint).
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required")

        fig, ax = plt.subplots(figsize=figsize, facecolor='black')
        ax.set_facecolor('black')

        if self.peaks:
            peaks = self.peaks[:top_n]
            peak_t = [p.time_sec / 3600 for p in peaks]
            peak_f = [p.frequency * 3600 for p in peaks]
            peak_a = [p.amplitude for p in peaks]

            # Size based on amplitude
            a_min, a_max = min(peak_a), max(peak_a)
            if a_max > a_min:
                sizes = [30 + 150 * (a - a_min) / (a_max - a_min) for a in peak_a]
            else:
                sizes = [80] * len(peak_a)

            # Plot peaks as stars
            scatter = ax.scatter(peak_t, peak_f, c=peak_a, s=sizes,
                                cmap='plasma', alpha=0.9, edgecolors='white',
                                linewidths=0.5)

            # Draw constellation lines
            self._draw_constellation_lines(ax, peaks)

            plt.colorbar(scatter, ax=ax, label="Power (dB)")

        ax.set_xlabel("Time (hours)", color='white')
        ax.set_ylabel("Frequency (cycles/hour)", color='white')
        ax.set_title(f"🌀 {self.event_name} - Constellation Fingerprint",
                    color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')

        for spine in ax.spines.values():
            spine.set_color('white')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
            logger.info(f"Saved constellation to {save_path}")

        plt.show()
        return fig

    def get_hash(self) -> str:
        """
        Genera un hash unico per questo fingerprint.

        Come Shazam usa hash per matching veloce nel database.
        """
        # Use peak positions to create a unique hash
        if not self.peaks:
            return "empty"

        # Quantize peaks to create stable hash
        peak_data = []
        for p in self.peaks[:20]:  # Top 20 peaks
            # Quantize time and frequency
            t_q = int(p.time_sec / 60)  # 1-minute bins
            f_q = int(p.frequency * 10000)  # 0.0001 Hz bins
            peak_data.append((t_q, f_q))

        # Create hash from peak positions
        import hashlib
        hash_input = str(sorted(peak_data)).encode()
        return hashlib.md5(hash_input).hexdigest()[:12]

    def compare(self, other: "StormFingerprint") -> float:
        """
        Confronta due fingerprint.

        Returns:
            Similarity score 0-1 (1 = identical pattern)
        """
        if not self.peaks or not other.peaks:
            return 0.0

        # Method 1: Compare spectral slopes
        slope_diff = abs(self.spectral_slope - other.spectral_slope)
        slope_sim = max(0, 1 - slope_diff / 2)

        # Method 2: Compare peak frequency distributions
        self_freqs = np.array([p.frequency for p in self.peaks[:20]])
        other_freqs = np.array([p.frequency for p in other.peaks[:20]])

        if len(self_freqs) > 0 and len(other_freqs) > 0:
            # Normalize and compare histograms
            self_hist, _ = np.histogram(self_freqs, bins=10, density=True)
            other_hist, _ = np.histogram(other_freqs, bins=10, density=True)

            # Cosine similarity
            norm1 = np.linalg.norm(self_hist)
            norm2 = np.linalg.norm(other_hist)
            if norm1 > 0 and norm2 > 0:
                freq_sim = np.dot(self_hist, other_hist) / (norm1 * norm2)
            else:
                freq_sim = 0.0
        else:
            freq_sim = 0.0

        # Combine
        similarity = 0.4 * slope_sim + 0.6 * freq_sim
        return float(similarity)


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_synthetic_storm():
    """Demo with synthetic storm data."""
    print("="*60)
    print("STORM FINGERPRINT DEMO")
    print("="*60)

    # Generate synthetic storm pressure data
    np.random.seed(42)

    n_hours = 120  # 5 days
    t = np.arange(n_hours)
    dt = 3600  # 1 hour

    # Base pressure
    base_p = 101325  # Pa (1 atm)

    # Storm passage (pressure drop)
    storm_center = 60
    storm_depth = 3500  # 35 hPa drop (strong storm)
    storm_width = 15
    storm = -storm_depth * np.exp(-((t - storm_center)**2) / (2 * storm_width**2))

    # Add oscillations (seiche-like, ~22 hour period for Adriatic)
    seiche_period = 22  # hours
    seiche = 500 * np.sin(2 * np.pi * t / seiche_period)

    # Add turbulence (brown noise)
    white = np.random.randn(n_hours)
    brown = np.cumsum(white) * 30

    # Combine
    pressure = base_p + storm + seiche + brown

    print(f"\nGenerated {n_hours} hours of synthetic storm data")
    print(f"Storm center: hour {storm_center}")
    print(f"Pressure drop: {storm_depth/100:.1f} hPa")

    # Create fingerprint
    print("\nExtracting fingerprint...")
    fp = StormFingerprint.from_pressure(
        pressure,
        dt_seconds=dt,
        event_name="Synthetic Storm (Sicily-like)",
        nperseg=32,
        peak_threshold=0.4
    )

    print(f"\n FINGERPRINT EXTRACTED:")
    print(f"   Spectral slope β = {fp.spectral_slope:.2f}")
    print(f"   Dominant frequency = {fp.dominant_frequency*3600:.3f} cycles/hour")
    print(f"   Number of peaks = {len(fp.peaks)}")
    print(f"   Hash = {fp.get_hash()}")

    # Classify noise color
    if fp.spectral_slope < 0.5:
        color = "WHITE (β≈0)"
    elif fp.spectral_slope < 1.5:
        color = "PINK (β≈1)"
    elif fp.spectral_slope < 1.85:
        color = "KOLMOGOROV (β≈5/3)"
    elif fp.spectral_slope < 2.5:
        color = "BROWN (β≈2)"
    else:
        color = "BLACK (β>2)"

    print(f"   Noise color: {color}")

    # Plot
    print("\nGenerating visualization...")
    try:
        fp.plot(save_path="storm_fingerprint_demo.png")
        print("\n Saved to: storm_fingerprint_demo.png")
    except Exception as e:
        print(f"Could not display plot: {e}")
        print("(Run in environment with display)")

    return fp


if __name__ == "__main__":
    demo_synthetic_storm()
