"""
🎵 Atmospheric Data Sonification
================================

Convert atmospheric data (pressure, velocity, sea level) to audible sound.

Based on the hypothesis that:
- Atmospheric phenomena follow 1/f^β spectral patterns (like brown/pink noise)
- These patterns can be mapped to human audible frequencies
- "Listening" to data may reveal patterns invisible to visual inspection

Research basis:
- Tornado infrasound detection (0.5-20 Hz, detectable 1hr before formation)
- Penn State hurricane sonification project
- Kolmogorov turbulence theory (-5/3 power law)

Usage:
    from surge_shazam.analysis.sonification import sonify_pressure_series

    # Load pressure data
    audio = sonify_pressure_series(
        pressure_pa=ds['msl'].values,  # Mean sea level pressure
        times=ds['time'].values,
        speedup=500,  # 500x faster → brings 1 Hz to 500 Hz
        output_path="storm_sound.wav"
    )
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 44100  # CD quality
NYQUIST = SAMPLE_RATE // 2


def normalize(data: np.ndarray, min_val: float = -1, max_val: float = 1) -> np.ndarray:
    """Normalize data to range [min_val, max_val]."""
    d_min, d_max = np.nanmin(data), np.nanmax(data)
    if d_max - d_min < 1e-10:
        return np.zeros_like(data)
    return min_val + (max_val - min_val) * (data - d_min) / (d_max - d_min)


def resample_to_audio(
    data: np.ndarray,
    original_dt_seconds: float,
    speedup: float = 500,
    target_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Resample time series data to audio sample rate.

    Args:
        data: Input time series
        original_dt_seconds: Time step of original data in seconds
        speedup: Time speedup factor (500 = 500x faster)
        target_rate: Target audio sample rate

    Returns:
        Resampled data at audio rate
    """
    # Original effective rate after speedup
    original_rate = speedup / original_dt_seconds  # samples per second

    # Number of output samples
    duration_seconds = len(data) * original_dt_seconds / speedup
    n_output = int(duration_seconds * target_rate)

    # Interpolate
    x_original = np.linspace(0, 1, len(data))
    x_target = np.linspace(0, 1, n_output)

    return np.interp(x_target, x_original, data)


def pressure_to_frequency(
    pressure_norm: np.ndarray,
    base_freq: float = 220,  # A3
    freq_range: float = 0.5  # ±50%
) -> np.ndarray:
    """
    Map normalized pressure to frequency.

    Lower pressure (storm) → lower pitch (ominous)
    Higher pressure (fair weather) → higher pitch (bright)
    """
    return base_freq * (1 + pressure_norm * freq_range)


def velocity_to_amplitude(
    velocity_norm: np.ndarray,
    min_amp: float = 0.1,
    max_amp: float = 0.8
) -> np.ndarray:
    """
    Map normalized velocity to amplitude.

    Higher wind → louder sound
    """
    return min_amp + velocity_norm * (max_amp - min_amp)


def synthesize_tone(
    frequency: np.ndarray,
    amplitude: np.ndarray,
    sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Synthesize audio from frequency and amplitude envelopes.

    Uses FM synthesis for richer timbre.
    """
    n_samples = len(frequency)
    t = np.arange(n_samples) / sample_rate

    # Phase accumulation for smooth frequency changes
    phase = np.cumsum(2 * np.pi * frequency / sample_rate)

    # Basic carrier
    carrier = np.sin(phase)

    # Add harmonics for richer sound (like wind)
    harmonics = (
        0.5 * np.sin(2 * phase) +
        0.25 * np.sin(3 * phase) +
        0.125 * np.sin(4 * phase)
    )

    # Mix
    audio = amplitude * (0.6 * carrier + 0.4 * harmonics)

    # Add subtle noise for texture (wind-like)
    noise = np.random.randn(n_samples) * 0.05 * amplitude
    audio += noise

    return np.clip(audio, -1, 1)


def sonify_pressure_series(
    pressure_pa: np.ndarray,
    times: Optional[np.ndarray] = None,
    dt_seconds: float = 3600,  # Default: hourly data
    speedup: float = 500,
    base_freq: float = 220,
    output_path: Optional[Union[str, Path]] = None,
    add_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert pressure time series to audible sound.

    Args:
        pressure_pa: Pressure in Pascals (or any consistent unit)
        times: Optional datetime array (used to compute dt if provided)
        dt_seconds: Time step in seconds (default 3600 = hourly)
        speedup: Time speedup factor (500 → 1 Hz becomes 500 Hz)
        base_freq: Base frequency in Hz
        output_path: Optional path to save WAV file
        add_velocity: Optional velocity array for amplitude modulation

    Returns:
        Audio signal as numpy array (float32, -1 to 1)

    Example:
        >>> import xarray as xr
        >>> ds = xr.open_dataset("era5_pressure.nc")
        >>> audio = sonify_pressure_series(
        ...     ds['msl'].values.flatten(),
        ...     speedup=500,
        ...     output_path="storm.wav"
        ... )
    """
    logger.info(f"Sonifying {len(pressure_pa)} pressure samples (speedup={speedup}x)")

    # Compute dt from times if provided
    if times is not None and len(times) > 1:
        if hasattr(times[0], 'astype'):
            # numpy datetime64
            dt_ns = (times[1] - times[0]).astype('timedelta64[s]').astype(float)
            dt_seconds = float(dt_ns)
        else:
            dt_seconds = (times[1] - times[0]).total_seconds()

    # Handle NaN values
    pressure_clean = np.nan_to_num(pressure_pa, nan=np.nanmean(pressure_pa))

    # Normalize pressure (-1 to 1)
    p_norm = normalize(pressure_clean, -1, 1)

    # Resample to audio rate
    p_audio = resample_to_audio(p_norm, dt_seconds, speedup)

    # Map to frequency
    freq = pressure_to_frequency(p_audio, base_freq)

    # Map velocity to amplitude (or use default)
    if add_velocity is not None:
        v_norm = normalize(np.nan_to_num(add_velocity), 0, 1)
        v_audio = resample_to_audio(v_norm, dt_seconds, speedup)
        amp = velocity_to_amplitude(v_audio)
    else:
        # Use pressure derivative for dynamics
        p_deriv = np.abs(np.gradient(p_audio))
        p_deriv_norm = normalize(p_deriv, 0, 1)
        amp = velocity_to_amplitude(p_deriv_norm, min_amp=0.3, max_amp=0.9)

    # Synthesize
    audio = synthesize_tone(freq, amp)

    # Duration info
    duration_sec = len(audio) / SAMPLE_RATE
    logger.info(f"Generated {duration_sec:.1f}s audio ({len(audio)} samples)")

    # Save if path provided
    if output_path:
        save_wav(audio, output_path)

    return audio.astype(np.float32)


def save_wav(audio: np.ndarray, path: Union[str, Path], sample_rate: int = SAMPLE_RATE):
    """Save audio array to WAV file."""
    try:
        import scipy.io.wavfile as wav
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(str(path), sample_rate, audio_int16)
        logger.info(f"Saved audio to {path}")
    except ImportError:
        logger.warning("scipy not installed - cannot save WAV")
        # Try wave module as fallback
        import wave
        import struct

        with wave.open(str(path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            audio_int16 = (audio * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        logger.info(f"Saved audio to {path} (using wave module)")


def compute_spectral_slope(
    data: np.ndarray,
    dt_seconds: float = 3600
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the spectral slope (β in 1/f^β) of a time series.

    Args:
        data: Time series data
        dt_seconds: Time step in seconds

    Returns:
        beta: Spectral slope (0=white, 1=pink, 2=brown)
        freqs: Frequency array
        psd: Power spectral density
    """
    from scipy import signal

    # Compute PSD
    freqs, psd = signal.welch(data, fs=1/dt_seconds, nperseg=min(256, len(data)//4))

    # Fit log-log slope (excluding DC component)
    valid = (freqs > 0) & (psd > 0)
    log_f = np.log10(freqs[valid])
    log_psd = np.log10(psd[valid])

    # Linear fit
    slope, intercept = np.polyfit(log_f, log_psd, 1)
    beta = -slope  # 1/f^β means negative slope in log-log

    return beta, freqs, psd


def classify_noise_color(beta: float) -> str:
    """Classify noise color based on spectral slope."""
    if beta < 0.5:
        return "white (β≈0)"
    elif beta < 1.5:
        return "pink (β≈1)"
    elif beta < 2.5:
        return "brown/red (β≈2)"
    else:
        return "black (β>2)"


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Atmospheric Sonification Test ===\n")

    # Generate synthetic storm pressure data
    print("Generating synthetic storm pressure data...")

    # Time: 72 hours, hourly
    n_hours = 72
    t = np.arange(n_hours)

    # Base pressure with storm passage
    base_pressure = 101325  # Pa (1 atm)

    # Storm: gradual drop then recovery
    storm_center = 36  # Hour 36
    storm_depth = 3000  # 30 hPa drop
    storm_width = 12  # Hours

    pressure = base_pressure - storm_depth * np.exp(-((t - storm_center) ** 2) / (2 * storm_width ** 2))

    # Add realistic fluctuations (brown noise-like)
    # Brown noise = cumulative sum of white noise
    white_noise = np.random.randn(n_hours) * 50
    brown_component = np.cumsum(white_noise) / 10
    pressure += brown_component

    # Compute spectral slope
    print("\n=== Spectral Analysis ===")
    try:
        beta, freqs, psd = compute_spectral_slope(pressure, dt_seconds=3600)
        color = classify_noise_color(beta)
        print(f"Spectral slope β = {beta:.2f}")
        print(f"Noise color: {color}")
    except ImportError:
        print("scipy not available for spectral analysis")

    # Sonify
    print("\n=== Sonification ===")
    print(f"Input: {n_hours} hours of pressure data")
    print(f"Speedup: 500x (1 Hz → 500 Hz)")

    audio = sonify_pressure_series(
        pressure,
        dt_seconds=3600,
        speedup=500,
        output_path="test_storm.wav"
    )

    print(f"Output: {len(audio)/SAMPLE_RATE:.1f} seconds of audio")
    print(f"\nListen to 'test_storm.wav' to hear the storm pass!")
    print("- Lower pitch = lower pressure (storm center)")
    print("- Volume variations = pressure fluctuations")
