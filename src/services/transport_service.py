"""
Volume Transport Service - Calculate ocean volume transport through gates.

Transport formula: Q = ∫ v_perp(x) × h(x) dx

Where:
- v_perp: velocity perpendicular to gate (m/s)
- h: water depth (m)
- dx: along-gate distance (m)
- Q: volume transport (m³/s), typically reported in Sverdrup (1 Sv = 10⁶ m³/s)

Perpendicular Velocity Formula:
    v(θ) = v_N × cos(θ) + v_E × sin(θ)
    
Where:
- v_N = vgos (northward velocity)
- v_E = ugos (eastward velocity)
- θ = angle of gate normal with respect to North
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Constants
SVERDRUP = 1e6  # 1 Sv = 10^6 m³/s


@dataclass
class VolumeTransportResult:
    """Result of volume transport calculation."""
    # Per-time-step transport (Sv)
    transport_sv: np.ndarray  # Shape: (n_time,)
    time_array: np.ndarray  # Shape: (n_time,)
    
    # Monthly climatology (Sv)
    monthly_mean: np.ndarray  # Shape: (12,) - Jan to Dec
    monthly_std: np.ndarray  # Shape: (12,)
    
    # Along-gate profiles (mean over time)
    v_perp_profile: np.ndarray  # Shape: (n_pts,) - m/s
    depth_profile: np.ndarray  # Shape: (n_pts,) - m
    x_km: np.ndarray  # Shape: (n_pts,)
    
    # Statistics
    mean_transport_sv: float
    std_transport_sv: float
    min_transport_sv: float
    max_transport_sv: float
    
    # Metadata
    gate_name: str
    positive_direction: str  # e.g., "into Arctic" or "northward"


def compute_gate_angles(
    gate_lon: np.ndarray,
    gate_lat: np.ndarray
) -> np.ndarray:
    """
    Compute the angle θ of the gate normal (perpendicular) at each point.
    
    The angle is measured from North (0°) clockwise.
    For a gate going West to East, the normal points North (θ=0).
    For a gate going South to North, the normal points East (θ=90°).
    
    Args:
        gate_lon: Longitude of gate points
        gate_lat: Latitude of gate points
        
    Returns:
        theta: Angle of gate normal from North (radians), shape (n_pts,)
    """
    n_pts = len(gate_lon)
    theta = np.zeros(n_pts)
    
    for i in range(n_pts):
        # Use central difference for interior, forward/backward at edges
        if i == 0:
            dx = gate_lon[1] - gate_lon[0]
            dy = gate_lat[1] - gate_lat[0]
            lat_mid = gate_lat[0]
        elif i == n_pts - 1:
            dx = gate_lon[i] - gate_lon[i - 1]
            dy = gate_lat[i] - gate_lat[i - 1]
            lat_mid = gate_lat[i]
        else:
            dx = gate_lon[i + 1] - gate_lon[i - 1]
            dy = gate_lat[i + 1] - gate_lat[i - 1]
            lat_mid = gate_lat[i]
        
        # Correct dx for latitude (degrees to approximate meters ratio)
        cos_lat = np.cos(np.deg2rad(lat_mid))
        dx_corrected = dx * cos_lat
        
        # Gate tangent angle from East (standard atan2 convention)
        gate_angle = np.arctan2(dy, dx_corrected)
        
        # Normal angle = tangent + 90° (perpendicular, to the right of gate direction)
        # Convert to angle from North: θ_from_north = π/2 - angle_from_east
        normal_from_east = gate_angle + np.pi / 2
        theta[i] = np.pi / 2 - normal_from_east
    
    return theta


def compute_perpendicular_velocity(
    ugos: np.ndarray,
    vgos: np.ndarray,
    gate_lon: np.ndarray,
    gate_lat: np.ndarray
) -> np.ndarray:
    """
    Compute velocity component perpendicular to the gate.
    
    Formula: v(θ) = v_N × cos(θ) + v_E × sin(θ)
    
    Where:
    - v_N = vgos (northward velocity)
    - v_E = ugos (eastward velocity)
    - θ = angle of gate normal from North
    
    Args:
        ugos: Eastward velocity (m/s), shape (n_pts, n_time) or (n_pts,)
        vgos: Northward velocity (m/s), shape (n_pts, n_time) or (n_pts,)
        gate_lon: Longitude of gate points
        gate_lat: Latitude of gate points
        
    Returns:
        v_perp: Perpendicular velocity (m/s), same shape as ugos
        Positive = flow to the "right" of the gate direction
    """
    # Compute gate normal angles
    theta = compute_gate_angles(gate_lon, gate_lat)
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    if ugos.ndim == 1:
        # Single time step
        v_perp = vgos * cos_theta + ugos * sin_theta
    else:
        # Multiple time steps: broadcast angles to (n_pts, n_time)
        cos_theta = cos_theta[:, np.newaxis]
        sin_theta = sin_theta[:, np.newaxis]
        v_perp = vgos * cos_theta + ugos * sin_theta
    
    return v_perp


def compute_perpendicular_velocity_with_angles(
    ugos: np.ndarray,
    vgos: np.ndarray,
    theta: np.ndarray
) -> np.ndarray:
    """
    Compute perpendicular velocity using pre-computed angles.
    
    Args:
        ugos: Eastward velocity (m/s)
        vgos: Northward velocity (m/s)
        theta: Gate normal angles from North (radians)
        
    Returns:
        v_perp: Perpendicular velocity (m/s)
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    if ugos.ndim == 1:
        v_perp = vgos * cos_theta + ugos * sin_theta
    else:
        cos_theta = cos_theta[:, np.newaxis]
        sin_theta = sin_theta[:, np.newaxis]
        v_perp = vgos * cos_theta + ugos * sin_theta
    
    return v_perp


def bin_along_gate(
    x_km: np.ndarray,
    values: np.ndarray,
    bin_size_km: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin values along gate into spatial averages.
    
    Args:
        x_km: Distance along gate (km), shape (n_pts,)
        values: Values to bin, shape (n_pts,) or (n_pts, n_time)
        bin_size_km: Size of bins in km (default 5km)
        
    Returns:
        bin_centers: Center of each bin (km)
        bin_means: Mean value in each bin
        bin_stds: Std dev in each bin
    """
    x_min, x_max = x_km.min(), x_km.max()
    n_bins = max(1, int(np.ceil((x_max - x_min) / bin_size_km)))
    
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Digitize: which bin each point belongs to
    bin_idx = np.digitize(x_km, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    if values.ndim == 1:
        # Single array
        bin_means = np.zeros(n_bins)
        bin_stds = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = bin_idx == i
            if np.any(mask):
                vals = values[mask]
                valid = vals[np.isfinite(vals)]
                if len(valid) > 0:
                    bin_means[i] = np.mean(valid)
                    bin_stds[i] = np.std(valid) if len(valid) > 1 else 0
                else:
                    bin_means[i] = np.nan
                    bin_stds[i] = np.nan
            else:
                bin_means[i] = np.nan
                bin_stds[i] = np.nan
    else:
        # 2D array (n_pts, n_time)
        n_time = values.shape[1]
        bin_means = np.zeros((n_bins, n_time))
        bin_stds = np.zeros((n_bins, n_time))
        
        for i in range(n_bins):
            mask = bin_idx == i
            if np.any(mask):
                vals = values[mask, :]  # (n_pts_in_bin, n_time)
                bin_means[i, :] = np.nanmean(vals, axis=0)
                bin_stds[i, :] = np.nanstd(vals, axis=0)
            else:
                bin_means[i, :] = np.nan
                bin_stds[i, :] = np.nan
    
    return bin_centers, bin_means, bin_stds


def compute_monthly_along_gate_profile(
    x_km: np.ndarray,
    values: np.ndarray,
    time_array: np.ndarray,
    bin_size_km: float = 5.0
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute monthly climatology of along-gate profiles.
    
    For each month (1-12), averages all values from that month
    across all years, then bins spatially.
    
    Args:
        x_km: Distance along gate (km)
        values: Values array, shape (n_pts, n_time)
        time_array: Time values
        bin_size_km: Spatial bin size
        
    Returns:
        Dict mapping month (1-12) to (bin_centers, bin_means, bin_stds)
    """
    import pandas as pd
    
    time_pd = pd.to_datetime(time_array)
    months = time_pd.month
    
    result = {}
    
    for month in range(1, 13):
        month_mask = months == month
        if not np.any(month_mask):
            result[month] = (np.array([]), np.array([]), np.array([]))
            continue
        
        # Average over all time steps in this month (across years)
        values_month = values[:, month_mask]  # (n_pts, n_time_month)
        values_mean = np.nanmean(values_month, axis=1)  # (n_pts,)
        
        # Bin spatially
        bin_centers, bin_means, bin_stds = bin_along_gate(x_km, values_mean, bin_size_km)
        result[month] = (bin_centers, bin_means, bin_stds)
    
    return result


def compute_segment_widths(
    gate_lon: np.ndarray,
    gate_lat: np.ndarray,
    x_km: np.ndarray
) -> np.ndarray:
    """
    Compute width of each gate segment in meters.
    
    Uses the x_km array to get segment widths.
    
    Args:
        gate_lon: Longitude points
        gate_lat: Latitude points  
        x_km: Distance along gate (km)
        
    Returns:
        widths: Segment widths (m), shape (n_pts,)
    """
    n_pts = len(x_km)
    widths = np.zeros(n_pts)
    
    # Central difference for interior points, forward/backward at edges
    for i in range(n_pts):
        if i == 0:
            widths[i] = (x_km[1] - x_km[0]) * 1000  # km to m
        elif i == n_pts - 1:
            widths[i] = (x_km[i] - x_km[i - 1]) * 1000
        else:
            widths[i] = (x_km[i + 1] - x_km[i - 1]) / 2 * 1000
    
    return widths


def calculate_volume_transport(
    ugos_matrix: np.ndarray,
    vgos_matrix: np.ndarray,
    depth_profile: np.ndarray,
    gate_lon: np.ndarray,
    gate_lat: np.ndarray,
    x_km: np.ndarray,
    time_array: np.ndarray,
    gate_name: str = "Unknown"
) -> Optional[VolumeTransportResult]:
    """
    Calculate volume transport through a gate.
    
    Q(t) = Σᵢ v_perp(i,t) × h(i) × Δx(i)
    
    Args:
        ugos_matrix: Eastward velocity (m/s), shape (n_pts, n_time)
        vgos_matrix: Northward velocity (m/s), shape (n_pts, n_time)
        depth_profile: Water depth (m), shape (n_pts,)
        gate_lon: Longitude points
        gate_lat: Latitude points
        x_km: Distance along gate (km)
        time_array: Time values
        gate_name: Name of the gate
        
    Returns:
        VolumeTransportResult with transport values
    """
    import pandas as pd
    
    n_pts, n_time = ugos_matrix.shape
    
    logger.info(f"Calculating volume transport for {gate_name}")
    logger.info(f"  Points: {n_pts}, Time steps: {n_time}")
    
    # Step 1: Compute perpendicular velocity
    v_perp = compute_perpendicular_velocity(ugos_matrix, vgos_matrix, gate_lon, gate_lat)
    logger.info(f"  v_perp range: [{np.nanmin(v_perp):.3f}, {np.nanmax(v_perp):.3f}] m/s")
    
    # Step 2: Get segment widths
    widths = compute_segment_widths(gate_lon, gate_lat, x_km)
    logger.info(f"  Total gate width: {np.sum(widths)/1000:.1f} km")
    
    # Step 3: Calculate transport for each time step
    # Q(t) = Σᵢ v_perp(i,t) × h(i) × Δx(i)
    transport_m3s = np.zeros(n_time)
    
    for t in range(n_time):
        # Sum over all gate points
        # Handle NaN values
        v_t = v_perp[:, t]
        valid = ~np.isnan(v_t) & ~np.isnan(depth_profile)
        
        if np.any(valid):
            transport_m3s[t] = np.sum(v_t[valid] * depth_profile[valid] * widths[valid])
        else:
            transport_m3s[t] = np.nan
    
    # Convert to Sverdrup
    transport_sv = transport_m3s / SVERDRUP
    
    logger.info(f"  Transport range: [{np.nanmin(transport_sv):.2f}, {np.nanmax(transport_sv):.2f}] Sv")
    
    # Step 4: Monthly climatology
    time_pd = pd.to_datetime(time_array)
    months = time_pd.month
    
    monthly_mean = np.zeros(12)
    monthly_std = np.zeros(12)
    
    for m in range(1, 13):
        mask = months == m
        if np.any(mask):
            monthly_mean[m - 1] = np.nanmean(transport_sv[mask])
            monthly_std[m - 1] = np.nanstd(transport_sv[mask])
    
    # Step 5: Mean profiles
    v_perp_mean = np.nanmean(v_perp, axis=1)
    
    # Step 6: Statistics
    valid_transport = transport_sv[~np.isnan(transport_sv)]
    
    return VolumeTransportResult(
        transport_sv=transport_sv,
        time_array=time_array,
        monthly_mean=monthly_mean,
        monthly_std=monthly_std,
        v_perp_profile=v_perp_mean,
        depth_profile=depth_profile,
        x_km=x_km,
        mean_transport_sv=float(np.mean(valid_transport)),
        std_transport_sv=float(np.std(valid_transport)),
        min_transport_sv=float(np.min(valid_transport)),
        max_transport_sv=float(np.max(valid_transport)),
        gate_name=gate_name,
        positive_direction="perpendicular to gate (rightward)"
    )
