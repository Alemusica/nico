"""
Volume Transport Service - Calculate ocean volume transport through gates.

Transport formula: Q = ∫ v_perp(x) × h(x) dx

Where:
- v_perp: velocity perpendicular to gate (m/s)
- h: water depth (m)
- dx: along-gate distance (m)
- Q: volume transport (m³/s), typically reported in Sverdrup (1 Sv = 10⁶ m³/s)
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


def compute_perpendicular_velocity(
    ugos: np.ndarray,
    vgos: np.ndarray,
    gate_lon: np.ndarray,
    gate_lat: np.ndarray
) -> np.ndarray:
    """
    Compute velocity component perpendicular to the gate.
    
    For each gate segment, calculates the angle of the gate line
    and projects the velocity onto the perpendicular direction.
    
    Args:
        ugos: Eastward velocity (m/s), shape (n_pts, n_time)
        vgos: Northward velocity (m/s), shape (n_pts, n_time)
        gate_lon: Longitude of gate points
        gate_lat: Latitude of gate points
        
    Returns:
        v_perp: Perpendicular velocity (m/s), shape (n_pts, n_time)
        Positive = flow to the "right" of the gate direction
    """
    n_pts = len(gate_lon)
    n_time = ugos.shape[1] if ugos.ndim > 1 else 1
    
    # Compute gate segment angles
    # For each point, use forward difference (or backward at end)
    angles = np.zeros(n_pts)
    
    for i in range(n_pts):
        if i < n_pts - 1:
            dx = gate_lon[i + 1] - gate_lon[i]
            dy = gate_lat[i + 1] - gate_lat[i]
        else:
            dx = gate_lon[i] - gate_lon[i - 1]
            dy = gate_lat[i] - gate_lat[i - 1]
        
        # Account for latitude in dx (approximate)
        cos_lat = np.cos(np.deg2rad(gate_lat[i]))
        dx_corrected = dx * cos_lat
        
        # Angle of gate segment (radians)
        gate_angle = np.arctan2(dy, dx_corrected)
        
        # Perpendicular angle (90° to the right)
        angles[i] = gate_angle + np.pi / 2
    
    # Project velocity onto perpendicular direction
    # v_perp = u * cos(perp_angle) + v * sin(perp_angle)
    cos_perp = np.cos(angles)
    sin_perp = np.sin(angles)
    
    if ugos.ndim == 1:
        v_perp = ugos * cos_perp + vgos * sin_perp
    else:
        # Broadcast angles to (n_pts, n_time)
        cos_perp = cos_perp[:, np.newaxis]
        sin_perp = sin_perp[:, np.newaxis]
        v_perp = ugos * cos_perp + vgos * sin_perp
    
    return v_perp


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
