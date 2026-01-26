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

SIGN CONVENTION:
    - Gates are ordered WEST to EAST (or SOUTH to NORTH for meridional gates)
    - Normal vector points "to the left" when walking along the gate
    - For Arctic gates: positive v_perp = flow NORTHWARD (into Arctic)
    - For Bering Strait: positive v_perp = flow NORTHWARD (into Arctic)
    - For Fram/Davis: positive v_perp = flow SOUTHWARD (out of Arctic)
    
    The sign can be flipped per-gate using the `invert_sign` parameter
    to ensure positive = inflow to Arctic.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Constants
SVERDRUP = 1e6  # 1 Sv = 10^6 m³/s

# Sign convention for each gate (True = flip sign so positive = Arctic inflow)
# Fram Strait, Davis Strait: outflow is southward, so DON'T flip
# Bering Strait: inflow is northward, so DON'T flip
GATE_SIGN_CONVENTION = {
    "fram_strait": False,  # positive = southward (out of Arctic) - DON'T flip
    "davis_strait": False,  # positive = southward (out of Arctic) - DON'T flip  
    "bering_strait": True,  # positive = northward (into Arctic) - already correct
    "denmark_strait": False,  # positive = southward
    "barents_sea_opening": True,  # positive = northward (into Barents)
    # Add more as needed...
}


def get_velocity_sign_convention(gate_name: str) -> Tuple[int, str]:
    """
    Get the sign convention for a gate.
    
    Args:
        gate_name: Name of the gate (e.g., "fram_strait", "bering_strait")
        
    Returns:
        sign: 1 or -1 to multiply v_perp
        direction_label: String describing positive direction (e.g., "Northward (into Arctic)")
    """
    # Normalize gate name
    gate_key = gate_name.lower().replace(" ", "_").replace("-", "_")
    
    # Check for partial matches
    for key in GATE_SIGN_CONVENTION:
        if key in gate_key:
            should_flip = GATE_SIGN_CONVENTION[key]
            if should_flip:
                return 1, "Northward (into Arctic)"
            else:
                return 1, "Southward (out of Arctic)"
    
    # Default: assume positive = northward component dominates
    logger.warning(f"No sign convention defined for gate '{gate_name}', using default (positive=northward)")
    return 1, "Northward"


def compute_normal_direction(gate_lon: np.ndarray, gate_lat: np.ndarray) -> str:
    """
    Determine the dominant direction of the gate normal.
    
    Returns:
        "N", "S", "E", or "W" for the direction the normal points
    """
    # Compute mean normal angle
    theta = compute_gate_angles(gate_lon, gate_lat)
    mean_theta = np.mean(theta)
    
    # Convert to compass direction
    # theta is angle from North, clockwise
    # 0° = N, 90° = E, 180° = S, 270° = W
    angle_deg = np.rad2deg(mean_theta) % 360
    
    if 315 <= angle_deg or angle_deg < 45:
        return "N"
    elif 45 <= angle_deg < 135:
        return "E"
    elif 135 <= angle_deg < 225:
        return "S"
    else:
        return "W"


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
    gate_lat: np.ndarray,
    gate_name: Optional[str] = None,
    return_info: bool = False
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
        gate_name: Optional gate name for logging direction convention
        return_info: If True, also return direction info dict
        
    Returns:
        v_perp: Perpendicular velocity (m/s), same shape as ugos
        
    Sign Convention:
        The normal is computed as 90° counterclockwise from gate direction.
        For a gate going West→East: normal points NORTH
        For a gate going South→North: normal points WEST
        
        Positive v_perp = flow in the direction of the normal
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
    
    # Log direction info
    if gate_name:
        normal_dir = compute_normal_direction(gate_lon, gate_lat)
        logger.info(f"Gate '{gate_name}': normal points {normal_dir}, positive v_perp = flow {normal_dir}")
    
    if return_info:
        normal_dir = compute_normal_direction(gate_lon, gate_lat)
        info = {
            "normal_direction": normal_dir,
            "positive_means": f"Flow toward {normal_dir}",
            "mean_v_perp": float(np.nanmean(v_perp)),
        }
        return v_perp, info
    
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


def compute_monthly_salt_flux_profile(
    x_km: np.ndarray,
    v_perp: np.ndarray,
    depth_profile: np.ndarray,
    time_array: np.ndarray,
    salinity: float = 34.8,
    density: float = 1027.0,
    bin_size_km: float = 5.0
) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute monthly climatology of salt flux along-gate profiles.
    
    Salt flux per unit width: f(x,t) = ρ × (S/1000) × v_perp(x,t) × H(x)
    
    Units: kg/(m·s) - salt flux per meter width of gate
    
    Args:
        x_km: Distance along gate (km)
        v_perp: Perpendicular velocity (m/s), shape (n_pts, n_time)
        depth_profile: Water depth (m), shape (n_pts,)
        time_array: Time values
        salinity: Salinity in PSU (default 34.8)
        density: Water density in kg/m³ (default 1027)
        bin_size_km: Spatial bin size
        
    Returns:
        Dict mapping month (1-12) to (bin_centers, bin_means, bin_stds)
        Values are in kg/(m·s) - salt flux per meter width
    """
    import pandas as pd
    
    # Compute local salt flux density: ρ × S/1000 × v × H
    # Shape: (n_pts, n_time)
    S_frac = salinity / 1000.0  # PSU to kg/kg
    salt_flux_local = density * S_frac * v_perp * depth_profile[:, np.newaxis]
    
    # Now compute monthly profiles
    time_pd = pd.to_datetime(time_array)
    months = time_pd.month
    
    result = {}
    
    for month in range(1, 13):
        month_mask = months == month
        if not np.any(month_mask):
            result[month] = (np.array([]), np.array([]), np.array([]))
            continue
        
        # Average over all time steps in this month (across years)
        flux_month = salt_flux_local[:, month_mask]  # (n_pts, n_time_month)
        flux_mean = np.nanmean(flux_month, axis=1)  # (n_pts,)
        
        # Bin spatially
        bin_centers, bin_means, bin_stds = bin_along_gate(x_km, flux_mean, bin_size_km)
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
