"""
Salt Flux Service
=================
Calculates salt transport through oceanic gates by combining:
- Geostrophic velocity from CMEMS L4 (DOT slope)
- Sea Surface Salinity/Density from CMEMS SSS dataset
- Bathymetry (capped at 250m)

Formula:
    F_salt(t) = Σᵢ ρ(sᵢ,t) × S(sᵢ,t) × u_normal(sᵢ,t) × H_eff(sᵢ) × Δs(i)

Where:
    - ρ = density from SSS dataset [kg/m³]
    - S = salinity from SSS dataset [PSU = g/kg = 10⁻³]
    - u_normal = velocity component normal to gate [m/s]
    - H_eff = min(bathymetry, 250m) [m]
    - Δs = segment length along gate [m]

Output units: kg/s of salt through the gate

Usage:
    flux_service = SaltFluxService()
    flux_data = flux_service.compute_salt_flux(
        cmems_data=cmems_pass_data,
        sss_data=sss_data,
        depth_array=bathymetry,
        depth_cap=250.0
    )
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple

from src.core.logging_config import get_logger, log_call

logger = get_logger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SaltFluxData:
    """
    Salt flux calculation results.
    
    Attributes:
        flux_series: Total salt flux [kg/s], shape (n_time,)
        flux_local: Local flux per meter [kg/(m·s)], shape (n_pts, n_time)
        time_array: Datetime array
        
        # Input summaries
        salinity_mean: Mean salinity along gate [PSU], shape (n_time,)
        density_mean: Mean density along gate [kg/m³], shape (n_time,)
        velocity_mean: Mean normal velocity [m/s], shape (n_time,)
        
        # Profiles
        salinity_profile: Mean S along gate [PSU], shape (n_pts,)
        density_profile: Mean ρ along gate [kg/m³], shape (n_pts,)
        depth_profile: Effective depth [m], shape (n_pts,)
        x_km: Distance along gate [km]
    """
    strait_name: str
    flux_series: np.ndarray      # [kg/s]
    flux_local: np.ndarray       # [kg/(m·s)]
    time_array: np.ndarray
    
    # Time series means
    salinity_mean: np.ndarray    # [PSU]
    density_mean: np.ndarray     # [kg/m³]
    velocity_mean: np.ndarray    # [m/s]
    
    # Spatial profiles (time-averaged)
    salinity_profile: np.ndarray
    density_profile: np.ndarray
    depth_profile: np.ndarray    # H_eff [m]
    x_km: np.ndarray
    
    # Gate geometry
    gate_lon_pts: np.ndarray
    gate_lat_pts: np.ndarray
    
    # Config
    depth_cap: float = 250.0
    
    # Stats
    flux_mean: float = 0.0       # Mean flux [kg/s]
    flux_std: float = 0.0        # Std flux [kg/s]
    flux_sv: float = 0.0         # Flux in Sv equivalent (for reference)


# ==============================================================================
# SALT FLUX SERVICE
# ==============================================================================

class SaltFluxService:
    """
    Service for computing salt transport through gates.
    
    Combines CMEMS L4 (velocity) + SSS (salinity, density) + bathymetry.
    """
    
    def __init__(self):
        logger.info("SaltFluxService initialized")
    
    @log_call(logger)
    def compute_salt_flux(
        self,
        cmems_data,  # CMEMSL4PassData
        sss_data,    # SSSData
        depth_array: Optional[np.ndarray] = None,
        depth_cap: float = 250.0,
        gate_angle: Optional[np.ndarray] = None,
    ) -> Optional[SaltFluxData]:
        """
        Compute salt flux through gate.
        
        Parameters
        ----------
        cmems_data : CMEMSL4PassData
            Contains ugos_matrix, vgos_matrix, x_km, gate geometry
        sss_data : SSSData
            Contains sos_matrix (salinity), dos_matrix (density)
        depth_array : np.ndarray, optional
            Bathymetry along gate [m]. If None, uses depth_cap everywhere.
        depth_cap : float
            Maximum integration depth [m]. Default 250m.
        gate_angle : np.ndarray, optional
            Angle of gate normal at each point [radians]. 
            If None, computed from gate geometry.
        
        Returns
        -------
        SaltFluxData or None
        """
        # Validate inputs
        if cmems_data is None or sss_data is None:
            logger.error("Missing CMEMS or SSS data")
            return None
        
        # Check time alignment
        if not self._check_time_alignment(cmems_data.time_array, sss_data.time_array):
            logger.warning("Time arrays not aligned, interpolating SSS to CMEMS times")
            sss_sos, sss_dos = self._interpolate_to_times(
                sss_data.sos_matrix, sss_data.dos_matrix,
                sss_data.time_array, cmems_data.time_array
            )
        else:
            sss_sos = sss_data.sos_matrix
            sss_dos = sss_data.dos_matrix
        
        n_pts = cmems_data.dot_matrix.shape[0]
        n_time = cmems_data.dot_matrix.shape[1]
        x_km = cmems_data.x_km
        
        logger.info(f"Computing salt flux: {n_pts} points × {n_time} times")
        
        # --- 1. VELOCITY NORMAL TO GATE ---
        u_normal = self._compute_normal_velocity(
            cmems_data.ugos_matrix,
            cmems_data.vgos_matrix,
            cmems_data.gate_lon_pts,
            cmems_data.gate_lat_pts,
            gate_angle,
        )
        
        if u_normal is None:
            logger.error("Failed to compute normal velocity")
            return None
        
        # --- 2. EFFECTIVE DEPTH ---
        if depth_array is not None:
            H_eff = np.minimum(np.abs(depth_array), depth_cap)
        else:
            H_eff = np.full(n_pts, depth_cap)
            logger.warning(f"No bathymetry provided, using constant depth = {depth_cap}m")
        
        # --- 3. SEGMENT LENGTHS ---
        ds_m = np.zeros(n_pts)
        ds_m[1:] = np.diff(x_km) * 1000  # km → m
        ds_m[0] = ds_m[1]  # First segment same as second
        
        # --- 4. LOCAL SALT FLUX ---
        # f_local(s,t) = ρ(s,t) × S(s,t) × u_n(s,t) × H_eff(s)
        # S in PSU = g/kg = 10⁻³, so multiply by 0.001 to get kg_salt/kg_water
        
        # Broadcast H_eff to (n_pts, n_time)
        H_eff_2d = H_eff[:, np.newaxis]
        
        # Local flux: [kg/m³] × [g/kg] × [m/s] × [m] = [kg/m³] × [10⁻³] × [m/s] × [m]
        # = [g/m²/s] = [10⁻³ kg/m²/s] ... wait, let's be careful
        #
        # Actually: ρ [kg/m³] × S [PSU ≈ g/kg = kg/1000kg] × u [m/s] × H [m]
        # = kg/m³ × (kg_salt/1000 kg_water) × m/s × m
        # = kg_salt / (1000 m² s)
        # 
        # So f_local has units kg_salt / (1000 m² s) per point
        # Multiply by ds [m] to get kg_salt / (1000 m s)
        # Sum over gate to get total flux
        
        # For simplicity: S in PSU ≈ kg_salt / (1000 kg_water)
        # ρ × S_psu / 1000 = kg_salt / m³
        # × u × H = kg_salt / m³ × m/s × m = kg_salt / (m·s) per unit width
        
        S_fraction = sss_sos / 1000.0  # PSU → kg/kg
        
        # Local flux density [kg/(m·s)] - flux per meter width of gate
        flux_local = sss_dos * S_fraction * u_normal * H_eff_2d
        
        # --- 5. TOTAL FLUX ---
        # F(t) = Σᵢ f_local(i,t) × ds(i)
        flux_series = np.nansum(flux_local * ds_m[:, np.newaxis], axis=0)
        
        # --- 6. STATISTICS ---
        salinity_mean = np.nanmean(sss_sos, axis=0)
        density_mean = np.nanmean(sss_dos, axis=0)
        velocity_mean = np.nanmean(u_normal, axis=0)
        
        salinity_profile = np.nanmean(sss_sos, axis=1)
        density_profile = np.nanmean(sss_dos, axis=1)
        
        flux_mean = float(np.nanmean(flux_series))
        flux_std = float(np.nanstd(flux_series))
        
        # Convert to Sv equivalent (1 Sv = 10⁶ m³/s)
        # This is just for reference - not physically meaningful for salt
        # But gives sense of scale: if mean density ~1025 and mean S ~35 PSU
        # then flux_kg/s ÷ (1025 × 35/1000) ≈ volume_flux_m³/s
        mean_rho = np.nanmean(sss_dos)
        mean_s = np.nanmean(sss_sos)
        if mean_rho > 0 and mean_s > 0:
            equiv_volume = flux_mean / (mean_rho * mean_s / 1000)
            flux_sv = equiv_volume / 1e6
        else:
            flux_sv = 0.0
        
        logger.info(f"Salt flux: mean={flux_mean:.2e} kg/s, std={flux_std:.2e} kg/s, ~{flux_sv:.3f} Sv equiv")
        
        return SaltFluxData(
            strait_name=cmems_data.strait_name,
            flux_series=flux_series,
            flux_local=flux_local,
            time_array=cmems_data.time_array,
            salinity_mean=salinity_mean,
            density_mean=density_mean,
            velocity_mean=velocity_mean,
            salinity_profile=salinity_profile,
            density_profile=density_profile,
            depth_profile=H_eff,
            x_km=x_km,
            gate_lon_pts=cmems_data.gate_lon_pts,
            gate_lat_pts=cmems_data.gate_lat_pts,
            depth_cap=depth_cap,
            flux_mean=flux_mean,
            flux_std=flux_std,
            flux_sv=flux_sv,
        )
    
    def _compute_normal_velocity(
        self,
        ugos: np.ndarray,
        vgos: np.ndarray,
        gate_lon: np.ndarray,
        gate_lat: np.ndarray,
        gate_angle: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute velocity component normal to gate.
        
        u_normal = u × sin(θ) + v × cos(θ)
        
        where θ is the angle of the gate tangent from East.
        """
        if ugos is None or vgos is None:
            logger.error("Missing ugos or vgos")
            return None
        
        n_pts = len(gate_lon)
        
        if gate_angle is None:
            # Compute gate angle from geometry
            gate_angle = np.zeros(n_pts)
            for i in range(n_pts):
                if i == 0:
                    dx = gate_lon[1] - gate_lon[0]
                    dy = gate_lat[1] - gate_lat[0]
                elif i == n_pts - 1:
                    dx = gate_lon[-1] - gate_lon[-2]
                    dy = gate_lat[-1] - gate_lat[-2]
                else:
                    dx = gate_lon[i+1] - gate_lon[i-1]
                    dy = gate_lat[i+1] - gate_lat[i-1]
                
                # Angle of tangent from East
                gate_angle[i] = np.arctan2(dy, dx)
        
        # Normal is perpendicular to tangent (rotate 90°)
        # n̂ = (sin(θ), cos(θ)) if tangent is (cos(θ), sin(θ))
        # Actually: tangent = (cos θ, sin θ), normal = (-sin θ, cos θ) or (sin θ, -cos θ)
        # We want flow crossing the gate, so:
        # u_normal = u × (-sin θ) + v × cos θ  [pointing "left" of gate direction]
        # Or reverse sign depending on convention
        
        # Standard convention: positive = flow from left to right looking along gate
        sin_theta = np.sin(gate_angle)[:, np.newaxis]
        cos_theta = np.cos(gate_angle)[:, np.newaxis]
        
        # u_normal = -u*sin(θ) + v*cos(θ)
        u_normal = -ugos * sin_theta + vgos * cos_theta
        
        return u_normal
    
    def _check_time_alignment(self, t1: np.ndarray, t2: np.ndarray, tol_days: float = 1.0) -> bool:
        """Check if two time arrays are aligned within tolerance."""
        if len(t1) != len(t2):
            return False
        
        t1_dt = pd.to_datetime(t1)
        t2_dt = pd.to_datetime(t2)
        
        diff_days = np.abs((t1_dt - t2_dt).total_seconds() / 86400)
        return np.all(diff_days < tol_days)
    
    def _interpolate_to_times(
        self,
        sos: np.ndarray,
        dos: np.ndarray,
        sss_times: np.ndarray,
        target_times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate SSS data to target times."""
        import xarray as xr
        
        n_pts = sos.shape[0]
        n_target = len(target_times)
        
        # Create DataArrays for interpolation
        sos_da = xr.DataArray(sos, dims=["space", "time"], coords={"time": sss_times})
        dos_da = xr.DataArray(dos, dims=["space", "time"], coords={"time": sss_times})
        
        # Interpolate
        sos_interp = sos_da.interp(time=target_times, method="linear").values
        dos_interp = dos_da.interp(time=target_times, method="linear").values
        
        logger.info(f"Interpolated SSS from {len(sss_times)} to {n_target} times")
        
        return sos_interp, dos_interp
