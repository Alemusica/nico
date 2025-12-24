"""
Shallow Water Equations for storm surge modeling.

Physics grounding for the hybrid model. These equations are used as
constraints in the PINN loss function to prevent spurious correlations.

Equations:
    Continuity:  ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
    Momentum x:  ∂u/∂t + u∂u/∂x + v∂u/∂y = -g∂η/∂x + τ_x/(ρh) - C_f*u*|u|/h
    Momentum y:  ∂v/∂t + u∂v/∂x + v∂v/∂y = -g∂η/∂y + τ_y/(ρh) - C_f*v*|v|/h

Where:
    h = water depth (bathymetry + surge)
    u, v = depth-averaged velocities
    η = sea surface elevation (surge height)
    τ = wind stress
    C_f = bottom friction coefficient
"""

import torch
import torch.nn as nn
from typing import Callable

from ..core.constants import G, RHO_WATER, RHO_AIR, CD_LOW, CD_HIGH


def compute_wind_stress(
    wind_u: torch.Tensor,
    wind_v: torch.Tensor,
    rho_air: float = RHO_AIR
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute wind stress from 10m wind components.
    
    Uses Smith & Banke (1975) drag coefficient.
    τ = ρ_air * C_D * |U| * U
    
    Args:
        wind_u: Eastward wind [m/s]
        wind_v: Northward wind [m/s]
        rho_air: Air density [kg/m³]
        
    Returns:
        tau_x, tau_y: Wind stress components [N/m²]
    """
    wind_speed = torch.sqrt(wind_u**2 + wind_v**2)
    
    # Drag coefficient varies with wind speed
    cd = torch.where(
        wind_speed < 5.0,
        torch.full_like(wind_speed, CD_LOW),
        torch.where(
            wind_speed > 25.0,
            torch.full_like(wind_speed, CD_HIGH),
            CD_LOW + (CD_HIGH - CD_LOW) * (wind_speed - 5.0) / 20.0
        )
    )
    
    tau_x = rho_air * cd * wind_speed * wind_u
    tau_y = rho_air * cd * wind_speed * wind_v
    
    return tau_x, tau_y


def inverse_barometer_effect(
    pressure: torch.Tensor,
    reference_pressure: float = 101325.0
) -> torch.Tensor:
    """
    Compute sea level change from atmospheric pressure anomaly.
    
    Δη = -(P - P_ref) / (ρ * g)
    
    1 hPa drop ≈ 1 cm rise (inverse barometer)
    
    Args:
        pressure: Atmospheric pressure [Pa]
        reference_pressure: Reference pressure [Pa]
        
    Returns:
        eta_ib: Sea level anomaly [m]
    """
    return -(pressure - reference_pressure) / (RHO_WATER * G)


class ShallowWaterResidual(nn.Module):
    """
    Computes SWE residuals for physics-informed loss.
    
    The residuals should be zero if the solution satisfies the equations.
    High residual = physics violation = penalize heavily.
    """
    
    def __init__(
        self,
        g: float = G,
        rho_water: float = RHO_WATER,
        friction_coef: float = 0.003
    ):
        super().__init__()
        self.g = g
        self.rho_water = rho_water
        self.friction_coef = friction_coef
    
    def forward(
        self,
        eta: torch.Tensor,      # Surface elevation [N, H, W] or [N, T, H, W]
        u: torch.Tensor,        # Velocity x [N, H, W]
        v: torch.Tensor,        # Velocity y [N, H, W]
        h: torch.Tensor,        # Total depth [N, H, W]
        tau_x: torch.Tensor,    # Wind stress x [N, H, W]
        tau_y: torch.Tensor,    # Wind stress y [N, H, W]
        dx: float,              # Grid spacing x [m]
        dy: float,              # Grid spacing y [m]
        dt: float,              # Time step [s]
    ) -> dict[str, torch.Tensor]:
        """
        Compute residuals of SWE.
        
        Returns dict with:
            - continuity_residual
            - momentum_x_residual
            - momentum_y_residual
            - total_residual (scalar)
        """
        # Spatial derivatives (central differences)
        # eta_x = ∂η/∂x
        eta_x = (torch.roll(eta, -1, dims=-1) - torch.roll(eta, 1, dims=-1)) / (2 * dx)
        eta_y = (torch.roll(eta, -1, dims=-2) - torch.roll(eta, 1, dims=-2)) / (2 * dy)
        
        # Flux derivatives for continuity
        hu = h * u
        hv = h * v
        hu_x = (torch.roll(hu, -1, dims=-1) - torch.roll(hu, 1, dims=-1)) / (2 * dx)
        hv_y = (torch.roll(hv, -1, dims=-2) - torch.roll(hv, 1, dims=-2)) / (2 * dy)
        
        # Velocity derivatives for momentum
        u_x = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2 * dx)
        u_y = (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)) / (2 * dy)
        v_x = (torch.roll(v, -1, dims=-1) - torch.roll(v, 1, dims=-1)) / (2 * dx)
        v_y = (torch.roll(v, -1, dims=-2) - torch.roll(v, 1, dims=-2)) / (2 * dy)
        
        # Speed for friction
        speed = torch.sqrt(u**2 + v**2 + 1e-8)
        
        # Friction terms
        friction_x = self.friction_coef * u * speed / (h + 1e-8)
        friction_y = self.friction_coef * v * speed / (h + 1e-8)
        
        # Wind forcing terms
        wind_x = tau_x / (self.rho_water * (h + 1e-8))
        wind_y = tau_y / (self.rho_water * (h + 1e-8))
        
        # Residuals (should be ≈ 0)
        # Note: time derivatives computed externally or as part of sequential prediction
        
        # Continuity: ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
        # For steady-state or diagnostic: just spatial terms
        continuity_res = hu_x + hv_y
        
        # Momentum x: u∂u/∂x + v∂u/∂y + g∂η/∂x - wind + friction = 0
        momentum_x_res = u * u_x + v * u_y + self.g * eta_x - wind_x + friction_x
        
        # Momentum y: u∂v/∂x + v∂v/∂y + g∂η/∂y - wind + friction = 0
        momentum_y_res = u * v_x + v * v_y + self.g * eta_y - wind_y + friction_y
        
        # Total residual (for loss)
        total = (
            torch.mean(continuity_res**2) +
            torch.mean(momentum_x_res**2) +
            torch.mean(momentum_y_res**2)
        )
        
        return {
            "continuity": continuity_res,
            "momentum_x": momentum_x_res,
            "momentum_y": momentum_y_res,
            "total": total,
        }


def physics_loss(
    residuals: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None
) -> torch.Tensor:
    """
    Compute weighted physics loss from SWE residuals.
    
    Args:
        residuals: Output from ShallowWaterResidual.forward()
        weights: Optional weights for each term
        
    Returns:
        Scalar loss tensor
    """
    if weights is None:
        weights = {"continuity": 1.0, "momentum_x": 1.0, "momentum_y": 1.0}
    
    loss = torch.tensor(0.0, device=residuals["continuity"].device)
    
    for key, weight in weights.items():
        if key in residuals and key != "total":
            loss = loss + weight * torch.mean(residuals[key]**2)
    
    return loss
