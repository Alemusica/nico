"""
Physical constants and model parameters for storm surge prediction.

No magic numbers in code - all constants defined here with units and sources.
"""

# =============================================================================
# Physical Constants
# =============================================================================

# Gravitational acceleration [m/s²]
G = 9.81

# Water density [kg/m³]
RHO_WATER = 1025.0  # Seawater

# Air density [kg/m³]
RHO_AIR = 1.225

# Earth rotation rate [rad/s]
OMEGA_EARTH = 7.2921e-5

# Coriolis parameter at 56°N (Denmark) [1/s]
# f = 2 * omega * sin(lat)
F_CORIOLIS_DK = 1.2e-4

# =============================================================================
# Wind Drag Coefficients
# =============================================================================

# Smith & Banke (1975) drag coefficient for wind stress
# τ = ρ_air * C_D * |U|² 
CD_LOW = 1.0e-3   # |U| < 5 m/s
CD_HIGH = 2.5e-3  # |U| > 25 m/s

# =============================================================================
# Inverse Barometer Effect
# =============================================================================

# Sea level rise per hPa pressure drop [m/hPa]
# Δη = -ΔP / (ρg) ≈ 0.01 m per 1 hPa
INVERSE_BAROMETER = 0.01

# =============================================================================
# Model Thresholds (Pipeline Gates)
# =============================================================================

# Stage 1: Fingerprint match threshold
FINGERPRINT_GATE = 0.60

# Stage 2: GNN ensemble confidence
ENSEMBLE_GATE = 0.70

# Stage 3: Physics residual threshold (lower = stricter)
PHYSICS_RESIDUAL_THRESHOLD = 0.05

# Final: Combined confidence for alert
ALERT_THRESHOLD = 0.80

# Gray zone: historical strong but physics weak
GRAY_ZONE_HISTORICAL_MIN = 0.75

# =============================================================================
# Loss Function Weights
# =============================================================================

# Initial physics weight (high = science dominates, prevents spurious correlations)
LAMBDA_PHYSICS_INIT = 0.90

# Decay factor per epoch (gradually trust data more)
LAMBDA_DECAY = 0.95

# Minimum physics weight
LAMBDA_PHYSICS_MIN = 0.30

# =============================================================================
# Data Resolution
# =============================================================================

# ERA5 grid spacing [degrees]
ERA5_RESOLUTION = 0.25

# Time resolution for fingerprinting [hours]
FINGERPRINT_TIME_RESOLUTION = 1.0

# Forecast horizon [hours]
FORECAST_HORIZON = 72

# =============================================================================
# Denmark Bounding Box
# =============================================================================

# Lat/Lon bounds for high-resolution focus
DENMARK_BOUNDS = {
    "lat_min": 54.5,
    "lat_max": 58.0,
    "lon_min": 7.5,
    "lon_max": 15.5,
}

# North Sea extended bounds (for teleconnections)
NORTH_SEA_BOUNDS = {
    "lat_min": 50.0,
    "lat_max": 62.0,
    "lon_min": -5.0,
    "lon_max": 15.0,
}

# Atlantic extended (for remote forcing detection)
ATLANTIC_BOUNDS = {
    "lat_min": 35.0,
    "lat_max": 65.0,
    "lon_min": -30.0,
    "lon_max": 15.0,
}
