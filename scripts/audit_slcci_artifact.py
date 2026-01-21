#!/usr/bin/env python
"""
🔬 AUDIT SCRIPT: SLCCI Periodic Artifact Diagnosis
===================================================

This script isolates and diagnoses the source of the periodic artifact
observed in SLCCI analysis outputs.

Steps:
1. Load raw SLCCI data directly (bypassing service layer)
2. Check for periodicity in RAW corssh values
3. Check for periodicity in DOT after geoid subtraction
4. Check for periodicity in binned/averaged data
5. Compare with SLCCI PLOTTER notebook methodology

Run: source .venv/bin/activate && python scripts/audit_slcci_artifact.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import RegularGridInterpolator

# Configuration
BASE_DIR = "/Users/nicolocaron/Desktop/ARCFRESH/J2"
GEOID_PATH = "/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc"
GATE_PATH = "/Users/nicolocaron/Documents/GitHub/nico/gates/fram_strait_S3_pass_481.shp"
PASS_NUMBER = 481
CYCLES = list(range(1, 100))  # Subset for speed
LON_BIN_SIZE = 0.01

print("="*70)
print("🔬 SLCCI PERIODIC ARTIFACT DIAGNOSTIC")
print("="*70)


# ==============================================================================
# STEP 1: Load raw data directly (minimal processing)
# ==============================================================================
print("\n📊 STEP 1: Loading raw SLCCI data...")

import geopandas as gpd
import os
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Load gate
gate = gpd.read_file(GATE_PATH)
if gate.crs is None:
    gate = gate.set_crs("EPSG:3413")
gate = gate.to_crs("EPSG:4326")
lon_min_g, lat_min_g, lon_max_g, lat_max_g = gate.total_bounds

print(f"Gate bounds: lat [{lat_min_g:.2f}, {lat_max_g:.2f}], lon [{lon_min_g:.2f}, {lon_max_g:.2f}]")

# Load cycles directly
raw_data = []
for cycle in CYCLES:
    cycle_str = str(cycle).zfill(3)
    filepath = f"{BASE_DIR}/SLCCI_ALTDB_J2_Cycle{cycle_str}_V2.nc"
    
    if not os.path.exists(filepath):
        continue
    
    try:
        with xr.open_dataset(filepath, decode_times=False) as ds:
            lon = ds["longitude"].values
            lat = ds["latitude"].values
            
            # Wrap longitude
            lon_wrapped = ((lon + 180) % 360) - 180
            
            # Spatial filter
            lat_buffer = 2.0
            lon_buffer = 5.0
            mask = (
                (lat >= lat_min_g - lat_buffer) & (lat <= lat_max_g + lat_buffer) &
                (lon_wrapped >= lon_min_g - lon_buffer) & (lon_wrapped <= lon_max_g + lon_buffer)
            )
            
            if mask.sum() == 0:
                continue
            
            # Pass filter
            if "pass" in ds.variables:
                pass_vals = np.round(ds["pass"].values).astype(int)
                mask = mask & (pass_vals == PASS_NUMBER)
            
            if mask.sum() == 0:
                continue
            
            # Quality filter
            if "validation_flag" in ds.variables:
                mask = mask & (ds["validation_flag"].values == 0)
            
            if mask.sum() == 0:
                continue
            
            # Extract data
            time_days = ds["time"].values[mask]
            corssh = ds["corssh"].values[mask]
            
            for i in range(len(time_days)):
                raw_data.append({
                    "cycle": cycle,
                    "time_days": time_days[i],
                    "lat": lat[mask][i],
                    "lon": lon_wrapped[mask][i],
                    "corssh": corssh[i],
                })
    except Exception as e:
        continue

df_raw = pd.DataFrame(raw_data)
df_raw["time"] = pd.to_datetime(df_raw["time_days"], origin="1950-01-01", unit="D")
df_raw = df_raw.sort_values("time")

print(f"✅ Loaded {len(df_raw)} raw observations from {df_raw['cycle'].nunique()} cycles")
print(f"   Time range: {df_raw['time'].min()} to {df_raw['time'].max()}")


# ==============================================================================
# STEP 2: Check periodicity in RAW corssh
# ==============================================================================
print("\n📊 STEP 2: Checking periodicity in RAW corssh values...")

# Group by time (daily average)
df_raw["date"] = df_raw["time"].dt.date
daily_corssh = df_raw.groupby("date")["corssh"].mean()

if len(daily_corssh) > 10:
    # Detrend
    detrended = signal.detrend(daily_corssh.dropna().values)
    
    # FFT
    fft = np.fft.fft(detrended)
    freqs = np.fft.fftfreq(len(detrended), d=1)  # d=1 day
    
    # Find dominant frequencies
    power = np.abs(fft)**2
    positive_mask = freqs > 0
    top_idx = np.argsort(power[positive_mask])[-5:]
    top_freqs = freqs[positive_mask][top_idx]
    top_periods = 1 / top_freqs  # in days
    
    print(f"   Dominant periods in RAW corssh (days): {top_periods}")
    
    # Check for ~10-day orbital period
    orbital_period = 9.9156  # Jason-2 repeat cycle
    for period in top_periods:
        if abs(period - orbital_period) < 1:
            print(f"   ⚠️ FOUND: Period {period:.2f} days matches J2 orbital cycle!")
else:
    print("   ⚠️ Not enough data for FFT analysis")


# ==============================================================================
# STEP 3: Add geoid and compute DOT
# ==============================================================================
print("\n📊 STEP 3: Computing DOT (corssh - geoid)...")

# Load geoid
ds_geoid = xr.open_dataset(GEOID_PATH)
lat_geoid = ds_geoid["lat"].values
lon_geoid = ds_geoid["lon"].values
geoid_values = ds_geoid["value"].values

lon_wrapped_geoid = ((lon_geoid + 180) % 360) - 180
sort_idx = np.argsort(lon_wrapped_geoid)
lon_sorted = lon_wrapped_geoid[sort_idx]
unique_idx = np.concatenate(([True], np.diff(lon_sorted) != 0))
lon_sorted = lon_sorted[unique_idx]
geoid_sorted = geoid_values[:, sort_idx][:, unique_idx]

geoid_interp = RegularGridInterpolator(
    (lat_geoid, lon_sorted),
    geoid_sorted,
    method="nearest",
    bounds_error=False,
    fill_value=np.nan,
)

# Interpolate geoid at observation points
points = np.column_stack([df_raw["lat"].values, df_raw["lon"].values])
df_raw["geoid"] = geoid_interp(points)
df_raw["dot"] = df_raw["corssh"] - df_raw["geoid"]

print(f"   DOT range: {df_raw['dot'].min():.4f} to {df_raw['dot'].max():.4f} m")

# Check DOT periodicity
daily_dot = df_raw.groupby("date")["dot"].mean()

if len(daily_dot) > 10:
    detrended_dot = signal.detrend(daily_dot.dropna().values)
    fft_dot = np.fft.fft(detrended_dot)
    power_dot = np.abs(fft_dot)**2
    
    positive_mask = freqs[:len(power_dot)] > 0
    if positive_mask.sum() > 5:
        top_idx_dot = np.argsort(power_dot[positive_mask])[-5:]
        top_freqs_dot = freqs[:len(power_dot)][positive_mask][top_idx_dot]
        top_periods_dot = 1 / top_freqs_dot
        print(f"   Dominant periods in DOT (days): {top_periods_dot}")


# ==============================================================================
# STEP 4: Apply monthly binning (SLCCI PLOTTER method)
# ==============================================================================
print("\n📊 STEP 4: Applying monthly longitude binning (SLCCI PLOTTER method)...")

df_raw["month"] = df_raw["time"].dt.month
df_raw["year"] = df_raw["time"].dt.year
df_raw["year_month"] = df_raw["time"].dt.to_period("M")

# Method A: Group by MONTH only (SLCCI PLOTTER style for 12-subplot)
print("\n   Method A: Group by MONTH (1-12), all years combined")
monthly_slopes_A = {}

for month in range(1, 13):
    month_data = df_raw[df_raw["month"] == month]
    if len(month_data) < 10:
        continue
    
    lon_min = month_data["lon"].min()
    lon_max = month_data["lon"].max()
    lon_bins = np.arange(lon_min, lon_max + LON_BIN_SIZE, LON_BIN_SIZE)
    
    if len(lon_bins) < 3:
        continue
    
    month_data_copy = month_data.copy()
    month_data_copy["lon_bin"] = pd.cut(month_data_copy["lon"], bins=lon_bins, labels=False)
    binned = month_data_copy.groupby("lon_bin")["dot"].mean().dropna()
    
    if len(binned) < 3:
        continue
    
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    x_vals = lon_centers[binned.index.astype(int)]
    y_vals = binned.values
    
    slope, _ = np.polyfit(x_vals, y_vals, 1)
    monthly_slopes_A[month] = slope

print(f"   Monthly slopes (m/deg): {monthly_slopes_A}")


# Method B: Group by YEAR_MONTH (time series)
print("\n   Method B: Group by YEAR_MONTH (chronological time series)")
time_periods = sorted(df_raw["year_month"].unique())
monthly_slopes_B = {}

for period in time_periods:
    period_data = df_raw[df_raw["year_month"] == period]
    if len(period_data) < 10:
        continue
    
    lon_min = period_data["lon"].min()
    lon_max = period_data["lon"].max()
    lon_bins = np.arange(lon_min, lon_max + LON_BIN_SIZE, LON_BIN_SIZE)
    
    if len(lon_bins) < 3:
        continue
    
    period_data_copy = period_data.copy()
    period_data_copy["lon_bin"] = pd.cut(period_data_copy["lon"], bins=lon_bins, labels=False)
    binned = period_data_copy.groupby("lon_bin")["dot"].mean().dropna()
    
    if len(binned) < 3:
        continue
    
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    x_vals = lon_centers[binned.index.astype(int)]
    y_vals = binned.values
    
    slope, _ = np.polyfit(x_vals, y_vals, 1)
    monthly_slopes_B[period] = slope

print(f"   Time series has {len(monthly_slopes_B)} monthly slope values")


# ==============================================================================
# STEP 5: Visualize and analyze periodicity
# ==============================================================================
print("\n📊 STEP 5: Generating diagnostic plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Raw corssh time series
ax1 = axes[0, 0]
ax1.scatter(df_raw["time"], df_raw["corssh"], s=1, alpha=0.3, label="Raw corssh")
ax1.set_xlabel("Time")
ax1.set_ylabel("corssh (m)")
ax1.set_title("Raw CORSSH Time Series")
ax1.legend()

# Plot 2: DOT time series
ax2 = axes[0, 1]
ax2.scatter(df_raw["time"], df_raw["dot"], s=1, alpha=0.3, c="orange", label="DOT")
ax2.set_xlabel("Time")
ax2.set_ylabel("DOT (m)")
ax2.set_title("DOT Time Series (corssh - geoid)")
ax2.legend()

# Plot 3: Monthly slopes (Method B - time series)
ax3 = axes[1, 0]
if monthly_slopes_B:
    dates = [pd.Timestamp(str(p)) for p in monthly_slopes_B.keys()]
    slopes = list(monthly_slopes_B.values())
    ax3.plot(dates, slopes, "-o", markersize=3)
    ax3.axhline(0, color="k", linewidth=0.5)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Slope (m/deg lon)")
    ax3.set_title("Monthly Slope Time Series (YEAR_MONTH)")
    
    # Add FFT analysis annotation
    if len(slopes) > 12:
        detrended_slopes = signal.detrend(slopes)
        fft_slopes = np.fft.fft(detrended_slopes)
        freqs_slopes = np.fft.fftfreq(len(detrended_slopes), d=1)  # d=1 month
        power_slopes = np.abs(fft_slopes)**2
        
        positive_mask = freqs_slopes > 0
        if positive_mask.sum() > 0:
            max_idx = np.argmax(power_slopes[positive_mask])
            dominant_freq = freqs_slopes[positive_mask][max_idx]
            dominant_period = 1 / dominant_freq if dominant_freq > 0 else np.inf
            ax3.annotate(f"Dominant period: {dominant_period:.1f} months", 
                        xy=(0.05, 0.95), xycoords="axes fraction",
                        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat"))

# Plot 4: Monthly average slope (Method A - 12 months)
ax4 = axes[1, 1]
if monthly_slopes_A:
    months = list(monthly_slopes_A.keys())
    slopes_A = list(monthly_slopes_A.values())
    ax4.bar(months, slopes_A, color="steelblue")
    ax4.axhline(0, color="k", linewidth=0.5)
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Mean Slope (m/deg lon)")
    ax4.set_title("Monthly Mean Slope (all years combined)")
    ax4.set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig("/Users/nicolocaron/Documents/GitHub/nico/scripts/slcci_artifact_diagnostic.png", dpi=150)
plt.show()

print("\n" + "="*70)
print("📋 DIAGNOSTIC SUMMARY")
print("="*70)
print(f"""
1. Raw data loaded: {len(df_raw)} observations
2. Time range: {df_raw['time'].min().date()} to {df_raw['time'].max().date()}
3. Monthly slopes computed with both methods

INTERPRETATION:
- If Plot 3 shows a clear ~12-month periodicity, it's a REAL seasonal signal
- If Plot 3 shows irregular periodicity, check for:
  a) Aliasing from orbital cycle (~10 days)
  b) Data gaps causing interpolation artifacts
  c) Indexing/ordering errors in the code

NEXT STEPS:
- Compare Plot 4 (12-month bar) with SLCCI PLOTTER notebook output
- If they match, the periodicity is PHYSICAL (seasonal DOT variation)
- If they don't match, there's a processing bug

Diagnostic plot saved to: scripts/slcci_artifact_diagnostic.png
""")
print("="*70)
