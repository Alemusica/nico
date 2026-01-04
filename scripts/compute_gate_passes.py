#!/usr/bin/env python
"""
Compute Gate Passes
===================
Script to compute and save closest passes for all gates.

Run this ONCE to generate config/gate_passes.yaml, then passes will be
loaded instantly without re-computation.

Usage:
    source .venv/bin/activate
    python scripts/compute_gate_passes.py

Output:
    config/gate_passes.yaml
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.services.slcci_service import SLCCIService, SLCCIConfig
from src.services.cmems_service import CMEMSService, CMEMSConfig


# Configuration - UPDATE THESE PATHS
SLCCI_BASE_DIR = "/Users/nicolocaron/Desktop/ARCFRESH/J2"
CMEMS_BASE_DIR = "/Users/nicolocaron/Desktop/ARCFRESH/COPERNICUS DATA"
GATES_DIR = Path(__file__).parent.parent / "gates"
OUTPUT_FILE = Path(__file__).parent.parent / "config" / "gate_passes.yaml"


def get_display_name(gate_name: str) -> str:
    """Convert gate filename to display name."""
    import re
    
    name = gate_name
    name = re.sub(r"_TPJ_pass_\d+", "", name)
    name = re.sub(r"_S3_pass_\d+", "", name)
    name = name.replace("_", " ").replace("-", " - ").title()
    
    return name


def extract_suggested_pass(gate_name: str) -> int | None:
    """Extract pass number from gate filename."""
    import re
    
    match = re.search(r"_(?:TPJ|S3)_pass_(\d+)", gate_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def main():
    print("=" * 60)
    print("Computing closest passes for all gates...")
    print("=" * 60)
    
    # Initialize services
    slcci_service = None
    cmems_service = None
    
    if Path(SLCCI_BASE_DIR).exists():
        print(f"\n📁 SLCCI: {SLCCI_BASE_DIR}")
        slcci_config = SLCCIConfig(base_dir=SLCCI_BASE_DIR)
        slcci_service = SLCCIService(slcci_config)
    else:
        print(f"\n⚠️  SLCCI directory not found: {SLCCI_BASE_DIR}")
    
    if Path(CMEMS_BASE_DIR).exists():
        print(f"📁 CMEMS: {CMEMS_BASE_DIR}")
        cmems_config = CMEMSConfig(base_dir=CMEMS_BASE_DIR)
        cmems_service = CMEMSService(cmems_config)
    else:
        print(f"⚠️  CMEMS directory not found: {CMEMS_BASE_DIR}")
    
    # Find all gate shapefiles
    gate_files = sorted(GATES_DIR.glob("*.shp"))
    print(f"\n🗺️  Found {len(gate_files)} gates in {GATES_DIR}\n")
    
    # Build output
    output = {
        "gates": {}
    }
    
    for gate_path in gate_files:
        gate_name = gate_path.stem
        print(f"\n📍 {gate_name}")
        
        gate_config = {
            "display_name": get_display_name(gate_name)
        }
        
        # Extract suggested pass if in filename
        suggested = extract_suggested_pass(gate_name)
        if suggested:
            gate_config["suggested_pass"] = suggested
            print(f"   Suggested pass: {suggested}")
        
        # Compute SLCCI closest passes
        if slcci_service:
            try:
                closest = slcci_service.find_closest_pass(str(gate_path), n_passes=5)
                passes = [p[0] for p in closest if p[1] != float('inf')]
                if passes:
                    gate_config["slcci_passes"] = passes
                    print(f"   SLCCI passes: {passes}")
                else:
                    gate_config["slcci_passes"] = [248, 163, 76, 220, 246]  # Default
                    print("   SLCCI passes: using defaults (no data found)")
            except Exception as e:
                print(f"   SLCCI error: {e}")
                gate_config["slcci_passes"] = [248, 163, 76, 220, 246]
        else:
            gate_config["slcci_passes"] = [248, 163, 76, 220, 246]
        
        # Compute CMEMS closest tracks
        if cmems_service:
            try:
                closest = cmems_service.find_closest_tracks(str(gate_path), n_tracks=5)
                tracks = [t[0] for t in closest] if closest else []
                if tracks:
                    gate_config["cmems_tracks"] = tracks
                    print(f"   CMEMS tracks: {tracks}")
                else:
                    gate_config["cmems_tracks"] = [481, 163, 248, 220, 76]  # Default
                    print("   CMEMS tracks: using defaults (no data found)")
            except Exception as e:
                print(f"   CMEMS error: {e}")
                gate_config["cmems_tracks"] = [481, 163, 248, 220, 76]
        else:
            gate_config["cmems_tracks"] = [481, 163, 248, 220, 76]
        
        output["gates"][gate_name] = gate_config
    
    # Write output
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    
    header = """# Pre-computed closest passes for each gate
# Format: gate_name -> { slcci_passes: [...], cmems_tracks: [...] }
# 
# These are computed ONCE and cached here for instant loading.
# Run `python scripts/compute_gate_passes.py` to regenerate.
#
# Last updated: """ + str(Path(sys.argv[0]).stat().st_mtime if len(sys.argv) > 0 else "unknown") + "\n\n"
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(header)
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Done! Saved {len(output['gates'])} gates.")
    print("\nYou can now use '5 Closest' mode and passes will load instantly!")


if __name__ == "__main__":
    main()
