#!/usr/bin/env python3
"""
Populate CMEMS L4 Cache for Key Arctic Gates

Downloads and caches CMEMS L4 data (2000-2020) for:
- Fram Strait
- Denmark Strait  
- Bering Strait
- Davis Strait (full, east, west)

Usage:
    python scripts/populate_cmems_l4_cache.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.cmems_l4_service import CMEMSL4Service, CMEMSL4Config
import time

# Gates to cache
GATES_TO_CACHE = [
    # Fram & Nordic Seas
    ("fram_strait_S3_pass_481", "gates/fram_strait_S3_pass_481.shp"),
    ("denmark_strait_TPJ_pass_246", "gates/denmark_strait_TPJ_pass_246.shp"),
    
    # Bering Strait
    ("bering_strait_TPJ_pass_076", "gates/bering_strait_TPJ_pass_076.shp"),
    
    # Davis Strait (all 3)
    ("davis_strait", "gates/davis_strait.shp"),
    ("davis_strait_east", "gates/davis_strait_east.shp"),
    ("davis_strait_west", "gates/davis_strait_west.shp"),
]

# Time range
TIME_START = "2000-01-01"
TIME_END = "2020-12-31"

# Variables to download
VARIABLES = ["adt", "sla", "ugos", "vgos"]


def main():
    print("=" * 70)
    print("CMEMS L4 CACHE POPULATION")
    print(f"Time range: {TIME_START} to {TIME_END}")
    print(f"Variables: {VARIABLES}")
    print(f"Gates: {len(GATES_TO_CACHE)}")
    print("=" * 70)
    
    service = CMEMSL4Service()
    
    results = []
    
    for i, (gate_name, gate_path) in enumerate(GATES_TO_CACHE, 1):
        print(f"\n[{i}/{len(GATES_TO_CACHE)}] Processing: {gate_name}")
        print("-" * 50)
        
        if not os.path.exists(gate_path):
            print(f"  ⚠️ Shapefile not found: {gate_path}")
            results.append((gate_name, "SKIPPED", "File not found"))
            continue
        
        config = CMEMSL4Config(
            gate_path=gate_path,
            time_start=TIME_START,
            time_end=TIME_END,
            variables=VARIABLES
        )
        
        start_time = time.time()
        
        try:
            # Force reload to ensure fresh data
            data = service.load_gate_data(
                config=config,
                force_reload=False,  # Use cache if available
                use_cache=True
            )
            
            elapsed = time.time() - start_time
            
            if data is not None:
                n_pts = data.dot_matrix.shape[0]
                n_time = data.dot_matrix.shape[1]
                print(f"  ✅ Success: {n_pts} pts × {n_time} times ({elapsed:.1f}s)")
                results.append((gate_name, "SUCCESS", f"{n_pts}×{n_time} in {elapsed:.1f}s"))
            else:
                print(f"  ❌ Failed: No data returned")
                results.append((gate_name, "FAILED", "No data"))
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ❌ Error: {e}")
            results.append((gate_name, "ERROR", str(e)[:50]))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success = sum(1 for _, status, _ in results if status == "SUCCESS")
    failed = len(results) - success
    
    print(f"\n✅ Success: {success}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    
    print("\nDetails:")
    for gate_name, status, info in results:
        emoji = "✅" if status == "SUCCESS" else "❌"
        print(f"  {emoji} {gate_name}: {info}")
    
    # Cache stats
    stats = service.get_cache_stats()
    print(f"\nCache Stats:")
    print(f"  L1 (raw): {stats.get('l1_entries', 0)} entries")
    print(f"  L2 (processed): {stats.get('l2_entries', 0)} entries")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
