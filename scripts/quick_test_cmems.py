#!/usr/bin/env python3
"""Quick test of CMEMS service."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.cmems_service import CMEMSService, CMEMSConfig, CACHE_DIR, MAX_WORKERS

print("OK - imports work")
print(f"CACHE_DIR: {CACHE_DIR}")
print(f"MAX_WORKERS: {MAX_WORKERS}")

config = CMEMSConfig()
print(f"use_parallel: {config.use_parallel}")
print(f"use_cache: {config.use_cache}")
print(f"source_mode: {config.source_mode}")
print(f"api_dataset: {config.api_dataset}")

service = CMEMSService(config)
bounds = {"lon_min": -10, "lon_max": 10, "lat_min": 40, "lat_max": 60}
cache_key = service._get_cache_key(bounds)
print(f"cache_key: {cache_key}")
print("ALL TESTS PASSED")
