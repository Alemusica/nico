#!/usr/bin/env python3
"""
🌊 Test Manuale: Lago Maggiore Flood Event - Ottobre 2000
=========================================================

Evento storico: 13-16 Ottobre 2000
- Precipitazioni estreme (600mm in 72h)
- Overflow Lago Maggiore
- 23 vittime

Obiettivo test:
1. Query datasets per bbox Nord Italia + time range
2. Caricare dati via catalog API
3. Visualizzare causal chains
4. Export per mappa
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import json

# ==============================================================================
# 1. CONFIGURAZIONE EVENTO
# ==============================================================================

EVENT_CONFIG = {
    "name": "Lago Maggiore Flood 2000",
    "bbox": {
        "west": 8.0,
        "south": 45.0,
        "east": 10.0,
        "north": 47.0
    },
    "time_range": {
        "start": "2000-10-10",
        "end": "2000-10-20"
    },
    "peak_date": "2000-10-15",
    "variables_of_interest": [
        "precipitation",
        "runoff",
        "soil_moisture",
        "msl",  # mean sea level pressure
        "water_level"
    ]
}

print("=" * 60)
print("🌊 LAGO MAGGIORE FLOOD - OTTOBRE 2000")
print("=" * 60)
print(f"📍 BBox: {EVENT_CONFIG['bbox']}")
print(f"📅 Time: {EVENT_CONFIG['time_range']['start']} → {EVENT_CONFIG['time_range']['end']}")
print(f"🎯 Variables: {EVENT_CONFIG['variables_of_interest']}")
print()

# ==============================================================================
# 2. QUERY CATALOG - QUALI DATASET SONO DISPONIBILI?
# ==============================================================================

print("📚 STEP 1: Query Catalog")
print("-" * 40)

try:
    from src.data_manager.intake_bridge import IntakeCatalogBridge
    
    bridge = IntakeCatalogBridge()
    all_datasets = bridge.list_datasets()
    
    # Filtra per variabili rilevanti
    matching_datasets = []
    for ds_id in all_datasets:
        try:
            meta = bridge.get_metadata(ds_id)
            variables = meta.get('variables', [])
            if any(v in variables for v in EVENT_CONFIG['variables_of_interest']):
                matching_datasets.append({
                    'id': ds_id,
                    'variables': variables,
                    'latency_class': meta.get('latency_class', 'unknown')
                })
        except:
            pass
    
    print(f"✅ Found {len(matching_datasets)} matching datasets (from {len(all_datasets)} total):")
    for ds in matching_datasets:
        latency = ds.get('latency_class', 'unknown')
        badge = {"live": "🟢", "daily": "🔵", "monthly": "🟡"}.get(latency, "⚪")
        print(f"   {badge} {ds['id']}: {', '.join(ds.get('variables', []))}")
    
except Exception as e:
    print(f"⚠️ IntakeBridge error: {e}")
    print("   Fallback: carico catalog.yaml direttamente...")
    
    import yaml
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "catalog.yaml"  # In root, not data/
    )
    
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            catalog_data = yaml.safe_load(f)
        
        sources = catalog_data.get('sources', {})
        print(f"✅ Loaded {len(sources)} sources from catalog.yaml:")
        
        for name, source in sources.items():
            metadata = source.get('metadata', {})
            variables = metadata.get('variables', [])
            latency = metadata.get('latency_class', 'unknown')
            badge = {"live": "🟢", "daily": "🔵", "monthly": "🟡"}.get(latency, "⚪")
            
            # Check variable match
            vars_match = any(v in variables for v in EVENT_CONFIG['variables_of_interest'])
            if vars_match:
                print(f"   {badge} {name}: {', '.join(variables[:5])}")
    else:
        print(f"❌ Catalog not found at {catalog_path}")

print()

# ==============================================================================
# 3. CHECK CAUSAL CHAINS - COSA CONOSCIAMO?
# ==============================================================================

print("🔗 STEP 2: Query Causal Chains")
print("-" * 40)

try:
    from src.data_manager.causal_graph import CausalGraphDB
    
    cgm = CausalGraphDB()
    
    # Query precursori per "flood"
    print("   Querying precursors for 'flood'...")
    precursors = cgm.get_precursors("flood")
    
    if precursors:
        print(f"   ✅ Found {len(precursors)} precursor chains:")
        for p in precursors:
            print(f"      • {p.get('source', 'unknown')} → flood (physics: {p.get('physics_score', 0):.2f})")
    else:
        print("   ⚠️ No precursor chains found in DB")
        print("   💡 Try: python -c \"from src.data_manager.causal_graph import CausalGraphDB; CausalGraphDB().seed_example_chains()\"")
    
    # Query anche per precipitation (common precursor)
    print()
    print("   Querying effects of ERA5 precipitation...")
    effects = cgm.get_effects("era5_reanalysis", "precipitation")
    
    if effects:
        print(f"   ✅ Found {len(effects)} effect chains:")
        for e in effects:
            print(f"      • precipitation → {e.get('target', '?')}.{e.get('target_var', '?')} (lag: {e.get('lag_days', '?')}d)")
    else:
        print("   ⚠️ No effect chains found")

except Exception as e:
    print(f"❌ Causal graph error: {e}")
    print("   Make sure SurrealDB is running: docker ps | grep surreal")

print()

# ==============================================================================
# 4. TEST API ENDPOINT (se running)
# ==============================================================================

print("🌐 STEP 3: Test API Endpoint")
print("-" * 40)

try:
    import requests
    
    API_BASE = "http://localhost:8000/api/v1"
    
    # Test catalog
    resp = requests.get(f"{API_BASE}/data/catalog", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        print(f"   ✅ API /catalog: {len(data)} datasets")
        
        # Test briefing per precipitazione
        briefing_resp = requests.get(
            f"{API_BASE}/data/catalog/briefing/era5_reanalysis",
            timeout=5
        )
        if briefing_resp.status_code == 200:
            brief = briefing_resp.json()
            print(f"   ✅ API /briefing: {brief.get('name', 'unknown')}")
            print(f"      Variables: {brief.get('variables', [])[:3]}...")
            print(f"      Time coverage: {brief.get('time_coverage', {})}")
        else:
            print(f"   ⚠️ API /briefing: {briefing_resp.status_code}")
    else:
        print(f"   ⚠️ API /catalog: {resp.status_code}")
        
except requests.exceptions.ConnectionError:
    print("   ⚠️ API not running. Start with: uvicorn api.main:app --reload")
except Exception as e:
    print(f"   ❌ API error: {e}")

print()

# ==============================================================================
# 5. SUMMARY & NEXT STEPS
# ==============================================================================

print("=" * 60)
print("📊 TEST SUMMARY")
print("=" * 60)
print("""
Per l'evento Lago Maggiore 2000, i dataset rilevanti sono:
- ERA5 Reanalysis (precipitazione, pressione, umidità suolo)
- CMEMS Reanalysis (SST, correnti marine)
- GNSS Tropospheric (vapore acqueo atmosferico)

Le catene causali attese:
1. ↑ Vapor acqueo (GNSS) → ↑ Precipitazione (ERA5) [lag: 6-12h]
2. ↑ Precipitazione intensa → ↑ Runoff → Flood [lag: 12-24h]
3. ↓ Pressione (msl) → Storm system → ↑ Precipitazione [lag: 12h]

Next Steps:
- [ ] Implementare download effettivo dati ERA5 per Oct 2000
- [ ] Creare visualization Kepler.gl con time animation
- [ ] Calcolare PCMCI su variabili precursor
""")

print("✅ Test completato!")
