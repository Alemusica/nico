#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SEED API REGISTRY TO SURREALDB                                  ║
║                                                                              ║
║   Imports all data source definitions from api_registry.py into SurrealDB.  ║
║   This creates a queryable catalog of all available APIs for the system.    ║
║                                                                              ║
║   Tables created:                                                            ║
║   - data_source: API metadata (endpoint, auth, variables, latency)          ║
║   - physics_var: Physics variable to data source mapping                    ║
║   - api_status: Current status and health of each API                       ║
║                                                                              ║
║   Usage:                                                                     ║
║     python scripts/seed_api_registry.py                                     ║
║                                                                              ║
║   Requires: SurrealDB running at localhost:8000                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import httpx
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.surge_shazam.data.api_registry import (
    API_REGISTRY,
    DataSource,
    DataCategory,
    Latency,
    Status,
    get_physics_variables_map,
)

# SurrealDB configuration
SURREAL_URL = "http://localhost:8000/sql"
SURREAL_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "surreal-ns": "surge_shazam",
    "surreal-db": "data_catalog",
    "Authorization": "Basic cm9vdDpyb290"  # root:root base64
}


def escape_string(s: str) -> str:
    """Escape a string for SurrealDB query."""
    if not s:
        return ""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '')


def source_to_surreal(source: DataSource) -> str:
    """Convert DataSource to SurrealDB CREATE query."""
    
    # Escape string fields
    name = escape_string(source.name)
    description = escape_string(source.description)
    base_url = escape_string(source.base_url)
    product_id = escape_string(source.product_id)
    docs_url = escape_string(source.docs_url)
    client_module = escape_string(source.client_module)
    
    # Convert auth config
    auth_json = json.dumps({
        "required": source.auth.required,
        "env_vars": source.auth.env_vars,
        "url_signup": source.auth.url_signup,
        "method": source.auth.method,
    })
    
    # Coverage temporal as array
    coverage = list(source.coverage_temporal)
    
    query = f'''
    DELETE data_source:{source.id};
    CREATE data_source:{source.id} SET
        id = "{source.id}",
        name = "{name}",
        category = "{source.category.value}",
        provider = "{source.provider}",
        base_url = "{base_url}",
        product_id = "{product_id}",
        variables = {json.dumps(source.variables)},
        spatial_resolution = "{source.spatial_resolution}",
        temporal_resolution = "{source.temporal_resolution}",
        coverage_spatial = "{source.coverage_spatial}",
        coverage_temporal = {json.dumps(coverage)},
        latency = "{source.latency.value}",
        latency_badge = "{source.latency.badge}",
        latency_hours = {source.latency.hours},
        auth = {auth_json},
        status = "{source.status.value}",
        client_module = "{client_module}",
        description = "{description}",
        docs_url = "{docs_url}",
        priority = "{source.priority}",
        physics_variables = {json.dumps(source.physics_variables)},
        imported_at = time::now()
    ;
    '''
    
    return query


def create_schema() -> str:
    """Create SurrealDB schema for data catalog."""
    return '''
    -- Data source table
    DEFINE TABLE IF NOT EXISTS data_source SCHEMAFULL;
    DEFINE FIELD id ON data_source TYPE string;
    DEFINE FIELD name ON data_source TYPE string;
    DEFINE FIELD category ON data_source TYPE string;
    DEFINE FIELD provider ON data_source TYPE string;
    DEFINE FIELD base_url ON data_source TYPE string;
    DEFINE FIELD product_id ON data_source TYPE string;
    DEFINE FIELD variables ON data_source TYPE array;
    DEFINE FIELD spatial_resolution ON data_source TYPE string;
    DEFINE FIELD temporal_resolution ON data_source TYPE string;
    DEFINE FIELD coverage_spatial ON data_source TYPE string;
    DEFINE FIELD coverage_temporal ON data_source TYPE array;
    DEFINE FIELD latency ON data_source TYPE string;
    DEFINE FIELD latency_badge ON data_source TYPE string;
    DEFINE FIELD latency_hours ON data_source TYPE int;
    DEFINE FIELD auth ON data_source TYPE object;
    DEFINE FIELD status ON data_source TYPE string;
    DEFINE FIELD client_module ON data_source TYPE string;
    DEFINE FIELD description ON data_source TYPE string;
    DEFINE FIELD docs_url ON data_source TYPE string;
    DEFINE FIELD priority ON data_source TYPE string;
    DEFINE FIELD physics_variables ON data_source TYPE array;
    DEFINE FIELD imported_at ON data_source TYPE datetime;
    
    -- Indexes
    DEFINE INDEX IF NOT EXISTS idx_category ON data_source FIELDS category;
    DEFINE INDEX IF NOT EXISTS idx_status ON data_source FIELDS status;
    DEFINE INDEX IF NOT EXISTS idx_latency ON data_source FIELDS latency_hours;
    DEFINE INDEX IF NOT EXISTS idx_priority ON data_source FIELDS priority;
    
    -- Physics variable mapping table
    DEFINE TABLE IF NOT EXISTS physics_var SCHEMAFULL;
    DEFINE FIELD name ON physics_var TYPE string;
    DEFINE FIELD sources ON physics_var TYPE array;
    DEFINE FIELD swe_symbol ON physics_var TYPE string;
    DEFINE FIELD description ON physics_var TYPE string;
    
    -- API status/health table
    DEFINE TABLE IF NOT EXISTS api_status SCHEMAFULL;
    DEFINE FIELD source_id ON api_status TYPE string;
    DEFINE FIELD last_check ON api_status TYPE datetime;
    DEFINE FIELD is_healthy ON api_status TYPE bool;
    DEFINE FIELD response_time_ms ON api_status TYPE int;
    DEFINE FIELD error_message ON api_status TYPE string;
    '''


def create_physics_var_queries() -> str:
    """Create queries for physics variable mappings."""
    physics_map = get_physics_variables_map()
    
    # Physics variable descriptions
    var_info = {
        "η": ("sea_level", "Sea surface height / surge height [m]"),
        "η_obs": ("observed_sea_level", "Observed sea level from tide gauges [m]"),
        "u_geo": ("geostrophic_u", "Eastward geostrophic velocity [m/s]"),
        "v_geo": ("geostrophic_v", "Northward geostrophic velocity [m/s]"),
        "T_surface": ("surface_temperature", "Sea surface temperature [K]"),
        "H_wave": ("wave_height", "Significant wave height [m]"),
        "τ_wind": ("wind_stress", "Wind stress [N/m²]"),
        "P_atm": ("atmospheric_pressure", "Atmospheric pressure [Pa]"),
        "precip": ("precipitation", "Precipitation rate [mm/day]"),
        "U_wind": ("wind_speed", "10m wind speed [m/s]"),
        "T_air": ("air_temperature", "Air temperature [K]"),
        "q_humidity": ("specific_humidity", "Specific humidity [kg/kg]"),
        "TWS": ("total_water_storage", "Terrestrial water storage [m]"),
        "T_ocean": ("ocean_temperature", "Ocean temperature profile [K]"),
        "S_ocean": ("ocean_salinity", "Ocean salinity [PSU]"),
        "NAO": ("nao_index", "North Atlantic Oscillation index"),
        "ENSO": ("enso_index", "El Niño Southern Oscillation index"),
    }
    
    queries = []
    for pvar, sources in physics_map.items():
        info = var_info.get(pvar, (pvar, f"Physics variable: {pvar}"))
        swe_symbol = pvar
        name = info[0]
        description = escape_string(info[1])
        
        query = f'''
        DELETE physics_var:{name};
        CREATE physics_var:{name} SET
            name = "{name}",
            swe_symbol = "{swe_symbol}",
            sources = {json.dumps(sources)},
            description = "{description}"
        ;
        '''
        queries.append(query)
    
    return '\n'.join(queries)


def execute_query(query: str) -> bool:
    """Execute a SurrealDB query."""
    try:
        response = httpx.post(
            SURREAL_URL,
            headers=SURREAL_HEADERS,
            content=query,
            timeout=30.0
        )
        
        if response.status_code == 200:
            result = response.json()
            # Check for errors
            for r in result:
                if r.get('status') == 'ERR':
                    print(f"   ⚠️ Query error: {r.get('detail', 'Unknown')}")
                    return False
            return True
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    print("=" * 70)
    print("  SEED API REGISTRY TO SURREALDB")
    print("=" * 70)
    print()
    
    # Test connection
    print(f"🔗 Connecting to SurrealDB at {SURREAL_URL}...")
    try:
        test_response = httpx.post(
            SURREAL_URL,
            headers=SURREAL_HEADERS,
            content="INFO FOR DB;",
            timeout=5.0
        )
        if test_response.status_code != 200:
            print(f"❌ Connection failed: {test_response.status_code}")
            print(f"   Response: {test_response.text}")
            print("\n   Make sure SurrealDB is running:")
            print("   surreal start --user root --pass root file:~/.surrealdb/data/surge_shazam.db")
            return 1
        print("✅ Connected!")
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        print("\n   Start SurrealDB with:")
        print("   surreal start --user root --pass root file:~/.surrealdb/data/surge_shazam.db")
        return 1
    
    # Create schema
    print("\n📐 Creating schema...")
    if execute_query(create_schema()):
        print("   ✅ Schema created")
    else:
        print("   ⚠️ Schema creation had issues")
    
    # Import data sources
    print(f"\n📡 Importing {len(API_REGISTRY)} data sources...")
    
    success_count = 0
    by_category = {}
    by_status = {}
    
    for source_id, source in API_REGISTRY.items():
        # Track stats
        cat = source.category.value
        by_category[cat] = by_category.get(cat, 0) + 1
        stat = source.status.value
        by_status[stat] = by_status.get(stat, 0) + 1
        
        # Create query
        query = source_to_surreal(source)
        
        print(f"   {source.latency.badge} {source_id}: {source.name}...")
        
        if execute_query(query):
            success_count += 1
            print(f"      ✅ Saved ({source.status.value})")
        else:
            print(f"      ⚠️ Failed")
    
    # Import physics variable mappings
    print("\n⚛️ Importing physics variable mappings...")
    if execute_query(create_physics_var_queries()):
        print("   ✅ Physics variables mapped")
    else:
        print("   ⚠️ Physics mapping had issues")
    
    # Summary
    print("\n" + "=" * 70)
    print("  IMPORT SUMMARY")
    print("=" * 70)
    print(f"\nTotal sources: {len(API_REGISTRY)}")
    print(f"Successfully imported: {success_count}")
    
    print("\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"   {cat:15}: {count}")
    
    print("\nBy status:")
    for stat, count in sorted(by_status.items()):
        emoji = "✅" if stat == "available" else "⚠️" if stat == "partial" else "🔧"
        print(f"   {emoji} {stat:15}: {count}")
    
    # Example queries
    print("\n📋 Example SurrealDB queries:")
    print('''
    # Get all real-time sources
    SELECT * FROM data_source WHERE latency_hours <= 6;
    
    # Get sources for sea level (η)
    SELECT * FROM data_source WHERE physics_variables CONTAINS "η";
    
    # Get HIGH priority sources that need implementation
    SELECT * FROM data_source WHERE priority = "HIGH" AND status = "todo";
    
    # Physics variables summary
    SELECT * FROM physics_var;
    ''')
    
    print("\n✅ Seeding complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
