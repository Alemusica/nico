#!/usr/bin/env python3
"""
Seed Knowledge Graph directly via SurrealDB queries.
Run: python scripts/seed_knowledge_graph.py
"""
from uuid import uuid4

# SurrealDB connection
SURREAL_URL = "ws://localhost:8001/rpc"
NAMESPACE = "causal"
DATABASE = "knowledge"


def seed_data():
    """Populate SurrealDB with demo knowledge graph data."""
    from surrealdb import Surreal
    
    print("🔗 Connecting to SurrealDB...")
    db = Surreal(SURREAL_URL)
    db.use(NAMESPACE, DATABASE)
    
    # ============================================
    # EVENTS
    # ============================================
    events = [
        {
            "id": f"event_{uuid4().hex[:8]}",
            "name": "Alluvione Lago Maggiore 2000",
            "event_type": "flood",
            "start_date": "2000-10-13T00:00:00Z",
            "end_date": "2000-10-16T00:00:00Z",
            "location": {"lat": 45.9, "lon": 8.6, "name": "Lago Maggiore, Italy"},
            "severity": 0.95,
            "description": "Major flooding event affecting the Lago Maggiore region with record water levels",
            "source": "historical_records"
        },
        {
            "id": f"event_{uuid4().hex[:8]}",
            "name": "NAO Negative Phase Oct 2000",
            "event_type": "climate_anomaly",
            "start_date": "2000-10-01T00:00:00Z",
            "end_date": "2000-10-31T00:00:00Z",
            "severity": 0.8,
            "description": "Strong negative NAO phase (NAO index = -2.1) preceding flooding",
            "source": "NOAA_CPC"
        },
        {
            "id": f"event_{uuid4().hex[:8]}",
            "name": "Mediterranean Cyclone Davide",
            "event_type": "storm",
            "start_date": "2000-10-10T00:00:00Z",
            "end_date": "2000-10-14T00:00:00Z",
            "location": {"lat": 43.0, "lon": 10.0, "name": "Ligurian Sea"},
            "severity": 0.85,
            "description": "Cyclonic system bringing heavy precipitation to northern Alps",
            "source": "ERA5_reanalysis"
        },
        {
            "id": f"event_{uuid4().hex[:8]}",
            "name": "North Atlantic Storm System",
            "event_type": "storm",
            "start_date": "2000-10-08T00:00:00Z",
            "end_date": "2000-10-12T00:00:00Z",
            "location": {"lat": 50.0, "lon": -20.0, "name": "North Atlantic"},
            "severity": 0.6,
            "description": "Atlantic low pressure system contributing to moisture transport",
            "source": "ERA5_reanalysis"
        },
        {
            "id": f"event_{uuid4().hex[:8]}",
            "name": "Venice Acqua Alta Nov 2019",
            "event_type": "flood",
            "start_date": "2019-11-12T00:00:00Z",
            "end_date": "2019-11-17T00:00:00Z",
            "location": {"lat": 45.4, "lon": 12.3, "name": "Venice, Italy"},
            "severity": 0.98,
            "description": "Record high tide reaching 187cm, second highest in history",
            "source": "ISPRA"
        },
        {
            "id": f"event_{uuid4().hex[:8]}",
            "name": "Genoa Flash Flood 2011",
            "event_type": "flood",
            "start_date": "2011-11-04T00:00:00Z",
            "location": {"lat": 44.4, "lon": 8.9, "name": "Genoa, Italy"},
            "severity": 0.92,
            "description": "Flash flood with 500mm rain in 6 hours",
            "source": "historical_records"
        },
    ]
    
    print("\n📍 Adding events...")
    for e in events:
        try:
            db.query(f"CREATE event:{e['id']} CONTENT $data", {"data": e})
            print(f"  ✓ {e['name']}")
        except Exception as ex:
            print(f"  ✗ {e['name']}: {ex}")
    
    # ============================================
    # PAPERS
    # ============================================
    papers = [
        {
            "id": f"paper_{uuid4().hex[:8]}",
            "title": "The October 2000 flooding in the Po Valley: Meteorological analysis",
            "authors": ["Buzzi A.", "Tartaglione N.", "Malguzzi P."],
            "year": 2001,
            "journal": "Natural Hazards and Earth System Sciences",
            "keywords": ["flooding", "Po Valley", "meteorology", "cyclone", "Alps"],
            "abstract": "Analysis of the meteorological conditions leading to the October 2000 floods in northern Italy."
        },
        {
            "id": f"paper_{uuid4().hex[:8]}",
            "title": "NAO influence on Alpine precipitation extremes",
            "authors": ["Beniston M.", "Jungo P."],
            "year": 2002,
            "journal": "Climate Dynamics",
            "keywords": ["NAO", "Alps", "precipitation", "extremes"],
            "abstract": "Study on how North Atlantic Oscillation affects precipitation patterns in the Alpine region."
        },
        {
            "id": f"paper_{uuid4().hex[:8]}",
            "title": "PCMCI-based causal discovery of flood precursors",
            "authors": ["Runge J.", "Nowack P.", "Kretschmer M."],
            "year": 2023,
            "journal": "Journal of Hydrology",
            "keywords": ["PCMCI", "causal discovery", "floods", "precursors"],
            "abstract": "Application of PCMCI algorithm to identify causal atmospheric precursors to flood events."
        },
        {
            "id": f"paper_{uuid4().hex[:8]}",
            "title": "Satellite altimetry for flood monitoring: SLCCI validation",
            "authors": ["ESA CCI Lakes Team", "Crétaux J.F."],
            "year": 2022,
            "journal": "Remote Sensing of Environment",
            "keywords": ["satellite", "altimetry", "lakes", "SLCCI"],
            "abstract": "Validation of ESA CCI Lakes satellite altimetry data for monitoring water levels."
        },
        {
            "id": f"paper_{uuid4().hex[:8]}",
            "title": "Atmospheric rivers and European flooding",
            "authors": ["Lavers D.A.", "Villarini G."],
            "year": 2020,
            "journal": "Nature Communications",
            "keywords": ["atmospheric rivers", "flooding", "Europe"],
            "abstract": "Analysis of atmospheric river events as drivers of major flooding in western Europe."
        },
        {
            "id": f"paper_{uuid4().hex[:8]}",
            "title": "Sea level rise and storm surge in the Adriatic",
            "authors": ["Lionello P.", "Sanna A."],
            "year": 2021,
            "journal": "Journal of Coastal Research",
            "keywords": ["sea level", "storm surge", "Adriatic", "Venice"],
            "abstract": "Investigation of compound flooding from sea level rise and storm surges in Venice."
        },
    ]
    
    print("\n📚 Adding papers...")
    for p in papers:
        try:
            db.query(f"CREATE paper:{p['id']} CONTENT $data", {"data": p})
            print(f"  ✓ {p['title'][:50]}...")
        except Exception as ex:
            print(f"  ✗ {p['title'][:30]}...: {ex}")
    
    # ============================================
    # PATTERNS (causal relationships)
    # ============================================
    patterns = [
        {
            "id": f"pattern_{uuid4().hex[:8]}",
            "name": "NAO-Precipitation Chain",
            "pattern_type": "causal_chain",
            "description": "NAO negative → Mediterranean moisture → Alpine precipitation → Flooding",
            "confidence": 0.85,
            "variables": ["NAO", "IVT", "precipitation", "runoff"],
            "lag_days": 7,
            "strength": 0.72
        },
        {
            "id": f"pattern_{uuid4().hex[:8]}",
            "name": "Atlantic SST Teleconnection",
            "pattern_type": "teleconnection",
            "description": "Atlantic SST anomaly → Jet stream shift → Storm track toward Alps",
            "confidence": 0.72,
            "variables": ["SST_atlantic", "geopotential_500", "storm_track"],
            "lag_days": 10,
            "strength": 0.65
        },
        {
            "id": f"pattern_{uuid4().hex[:8]}",
            "name": "Blocking Precursor",
            "pattern_type": "precursor",
            "description": "Atlantic blocking → Cyclone stalling → Persistent precipitation",
            "confidence": 0.78,
            "variables": ["blocking_index", "cyclone_position", "precipitation"],
            "lag_days": 5,
            "strength": 0.68
        },
        {
            "id": f"pattern_{uuid4().hex[:8]}",
            "name": "Venice Compound Flood",
            "pattern_type": "compound",
            "description": "High tide + Storm surge + River discharge → Acqua Alta",
            "confidence": 0.92,
            "variables": ["tide", "storm_surge", "river_discharge", "water_level"],
            "lag_days": 0,
            "strength": 0.88
        },
        {
            "id": f"pattern_{uuid4().hex[:8]}",
            "name": "Mediterranean Feedback",
            "pattern_type": "feedback",
            "description": "Warm SST → Evaporation → Intense precipitation → Runoff",
            "confidence": 0.68,
            "variables": ["SST_med", "evaporation", "precipitation", "runoff"],
            "lag_days": 4,
            "strength": 0.55
        },
    ]
    
    print("\n🔗 Adding patterns...")
    for pt in patterns:
        try:
            db.query(f"CREATE pattern:{pt['id']} CONTENT $data", {"data": pt})
            print(f"  ✓ {pt['pattern_type']}: {pt['name']}")
        except Exception as ex:
            print(f"  ✗ {pt['pattern_type']}: {ex}")
    
    # ============================================
    # Get stats
    # ============================================
    events_count = db.query("SELECT count() FROM event GROUP ALL")
    papers_count = db.query("SELECT count() FROM paper GROUP ALL")
    patterns_count = db.query("SELECT count() FROM pattern GROUP ALL")
    
    def get_count(result):
        if result and len(result) > 0:
            if isinstance(result[0], dict):
                return result[0].get('count', 0)
            elif isinstance(result[0], list) and len(result[0]) > 0:
                return result[0][0].get('count', 0)
        return 0
    
    print(f"\n{'='*50}")
    print(f"✅ Knowledge Graph seeded successfully!")
    print(f"{'='*50}")
    print(f"  📍 Events:   {get_count(events_count)}")
    print(f"  📚 Papers:   {get_count(papers_count)}")
    print(f"  🔗 Patterns: {get_count(patterns_count)}")
    print(f"{'='*50}")
    
    db.close()


if __name__ == "__main__":
    seed_data()
