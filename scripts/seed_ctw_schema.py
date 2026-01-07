#!/usr/bin/env python3
"""
Seed API Registry - CTW Schema Setup and Demo Data

Crea lo schema SurrealDB ottimizzato per Climate Tipping Warning e popola
con dati demo per observation, causal_link, fingerprint, alert.

Usage:
    python scripts/seed_api_registry.py [--schema-only] [--data-only] [--reset]

Requirements:
    pip install surrealdb numpy
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np

# Configurazione
SURREAL_URL = "ws://localhost:8000/rpc"
NAMESPACE = "causal"
DATABASE = "knowledge"
USERNAME = "root"
PASSWORD = "root"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Schema Definitions
# =============================================================================

SCHEMA_CTW = """
-- ============================================
-- OBSERVATION - Dati satellitari/aircraft
-- ============================================
DEFINE TABLE observation SCHEMAFULL;
DEFINE FIELD id ON observation TYPE string;
DEFINE FIELD source ON observation TYPE string;
DEFINE FIELD source_id ON observation TYPE option<string>;
DEFINE FIELD timestamp ON observation TYPE datetime;
DEFINE FIELD time_resolution ON observation TYPE string;
DEFINE FIELD location ON observation TYPE object;
DEFINE FIELD region ON observation TYPE option<string>;
DEFINE FIELD altitude ON observation TYPE option<float>;
DEFINE FIELD variables ON observation TYPE object;
DEFINE FIELD quality_flag ON observation TYPE option<int>;
DEFINE FIELD processing_level ON observation TYPE option<string>;
DEFINE FIELD metadata ON observation TYPE option<object>;
DEFINE FIELD created_at ON observation TYPE datetime DEFAULT time::now();

DEFINE INDEX obs_id ON observation FIELDS id UNIQUE;
DEFINE INDEX obs_source ON observation FIELDS source;
DEFINE INDEX obs_timestamp ON observation FIELDS timestamp;
DEFINE INDEX obs_region ON observation FIELDS region;
DEFINE INDEX obs_source_time ON observation FIELDS source, timestamp;
DEFINE INDEX obs_spatiotemporal ON observation FIELDS region, timestamp;

-- ============================================
-- CAUSAL_LINK - Relazioni PCMCI
-- ============================================
DEFINE TABLE causal_link SCHEMAFULL;
DEFINE FIELD id ON causal_link TYPE string;
DEFINE FIELD name ON causal_link TYPE option<string>;
DEFINE FIELD driver ON causal_link TYPE string;
DEFINE FIELD target ON causal_link TYPE string;
DEFINE FIELD driver_region ON causal_link TYPE option<string>;
DEFINE FIELD target_region ON causal_link TYPE option<string>;
DEFINE FIELD lag ON causal_link TYPE int;
DEFINE FIELD lag_unit ON causal_link TYPE string;
DEFINE FIELD strength ON causal_link TYPE float;
DEFINE FIELD p_value ON causal_link TYPE float;
DEFINE FIELD confidence_interval ON causal_link TYPE option<array<float>>;
DEFINE FIELD link_type ON causal_link TYPE string;
DEFINE FIELD mechanism ON causal_link TYPE option<string>;
DEFINE FIELD is_validated ON causal_link TYPE bool DEFAULT false;
DEFINE FIELD validation_source ON causal_link TYPE option<string>;
DEFINE FIELD valid_from ON causal_link TYPE option<datetime>;
DEFINE FIELD valid_to ON causal_link TYPE option<datetime>;
DEFINE FIELD seasonality ON causal_link TYPE option<array<string>>;
DEFINE FIELD pcmci_params ON causal_link TYPE option<object>;
DEFINE FIELD created_at ON causal_link TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON causal_link TYPE option<datetime>;

DEFINE INDEX causal_id ON causal_link FIELDS id UNIQUE;
DEFINE INDEX causal_driver ON causal_link FIELDS driver;
DEFINE INDEX causal_target ON causal_link FIELDS target;
DEFINE INDEX causal_strength ON causal_link FIELDS strength;
DEFINE INDEX causal_pvalue ON causal_link FIELDS p_value;
DEFINE INDEX causal_lag ON causal_link FIELDS lag;
DEFINE INDEX causal_validated ON causal_link FIELDS is_validated;

-- ============================================
-- FINGERPRINT - Embedding MiniRocket
-- ============================================
DEFINE TABLE fingerprint SCHEMAFULL;
DEFINE FIELD id ON fingerprint TYPE string;
DEFINE FIELD name ON fingerprint TYPE string;
DEFINE FIELD description ON fingerprint TYPE option<string>;
DEFINE FIELD embedding ON fingerprint TYPE array<float>;
DEFINE FIELD embedding_version ON fingerprint TYPE string;
DEFINE FIELD source_type ON fingerprint TYPE string;
DEFINE FIELD source_id ON fingerprint TYPE string;
DEFINE FIELD window_start ON fingerprint TYPE datetime;
DEFINE FIELD window_end ON fingerprint TYPE datetime;
DEFINE FIELD variables_used ON fingerprint TYPE array<string>;
DEFINE FIELD region ON fingerprint TYPE option<string>;
DEFINE FIELD stats ON fingerprint TYPE option<object>;
DEFINE FIELD created_at ON fingerprint TYPE datetime DEFAULT time::now();
DEFINE FIELD model_config ON fingerprint TYPE option<object>;

DEFINE INDEX fp_id ON fingerprint FIELDS id UNIQUE;
DEFINE INDEX fp_source ON fingerprint FIELDS source_type, source_id;
DEFINE INDEX fp_region ON fingerprint FIELDS region;

-- Vector index HNSW per similarity search (100 dimensions)
DEFINE INDEX fp_embedding_hnsw ON fingerprint FIELDS embedding HNSW DIMENSION 100 DIST COSINE;

-- ============================================
-- ALERT - Warning generati
-- ============================================
DEFINE TABLE alert SCHEMAFULL;
DEFINE FIELD id ON alert TYPE string;
DEFINE FIELD alert_code ON alert TYPE string;
DEFINE FIELD alert_type ON alert TYPE string;
DEFINE FIELD severity ON alert TYPE string;
DEFINE FIELD confidence ON alert TYPE float;
DEFINE FIELD issued_at ON alert TYPE datetime;
DEFINE FIELD valid_from ON alert TYPE datetime;
DEFINE FIELD valid_until ON alert TYPE datetime;
DEFINE FIELD lead_time_hours ON alert TYPE int;
DEFINE FIELD affected_region ON alert TYPE string;
DEFINE FIELD affected_area ON alert TYPE option<object>;
DEFINE FIELD affected_population ON alert TYPE option<int>;
DEFINE FIELD headline ON alert TYPE string;
DEFINE FIELD description ON alert TYPE string;
DEFINE FIELD recommended_actions ON alert TYPE option<array<string>>;
DEFINE FIELD trigger_type ON alert TYPE string;
DEFINE FIELD trigger_details ON alert TYPE object;
DEFINE FIELD status ON alert TYPE string DEFAULT "active";
DEFINE FIELD verification ON alert TYPE option<object>;
DEFINE FIELD created_at ON alert TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON alert TYPE option<datetime>;
DEFINE FIELD created_by ON alert TYPE option<string>;

DEFINE INDEX alert_id ON alert FIELDS id UNIQUE;
DEFINE INDEX alert_code ON alert FIELDS alert_code UNIQUE;
DEFINE INDEX alert_type ON alert FIELDS alert_type;
DEFINE INDEX alert_severity ON alert FIELDS severity;
DEFINE INDEX alert_status ON alert FIELDS status;
DEFINE INDEX alert_issued ON alert FIELDS issued_at;
DEFINE INDEX alert_region ON alert FIELDS affected_region;
DEFINE INDEX alert_valid ON alert FIELDS valid_from, valid_until;

-- ============================================
-- EDGE TABLES (Relazioni)
-- ============================================

-- Causal link triggera evento
DEFINE TABLE triggers SCHEMAFULL TYPE RELATION FROM causal_link TO event;
DEFINE FIELD contribution ON triggers TYPE float;
DEFINE FIELD lag_observed ON triggers TYPE option<int>;
DEFINE FIELD notes ON triggers TYPE option<string>;

-- Observation ha fingerprint
DEFINE TABLE has_fingerprint SCHEMAFULL TYPE RELATION FROM observation TO fingerprint;
DEFINE FIELD extraction_params ON has_fingerprint TYPE option<object>;

-- Alert matched by fingerprint
DEFINE TABLE matched_by SCHEMAFULL TYPE RELATION FROM alert TO fingerprint;
DEFINE FIELD similarity_score ON matched_by TYPE float;
DEFINE FIELD matched_at ON matched_by TYPE datetime DEFAULT time::now();

-- Alert refers to event (verifica)
DEFINE TABLE refers_to SCHEMAFULL TYPE RELATION FROM alert TO event;
DEFINE FIELD was_correct ON refers_to TYPE bool;
DEFINE FIELD temporal_error_hours ON refers_to TYPE option<int>;
DEFINE FIELD spatial_error_km ON refers_to TYPE option<float>;

-- Causal link derived from observation
DEFINE TABLE derived_from SCHEMAFULL TYPE RELATION FROM causal_link TO observation;
DEFINE FIELD analysis_id ON derived_from TYPE option<string>;
DEFINE FIELD contribution_weight ON derived_from TYPE option<float>;
"""


# =============================================================================
# Demo Data
# =============================================================================

def generate_demo_observations() -> list[dict[str, Any]]:
    """Genera observation demo da diverse sorgenti."""
    observations = []
    base_time = datetime(2000, 10, 10, 0, 0, 0)  # Periodo Lago Maggiore 2000
    
    sources = [
        {"source": "era5", "region": "alpine", "vars": ["precipitation", "mslp", "wind_u10", "wind_v10"]},
        {"source": "cmems", "region": "mediterranean", "vars": ["sst", "sea_level"]},
        {"source": "slcci", "region": "alpine", "vars": ["lake_level", "lake_area"]},
    ]
    
    for src in sources:
        for hour in range(0, 168, 6):  # 7 giorni, ogni 6 ore
            timestamp = base_time + timedelta(hours=hour)
            
            # Simula variabili con trend verso l'evento
            progress = hour / 168  # 0 -> 1
            
            variables = {}
            if "precipitation" in src["vars"]:
                # Precipitazione aumenta verso il picco
                variables["precipitation"] = max(0, 5 + 80 * progress + np.random.normal(0, 10))
            if "mslp" in src["vars"]:
                # Pressione scende
                variables["mslp"] = 1015 - 20 * progress + np.random.normal(0, 2)
            if "sst" in src["vars"]:
                variables["sst"] = 20 + np.random.normal(0, 0.5)
            if "sea_level" in src["vars"]:
                variables["sea_level"] = 0.2 + 0.5 * progress + np.random.normal(0, 0.05)
            if "lake_level" in src["vars"]:
                variables["lake_level"] = 193.5 + 2.5 * progress + np.random.normal(0, 0.1)
            if "wind_u10" in src["vars"]:
                variables["wind_u10"] = 5 + 10 * progress + np.random.normal(0, 2)
            if "wind_v10" in src["vars"]:
                variables["wind_v10"] = -3 - 8 * progress + np.random.normal(0, 2)
            
            # Location basata su regione
            if src["region"] == "alpine":
                lon, lat = 8.6 + np.random.uniform(-0.5, 0.5), 45.9 + np.random.uniform(-0.3, 0.3)
            else:
                lon, lat = 9.0 + np.random.uniform(-2, 2), 43.0 + np.random.uniform(-1, 1)
            
            obs = {
                "id": f"obs_{src['source']}_{uuid4().hex[:8]}",
                "source": src["source"],
                "timestamp": timestamp.isoformat() + "Z",
                "time_resolution": "6hourly",
                "location": {"type": "Point", "coordinates": [lon, lat]},
                "region": src["region"],
                "variables": variables,
                "quality_flag": 0,
                "processing_level": "L3",
            }
            observations.append(obs)
    
    return observations


def generate_demo_causal_links() -> list[dict[str, Any]]:
    """Genera causal_link demo basati su PCMCI analysis."""
    links = [
        {
            "id": f"cl_{uuid4().hex[:8]}",
            "name": "NAO → IVT",
            "driver": "NAO_index",
            "target": "IVT",
            "driver_region": "north_atlantic",
            "target_region": "alpine",
            "lag": 7,
            "lag_unit": "days",
            "strength": -0.72,
            "p_value": 0.001,
            "confidence_interval": [-0.82, -0.62],
            "link_type": "direct",
            "mechanism": "NAO negative phase shifts jet stream southward, increasing moisture transport to Alps",
            "is_validated": True,
            "validation_source": "Beniston2002",
            "seasonality": ["SON", "DJF"],
            "pcmci_params": {"tau_max": 10, "pc_alpha": 0.05, "cond_ind_test": "ParCorr"},
        },
        {
            "id": f"cl_{uuid4().hex[:8]}",
            "name": "IVT → Precipitation",
            "driver": "IVT",
            "target": "precipitation",
            "driver_region": "alpine",
            "target_region": "alpine",
            "lag": 1,
            "lag_unit": "days",
            "strength": 0.85,
            "p_value": 0.0001,
            "confidence_interval": [0.78, 0.92],
            "link_type": "direct",
            "mechanism": "Integrated vapor transport delivers moisture that precipitates on orographic lift",
            "is_validated": True,
            "validation_source": "Lavers2020",
            "seasonality": ["SON", "DJF", "MAM"],
        },
        {
            "id": f"cl_{uuid4().hex[:8]}",
            "name": "Precipitation → Lake Level",
            "driver": "precipitation",
            "target": "lake_level",
            "driver_region": "alpine",
            "target_region": "alpine",
            "lag": 2,
            "lag_unit": "days",
            "strength": 0.78,
            "p_value": 0.0005,
            "confidence_interval": [0.68, 0.88],
            "link_type": "direct",
            "mechanism": "Runoff from Alpine catchment feeds into lake system",
            "is_validated": True,
            "validation_source": "Buzzi2001",
        },
        {
            "id": f"cl_{uuid4().hex[:8]}",
            "name": "SST Med → Evaporation",
            "driver": "sst_mediterranean",
            "target": "evaporation",
            "driver_region": "mediterranean",
            "target_region": "mediterranean",
            "lag": 0,
            "lag_unit": "days",
            "strength": 0.65,
            "p_value": 0.01,
            "link_type": "contemporaneous",
            "mechanism": "Warm SST increases latent heat flux and evaporation",
            "is_validated": False,
        },
        {
            "id": f"cl_{uuid4().hex[:8]}",
            "name": "Blocking → Cyclone Stalling",
            "driver": "blocking_index",
            "target": "cyclone_residence_time",
            "driver_region": "north_atlantic",
            "target_region": "mediterranean",
            "lag": 3,
            "lag_unit": "days",
            "strength": 0.58,
            "p_value": 0.02,
            "link_type": "indirect",
            "mechanism": "Atlantic blocking prevents cyclone eastward progression",
            "is_validated": True,
            "validation_source": "expert_validation",
        },
    ]
    return links


def generate_demo_fingerprints() -> list[dict[str, Any]]:
    """Genera fingerprint demo con embedding MiniRocket simulati."""
    fingerprints = []
    
    events = [
        {"name": "Lago Maggiore 2000", "event_id": "event_lago2000", "region": "alpine"},
        {"name": "Venice Acqua Alta 2019", "event_id": "event_venice2019", "region": "adriatic"},
        {"name": "Genoa Flash Flood 2011", "event_id": "event_genoa2011", "region": "ligurian"},
    ]
    
    for event in events:
        # Genera embedding pseudo-random ma riproducibile per evento
        np.random.seed(hash(event["event_id"]) % (2**32))
        embedding = np.random.randn(100).tolist()
        
        # Normalizza
        norm = np.linalg.norm(embedding)
        embedding = [e / norm for e in embedding]
        
        fp = {
            "id": f"fp_{uuid4().hex[:8]}",
            "name": f"Fingerprint {event['name']}",
            "description": f"MiniRocket embedding for {event['name']} event precursor pattern",
            "embedding": embedding,
            "embedding_version": "minirocket_v1.0",
            "source_type": "event",
            "source_id": event["event_id"],
            "window_start": "2000-10-01T00:00:00Z",
            "window_end": "2000-10-13T00:00:00Z",
            "variables_used": ["precipitation", "mslp", "ivt", "lake_level"],
            "region": event["region"],
            "stats": {
                "mean": float(np.mean(embedding)),
                "std": float(np.std(embedding)),
                "min": float(np.min(embedding)),
                "max": float(np.max(embedding)),
            },
            "model_config": {
                "n_kernels": 10000,
                "max_dilations_per_kernel": 32,
            },
        }
        fingerprints.append(fp)
    
    return fingerprints


def generate_demo_alerts() -> list[dict[str, Any]]:
    """Genera alert demo."""
    alerts = [
        {
            "id": f"alert_{uuid4().hex[:8]}",
            "alert_code": "CTW-2000-001-FLOOD-ALPINE",
            "alert_type": "flood",
            "severity": "warning",
            "confidence": 0.87,
            "issued_at": "2000-10-11T06:00:00Z",
            "valid_from": "2000-10-13T00:00:00Z",
            "valid_until": "2000-10-16T00:00:00Z",
            "lead_time_hours": 42,
            "affected_region": "alpine",
            "affected_population": 500000,
            "headline": "Major flood risk for Lago Maggiore region",
            "description": "Pattern matching indicates high probability of significant flooding. "
                          "NAO negative phase combined with Mediterranean moisture transport "
                          "suggests 48-72h window of extreme precipitation.",
            "recommended_actions": [
                "Monitor lake levels hourly",
                "Prepare evacuation routes for lakeside communities",
                "Alert emergency services",
                "Issue public warning",
            ],
            "trigger_type": "fingerprint_match",
            "trigger_details": {
                "matched_fingerprint_id": "fp_lago2000",
                "similarity_score": 0.87,
                "causal_chain": ["NAO-", "IVT+", "precipitation+", "lake_level+"],
            },
            "status": "verified",
            "verification": {
                "verified_at": "2000-10-20T00:00:00Z",
                "actual_event_id": "event_lago2000",
                "hit_rate": 0.92,
                "false_alarm_rate": 0.08,
            },
            "created_by": "system",
        },
        {
            "id": f"alert_{uuid4().hex[:8]}",
            "alert_code": "CTW-2024-TEST-001",
            "alert_type": "flood",
            "severity": "advisory",
            "confidence": 0.65,
            "issued_at": datetime.now().isoformat() + "Z",
            "valid_from": (datetime.now() + timedelta(days=2)).isoformat() + "Z",
            "valid_until": (datetime.now() + timedelta(days=5)).isoformat() + "Z",
            "lead_time_hours": 48,
            "affected_region": "mediterranean",
            "headline": "Elevated flood risk - Mediterranean coast",
            "description": "Moderate fingerprint match detected. Monitoring recommended.",
            "trigger_type": "threshold",
            "trigger_details": {
                "threshold_exceeded": ["ivt", "precipitation_forecast"],
                "current_values": {"ivt": 420, "precip_forecast": 85},
            },
            "status": "active",
            "created_by": "system",
        },
    ]
    return alerts


# =============================================================================
# Database Operations
# =============================================================================

def connect_db():
    """Connetti a SurrealDB."""
    try:
        from surrealdb import Surreal
    except ImportError:
        logger.error("❌ surrealdb package non installato. Esegui: pip install surrealdb")
        sys.exit(1)
    
    logger.info(f"🔗 Connessione a {SURREAL_URL}...")
    db = Surreal(SURREAL_URL)
    db.signin({"user": USERNAME, "pass": PASSWORD})
    db.use(NAMESPACE, DATABASE)
    logger.info(f"✅ Connesso a {NAMESPACE}/{DATABASE}")
    return db


def apply_schema(db, reset: bool = False):
    """Applica lo schema CTW."""
    logger.info("📋 Applicazione schema CTW...")
    
    if reset:
        logger.warning("⚠️  Reset richiesto - eliminazione tabelle esistenti...")
        tables = ["observation", "causal_link", "fingerprint", "alert", 
                  "triggers", "has_fingerprint", "matched_by", "refers_to", "derived_from"]
        for table in tables:
            try:
                db.query(f"REMOVE TABLE {table};")
                logger.info(f"   🗑️  Rimossa tabella {table}")
            except Exception:
                pass
    
    # Applica schema in blocchi
    statements = [s.strip() for s in SCHEMA_CTW.split(";") if s.strip() and not s.strip().startswith("--")]
    
    for stmt in statements:
        if stmt:
            try:
                db.query(stmt + ";")
            except Exception as e:
                # Ignora errori di "già esiste"
                if "already exists" not in str(e).lower():
                    logger.warning(f"   ⚠️  {stmt[:50]}...: {e}")
    
    logger.info("✅ Schema CTW applicato")


def seed_data(db):
    """Inserisce dati demo."""
    logger.info("🌱 Seeding dati demo...")
    
    # Observations
    observations = generate_demo_observations()
    logger.info(f"   📡 Inserimento {len(observations)} observations...")
    for obs in observations:
        try:
            db.query(f"CREATE observation:{obs['id']} CONTENT $data;", {"data": obs})
        except Exception as e:
            logger.warning(f"      ⚠️  {obs['id']}: {e}")
    
    # Causal Links
    causal_links = generate_demo_causal_links()
    logger.info(f"   🔗 Inserimento {len(causal_links)} causal_link...")
    for cl in causal_links:
        try:
            db.query(f"CREATE causal_link:{cl['id']} CONTENT $data;", {"data": cl})
        except Exception as e:
            logger.warning(f"      ⚠️  {cl['id']}: {e}")
    
    # Fingerprints
    fingerprints = generate_demo_fingerprints()
    logger.info(f"   🎵 Inserimento {len(fingerprints)} fingerprint...")
    for fp in fingerprints:
        try:
            db.query(f"CREATE fingerprint:{fp['id']} CONTENT $data;", {"data": fp})
        except Exception as e:
            logger.warning(f"      ⚠️  {fp['id']}: {e}")
    
    # Alerts
    alerts = generate_demo_alerts()
    logger.info(f"   🚨 Inserimento {len(alerts)} alert...")
    for alert in alerts:
        try:
            db.query(f"CREATE alert:{alert['id']} CONTENT $data;", {"data": alert})
        except Exception as e:
            logger.warning(f"      ⚠️  {alert['id']}: {e}")
    
    logger.info("✅ Seeding completato")


def show_stats(db):
    """Mostra statistiche database."""
    logger.info("\n📊 STATISTICHE DATABASE")
    logger.info("=" * 50)
    
    tables = ["paper", "event", "pattern", "climate_index", 
              "observation", "causal_link", "fingerprint", "alert"]
    
    for table in tables:
        try:
            result = db.query(f"SELECT count() FROM {table} GROUP ALL;")
            count = result[0]["count"] if result and result[0] else 0
            logger.info(f"   {table:20s}: {count:>6}")
        except Exception:
            logger.info(f"   {table:20s}: {'N/A':>6}")
    
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Seed API Registry per CTW")
    parser.add_argument("--schema-only", action="store_true", help="Solo schema, no dati")
    parser.add_argument("--data-only", action="store_true", help="Solo dati, assume schema esistente")
    parser.add_argument("--reset", action="store_true", help="Reset tabelle CTW prima di creare")
    args = parser.parse_args()
    
    try:
        db = connect_db()
        
        if not args.data_only:
            apply_schema(db, reset=args.reset)
        
        if not args.schema_only:
            seed_data(db)
        
        show_stats(db)
        
        db.close()
        logger.info("\n✅ Completato con successo!")
        
    except Exception as e:
        logger.error(f"\n❌ Errore: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
