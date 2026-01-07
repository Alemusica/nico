# 🏗️ CTW Architecture Gap Analysis

> **Data**: 2026-01-07  
> **Branch**: `feature/api-registry-alessio`  
> **Analisi**: Cosa c'è, cosa manca, priorità

---

## 📊 Status Attuale

### ✅ IMPLEMENTATO (Funzionante)

| Componente | File | Linee | Descrizione |
|------------|------|-------|-------------|
| **API Registry** | `data/api_registry.py` | 576 | Catalogo 18 data sources |
| **CMEMS Client** | `data/cmems_client.py` | 438 | Sea level, SST, waves |
| **ERA5 Client** | `data/era5_client.py` | 552 | Reanalysis atmosferica |
| **GPM Client** | `data/gpm_client.py` | 436 | Precipitazione near-RT |
| **Aircraft Client** | `data/aircraft_client.py` | 648 | AMDAR, Mode-S |
| **Tide Gauge Client** | `data/tide_gauge_client.py` | 523 | IOC sea level |
| **Sentinel Client** | `data/sentinel_client.py` | 546 | SAR, Ocean color |
| **GRACE Client** | `data/grace_client.py` | 390 | Water storage |
| **Argo Client** | `data/argo_client.py` | 452 | Ocean T/S profiles |
| **Climate Indices** | `data/climate_indices.py` | 506 | NAO, ENSO, AMO |
| **Semantic Scholar** | `data/knowledge/semantic_scholar.py` | 473 | Paper search |
| **Health Checker** | `data/health_checker.py` | 436 | API monitoring |
| **SWE Physics** | `physics/swe.py` | 192 | Shallow Water Equations |
| **Gray Zone** | `buffer/gray_zone.py` | 290 | Pattern validation queue |
| **Gates** | `pipeline/gates.py` | 294 | Satellite pass gates |
| **Config** | `core/config.py` | 149 | System configuration |

**Totale implementato**: ~6500 linee

### ❌ VUOTO (File esistente, nessun codice)

| Componente | File | Priorità | Blocca |
|------------|------|----------|--------|
| **PCMCI Runner** | `causal/pcmci_runner.py` | 🔴 CRITICO | Pattern discovery |
| **Graph Builder** | `causal/graph_builder.py` | 🔴 CRITICO | Causal storage |
| **Teleconnections** | `causal/teleconnections.py` | 🟡 ALTO | Known patterns |
| **GNN Model** | `physics/gnn_model.py` | 🔴 CRITICO | Physics ML |
| **Loss Functions** | `physics/loss_functions.py` | 🟡 ALTO | PINN training |
| **Boundary** | `physics/boundary.py` | 🟢 MEDIO | Edge handling |
| **Spectrogram** | `fingerprinting/spectrogram.py` | 🔴 CRITICO | Event fingerprint |
| **Hasher** | `fingerprinting/hasher.py` | 🟡 ALTO | Pattern hashing |
| **Matcher** | `fingerprinting/matcher.py` | 🟡 ALTO | Pattern matching |
| **Peaks** | `fingerprinting/peaks.py` | 🟡 ALTO | Peak detection |
| **Database** | `fingerprinting/database.py` | 🟢 MEDIO | Fingerprint storage |
| **Pipeline Stages** | `pipeline/stages.py` | 🔴 CRITICO | Main pipeline |
| **Ensemble** | `pipeline/ensemble.py` | 🟡 ALTO | Multi-model |
| **Dashboard** | `visualization/dashboard.py` | 🟢 MEDIO | Monitoring UI |
| **Maps** | `visualization/maps.py` | 🟢 MEDIO | Geo visualization |
| **Replay Buffer** | `buffer/replay_buffer.py` | 🟡 ALTO | Experience storage |
| **Validator** | `buffer/validator.py` | 🟡 ALTO | Pattern validation |
| **Preprocessors** | `data/preprocessors/*.py` | 🟡 ALTO | Data pipeline |

---

## 🎯 Priorità di Implementazione

### Fase 1: Core Pipeline (Blocca tutto il resto)

```
┌─────────────────────────────────────────────────────────────────┐
│  1. PCMCI Runner          → Scopre relazioni causali            │
│  2. Pipeline Stages       → Orchestrazione end-to-end           │
│  3. Preprocessors         → Normalizzazione, interpolazione     │
└─────────────────────────────────────────────────────────────────┘
```

**Stima**: 2-3 giorni

### Fase 2: Physics-Informed ML

```
┌─────────────────────────────────────────────────────────────────┐
│  4. GNN Model             → Graph Neural Network per SWE        │
│  5. Loss Functions        → Physics loss + data loss            │
│  6. Boundary              → Condizioni al contorno              │
└─────────────────────────────────────────────────────────────────┘
```

**Stima**: 3-5 giorni  
**Dipende da**: Fase 1  
**Riferimento**: `Remembrance/binaural_golden/src/core/jax_plate_fem.py`

### Fase 3: Fingerprinting (Shazam-like)

```
┌─────────────────────────────────────────────────────────────────┐
│  7. Spectrogram           → STFT delle time series              │
│  8. Peaks                 → Estrazione picchi caratteristici    │
│  9. Hasher                → Locality-Sensitive Hashing          │
│  10. Matcher              → Ricerca pattern simili              │
└─────────────────────────────────────────────────────────────────┘
```

**Stima**: 3-4 giorni  
**Dipende da**: Fase 1

### Fase 4: Knowledge Integration

```
┌─────────────────────────────────────────────────────────────────┐
│  11. Graph Builder        → Costruzione grafo causale           │
│  12. Teleconnections      → Pattern noti (NAO→Europe)           │
│  13. SurrealDB MCP        → Query da Copilot/Claude             │
└─────────────────────────────────────────────────────────────────┘
```

**Stima**: 2-3 giorni  
**Riferimento**: `Remembrance/binaural_golden/src/utils/surrealdb_mcp_server.py`

### Fase 5: Validation & Experience

```
┌─────────────────────────────────────────────────────────────────┐
│  14. Replay Buffer        → Storage esperienze passate          │
│  15. Validator            → Validazione physics + experience    │
│  16. Ensemble             → Multi-model prediction              │
└─────────────────────────────────────────────────────────────────┘
```

**Stima**: 2-3 giorni  
**Riferimento**: `Remembrance/binaural_golden/src/core/rdnn_memory.py`

### Fase 6: Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│  17. Dashboard            → Streamlit monitoring                │
│  18. Maps                 → Plotly/Folium geo viz               │
└─────────────────────────────────────────────────────────────────┘
```

**Stima**: 1-2 giorni

---

## 🔗 Dipendenze tra Componenti

```
                    ┌─────────────┐
                    │ API Clients │ ✅ DONE
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │Preprocessors│ ❌ TODO
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌─────▼─────┐     ┌─────▼─────┐
    │  PCMCI  │      │Fingerprint│     │  Physics  │
    │ Runner  │      │  Engine   │     │   (SWE)   │
    │  ❌     │      │    ❌     │     │    ✅     │
    └────┬────┘      └─────┬─────┘     └─────┬─────┘
         │                 │                 │
         │           ┌─────▼─────┐           │
         │           │  Matcher  │           │
         │           │    ❌     │           │
         │           └─────┬─────┘           │
         │                 │                 │
         └────────┬────────┴────────┬────────┘
                  │                 │
           ┌──────▼──────┐   ┌──────▼──────┐
           │Graph Builder│   │ GNN Model   │
           │     ❌      │   │     ❌      │
           └──────┬──────┘   └──────┬──────┘
                  │                 │
                  └────────┬────────┘
                           │
                    ┌──────▼──────┐
                    │  Ensemble   │
                    │     ❌      │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │Early Warning│
                    │   (exists)  │
                    └─────────────┘
```

---

## 📋 SurrealDB Schema per CTW

Basato su best practices 2025/2026:

```sql
-- Namespace: surge_shazam
-- Database: early_warning

-- 1. Data Sources (già in data_catalog)
DEFINE TABLE data_source SCHEMAFULL;

-- 2. Observations (time series with composite ID)
DEFINE TABLE observation SCHEMAFULL;
DEFINE FIELD source_id ON observation TYPE string;
DEFINE FIELD variable ON observation TYPE string;
DEFINE FIELD timestamp ON observation TYPE datetime;
DEFINE FIELD location ON observation TYPE geometry(point);
DEFINE FIELD value ON observation TYPE float;
DEFINE FIELD quality ON observation TYPE int;
-- Composite ID: observation:⟨source_variable_timestamp⟩
DEFINE INDEX idx_obs_time ON observation FIELDS timestamp;
DEFINE INDEX idx_obs_source ON observation FIELDS source_id;

-- 3. Causal Links (graph edges)
DEFINE TABLE causal_link SCHEMAFULL;
DEFINE FIELD cause ON causal_link TYPE string;
DEFINE FIELD effect ON causal_link TYPE string;
DEFINE FIELD lag_hours ON causal_link TYPE int;
DEFINE FIELD strength ON causal_link TYPE float;
DEFINE FIELD physics_score ON causal_link TYPE float;
DEFINE FIELD experience_score ON causal_link TYPE float;
DEFINE FIELD discovered_at ON causal_link TYPE datetime;
DEFINE FIELD status ON causal_link TYPE string; -- validated, gray_zone, rejected
-- Graph edge: RELATE cause:X -> causal_link -> effect:Y

-- 4. Events (historical floods, storms)
DEFINE TABLE event SCHEMAFULL;
DEFINE FIELD name ON event TYPE string;
DEFINE FIELD event_type ON event TYPE string;
DEFINE FIELD start_date ON event TYPE datetime;
DEFINE FIELD end_date ON event TYPE datetime;
DEFINE FIELD location ON event TYPE geometry(polygon);
DEFINE FIELD severity ON event TYPE float;
DEFINE FIELD sources ON event TYPE array; -- papers, news, witnesses
DEFINE FIELD precursors ON event TYPE array; -- detected precursor patterns

-- 5. Fingerprints (Shazam-like patterns)
DEFINE TABLE fingerprint SCHEMAFULL;
DEFINE FIELD event_id ON fingerprint TYPE string;
DEFINE FIELD hash ON fingerprint TYPE string;
DEFINE FIELD peaks ON fingerprint TYPE array;
DEFINE FIELD duration_hours ON fingerprint TYPE int;
DEFINE FIELD variables ON fingerprint TYPE array;
DEFINE INDEX idx_fp_hash ON fingerprint FIELDS hash;

-- 6. Alerts (real-time warnings)
DEFINE TABLE alert SCHEMAFULL;
DEFINE FIELD level ON alert TYPE string; -- green, yellow, orange, red
DEFINE FIELD region ON alert TYPE geometry(polygon);
DEFINE FIELD timestamp ON alert TYPE datetime;
DEFINE FIELD confidence ON alert TYPE float;
DEFINE FIELD precursors ON alert TYPE array;
DEFINE FIELD matched_events ON alert TYPE array; -- historical matches
DEFINE FIELD expires_at ON alert TYPE datetime;

-- 7. Papers (from Semantic Scholar)
DEFINE TABLE paper SCHEMAFULL;
DEFINE FIELD paper_id ON paper TYPE string;
DEFINE FIELD title ON paper TYPE string;
DEFINE FIELD authors ON paper TYPE array;
DEFINE FIELD year ON paper TYPE int;
DEFINE FIELD abstract ON paper TYPE string;
DEFINE FIELD domains ON paper TYPE array;
DEFINE FIELD relevance ON paper TYPE string;
```

---

## 🚀 Prossimi Passi Concreti

### Immediato (questa sessione)
1. ~~Completare API clients~~ ✅
2. ~~Aggiornare registry status~~ ✅
3. ~~Commit changes~~ ✅

### Prossima Sessione
1. **Implementare PCMCI Runner** (usa Tigramite)
2. **Implementare Preprocessors** (interpolazione, normalizzazione)
3. **Creare MCP Server per CTW** (basato su Remembrance)

### Settimana 1
- Pipeline stages completa
- Fingerprinting base (spectrogram + peaks)
- Test end-to-end su Lago Maggiore 2000

### Settimana 2
- GNN Model (transfer da JAX plate FEM)
- Graph Builder + SurrealDB
- Integration tests

### Settimana 3
- Dashboard Streamlit
- Ensemble predictions
- Documentation

---

## 📚 Riferimenti Utili

| Componente | Riferimento in Remembrance |
|------------|---------------------------|
| Physics ML | `jax_plate_fem.py` - Pattern JAX + AD |
| Experience Memory | `rdnn_memory.py` - LSTM con stato |
| SurrealDB MCP | `surrealdb_mcp_server.py` - Tool exposure |
| Schema Design | `SURREALDB_MCP_GUIDE.md` |

---

*Ultimo aggiornamento: 2026-01-07*
