# 🗄️ SurrealDB Schema per CTW (Climate Tipping Warning)

## Panoramica

Schema multi-model ottimizzato per il sistema CTW con supporto per:
- **Document store**: Eventi storici, paper, alert
- **Graph database**: Relazioni causali PCMCI, teleconnessioni
- **Vector search**: Fingerprint MiniRocket per pattern matching
- **Geospatial**: Observation satellitari/aircraft con query spaziali
- **Time series**: Dati climatici con indexing temporale

## Configurazione

```
SurrealDB Server: localhost:8000
Namespace: causal
Database: knowledge
Auth: root/root (dev)
```

## Architettura Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE GRAPH CTW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     validates      ┌──────────────┐                       │
│  │    paper     │ ─────────────────▶ │   pattern    │                       │
│  │  (document)  │                    │   (causal)   │                       │
│  └──────────────┘                    └──────────────┘                       │
│         │                                   │                                │
│         │ documents                         │ observed_in                    │
│         ▼                                   ▼                                │
│  ┌──────────────┐                    ┌──────────────┐                       │
│  │    event     │ ◀───────────────── │ causal_link  │                       │
│  │ (historical) │     triggered_by   │   (PCMCI)    │                       │
│  └──────────────┘                    └──────────────┘                       │
│         ▲                                   │                                │
│         │ generated_from                    │ derived_from                   │
│         │                                   ▼                                │
│  ┌──────────────┐                    ┌──────────────┐                       │
│  │    alert     │                    │ observation  │                       │
│  │  (warning)   │                    │ (satellite)  │                       │
│  └──────────────┘                    └──────────────┘                       │
│         ▲                                   │                                │
│         │ matched_by                        │ has_fingerprint                │
│         │                                   ▼                                │
│  ┌──────────────┐                    ┌──────────────┐                       │
│  │ fingerprint  │ ◀───────────────── │climate_index │                       │
│  │ (MiniRocket) │   correlates_with  │   (NAO,etc)  │                       │
│  └──────────────┘                    └──────────────┘                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tabelle Esistenti (già implementate)

### `paper` - Letteratura scientifica
```sql
DEFINE TABLE paper SCHEMAFULL;
DEFINE FIELD id ON paper TYPE string;
DEFINE FIELD title ON paper TYPE string;
DEFINE FIELD authors ON paper TYPE array<string>;
DEFINE FIELD abstract ON paper TYPE string;
DEFINE FIELD doi ON paper TYPE option<string>;
DEFINE FIELD year ON paper TYPE int;
DEFINE FIELD journal ON paper TYPE option<string>;
DEFINE FIELD keywords ON paper TYPE array<string>;
DEFINE FIELD embedding ON paper TYPE option<array<float>>;
DEFINE FIELD created_at ON paper TYPE datetime DEFAULT time::now();

-- Indexes
DEFINE INDEX paper_id ON paper FIELDS id UNIQUE;
DEFINE INDEX paper_year ON paper FIELDS year;
DEFINE INDEX paper_keywords ON paper FIELDS keywords;
DEFINE INDEX paper_embedding_idx ON paper FIELDS embedding MTREE DIMENSION 1536 DIST COSINE TYPE F32;
```

### `event` - Eventi storici
```sql
DEFINE TABLE event SCHEMAFULL;
DEFINE FIELD id ON event TYPE string;
DEFINE FIELD name ON event TYPE string;
DEFINE FIELD description ON event TYPE string;
DEFINE FIELD event_type ON event TYPE string;  -- flood, storm, drought, etc.
DEFINE FIELD start_date ON event TYPE datetime;
DEFINE FIELD end_date ON event TYPE option<datetime>;
DEFINE FIELD location ON event TYPE option<object>;  -- GeoJSON
DEFINE FIELD severity ON event TYPE option<float>;   -- 0.0-1.0
DEFINE FIELD source ON event TYPE option<string>;
DEFINE FIELD created_at ON event TYPE datetime DEFAULT time::now();

-- Indexes
DEFINE INDEX event_id ON event FIELDS id UNIQUE;
DEFINE INDEX event_type_idx ON event FIELDS event_type;
DEFINE INDEX event_dates ON event FIELDS start_date, end_date;
```

### `pattern` - Pattern causali generici
```sql
DEFINE TABLE pattern SCHEMAFULL;
DEFINE FIELD id ON pattern TYPE string;
DEFINE FIELD name ON pattern TYPE string;
DEFINE FIELD description ON pattern TYPE string;
DEFINE FIELD pattern_type ON pattern TYPE string;  -- causal_chain, teleconnection, precursor, compound, feedback
DEFINE FIELD variables ON pattern TYPE array<string>;
DEFINE FIELD lag_days ON pattern TYPE option<int>;
DEFINE FIELD strength ON pattern TYPE option<float>;
DEFINE FIELD confidence ON pattern TYPE option<float>;
DEFINE FIELD metadata ON pattern TYPE option<object>;
DEFINE FIELD created_at ON pattern TYPE datetime DEFAULT time::now();

-- Indexes
DEFINE INDEX pattern_id ON pattern FIELDS id UNIQUE;
DEFINE INDEX pattern_type_idx ON pattern FIELDS pattern_type;
DEFINE INDEX pattern_vars ON pattern FIELDS variables;
```

### `climate_index` - Indici climatici (NAO, ENSO, etc.)
```sql
DEFINE TABLE climate_index SCHEMAFULL;
DEFINE FIELD id ON climate_index TYPE string;
DEFINE FIELD name ON climate_index TYPE string;
DEFINE FIELD abbreviation ON climate_index TYPE string;
DEFINE FIELD description ON climate_index TYPE string;
DEFINE FIELD source_url ON climate_index TYPE option<string>;
DEFINE FIELD time_series ON climate_index TYPE option<array<object>>;
DEFINE FIELD created_at ON climate_index TYPE datetime DEFAULT time::now();

-- Indexes
DEFINE INDEX climate_id ON climate_index FIELDS id UNIQUE;
DEFINE INDEX climate_abbrev ON climate_index FIELDS abbreviation;
```

---

## Nuove Tabelle CTW

### `observation` - Dati satellitari/aircraft con geolocation

Dati provenienti da ERA5, CMEMS, CYGNSS, AMDAR, etc.

```sql
DEFINE TABLE observation SCHEMAFULL;

-- Identificazione
DEFINE FIELD id ON observation TYPE string;
DEFINE FIELD source ON observation TYPE string;           -- era5, cmems, cygnss, amdar, slcci
DEFINE FIELD source_id ON observation TYPE option<string>; -- ID originale dal provider

-- Timestamp (ottimizzato per time series)
DEFINE FIELD timestamp ON observation TYPE datetime;
DEFINE FIELD time_resolution ON observation TYPE string;  -- hourly, daily, monthly

-- Geolocation (GeoJSON format)
DEFINE FIELD location ON observation TYPE object;         -- { type: "Point", coordinates: [lon, lat] }
DEFINE FIELD region ON observation TYPE option<string>;   -- mediterranean, alpine, adriatic, etc.
DEFINE FIELD altitude ON observation TYPE option<float>;  -- meters (per AMDAR aircraft)

-- Variabili misurate (sparse - solo quelle presenti)
DEFINE FIELD variables ON observation TYPE object;
-- Esempi di campi in variables:
-- {
--   "sea_level": 0.45,           -- meters
--   "sst": 18.5,                 -- °C
--   "precipitation": 12.3,       -- mm
--   "wind_u10": 5.2,            -- m/s
--   "wind_v10": -3.1,           -- m/s
--   "mslp": 1013.25,            -- hPa
--   "ivt": 450.0,               -- kg/m/s (Integrated Vapor Transport)
--   "geopotential_500": 5520,   -- m
--   "temperature": 285.5,       -- K
--   "humidity": 0.85            -- 0-1
-- }

-- Quality flags
DEFINE FIELD quality_flag ON observation TYPE option<int>;  -- 0=good, 1=suspect, 2=bad
DEFINE FIELD processing_level ON observation TYPE option<string>;  -- L1, L2, L3, L4

-- Metadata
DEFINE FIELD metadata ON observation TYPE option<object>;
DEFINE FIELD created_at ON observation TYPE datetime DEFAULT time::now();

-- Indexes per query performanti
DEFINE INDEX obs_id ON observation FIELDS id UNIQUE;
DEFINE INDEX obs_source ON observation FIELDS source;
DEFINE INDEX obs_timestamp ON observation FIELDS timestamp;
DEFINE INDEX obs_region ON observation FIELDS region;
DEFINE INDEX obs_source_time ON observation FIELDS source, timestamp;

-- Composite index per query spazio-temporali
DEFINE INDEX obs_spatiotemporal ON observation FIELDS region, timestamp;
```

**Query esempio:**
```sql
-- Tutte le osservazioni mediterranee dell'ultimo mese
SELECT * FROM observation 
WHERE region = "mediterranean" 
  AND timestamp > time::now() - 30d
ORDER BY timestamp DESC;

-- Osservazioni vicino a un punto (entro 100km)
SELECT *, geo::distance(location, $point) AS dist FROM observation
WHERE geo::distance(location, { type: "Point", coordinates: [8.6, 45.9] }) < 100000
ORDER BY dist;
```

---

### `causal_link` - Relazioni PCMCI specifiche

Archi causali scoperti da PCMCI con tutti i parametri statistici.

```sql
DEFINE TABLE causal_link SCHEMAFULL;

-- Identificazione
DEFINE FIELD id ON causal_link TYPE string;
DEFINE FIELD name ON causal_link TYPE option<string>;

-- Nodi causali (driver → target)
DEFINE FIELD driver ON causal_link TYPE string;           -- Nome variabile driver
DEFINE FIELD target ON causal_link TYPE string;           -- Nome variabile target
DEFINE FIELD driver_region ON causal_link TYPE option<string>;
DEFINE FIELD target_region ON causal_link TYPE option<string>;

-- Parametri PCMCI
DEFINE FIELD lag ON causal_link TYPE int;                 -- Lag temporale in unità dati
DEFINE FIELD lag_unit ON causal_link TYPE string;         -- days, hours, months
DEFINE FIELD strength ON causal_link TYPE float;          -- Coefficiente correlazione parziale
DEFINE FIELD p_value ON causal_link TYPE float;           -- Significatività statistica
DEFINE FIELD confidence_interval ON causal_link TYPE option<array<float>>;  -- [lower, upper]

-- Tipo di link
DEFINE FIELD link_type ON causal_link TYPE string;        -- direct, indirect, contemporaneous
DEFINE FIELD mechanism ON causal_link TYPE option<string>; -- Descrizione fisica del meccanismo

-- Validazione
DEFINE FIELD is_validated ON causal_link TYPE bool DEFAULT false;
DEFINE FIELD validation_source ON causal_link TYPE option<string>;  -- paper_id, expert, reanalysis

-- Periodo di validità
DEFINE FIELD valid_from ON causal_link TYPE option<datetime>;
DEFINE FIELD valid_to ON causal_link TYPE option<datetime>;
DEFINE FIELD seasonality ON causal_link TYPE option<array<string>>;  -- ["DJF", "MAM", "JJA", "SON"]

-- Metadata
DEFINE FIELD pcmci_params ON causal_link TYPE option<object>;
-- {
--   "tau_max": 10,
--   "pc_alpha": 0.05,
--   "cond_ind_test": "ParCorr",
--   "n_samples": 10000
-- }
DEFINE FIELD created_at ON causal_link TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON causal_link TYPE option<datetime>;

-- Indexes
DEFINE INDEX causal_id ON causal_link FIELDS id UNIQUE;
DEFINE INDEX causal_driver ON causal_link FIELDS driver;
DEFINE INDEX causal_target ON causal_link FIELDS target;
DEFINE INDEX causal_strength ON causal_link FIELDS strength;
DEFINE INDEX causal_pvalue ON causal_link FIELDS p_value;
DEFINE INDEX causal_lag ON causal_link FIELDS lag;
DEFINE INDEX causal_validated ON causal_link FIELDS is_validated;
```

**Query esempio:**
```sql
-- Tutti i link significativi (p < 0.05) che influenzano sea_level
SELECT * FROM causal_link 
WHERE target = "sea_level" 
  AND p_value < 0.05
ORDER BY strength DESC;

-- Catena causale: trova tutti i driver indiretti
SELECT ->triggers->causal_link.* AS downstream FROM causal_link
WHERE driver = "NAO" AND is_validated = true;
```

---

### `fingerprint` - Embedding MiniRocket per pattern matching

Signature numeriche per riconoscimento rapido di pattern climatici.

```sql
DEFINE TABLE fingerprint SCHEMAFULL;

-- Identificazione
DEFINE FIELD id ON fingerprint TYPE string;
DEFINE FIELD name ON fingerprint TYPE string;
DEFINE FIELD description ON fingerprint TYPE option<string>;

-- Embedding MiniRocket (ottimizzato per 100 dimensioni)
DEFINE FIELD embedding ON fingerprint TYPE array<float>;  -- array[100] float
DEFINE FIELD embedding_version ON fingerprint TYPE string; -- v1.0, v2.0

-- Riferimento al pattern/evento associato
DEFINE FIELD source_type ON fingerprint TYPE string;      -- event, pattern, observation_window
DEFINE FIELD source_id ON fingerprint TYPE string;        -- ID dell'entità sorgente

-- Parametri estrazione
DEFINE FIELD window_start ON fingerprint TYPE datetime;
DEFINE FIELD window_end ON fingerprint TYPE datetime;
DEFINE FIELD variables_used ON fingerprint TYPE array<string>;  -- ["sst", "mslp", "ivt"]
DEFINE FIELD region ON fingerprint TYPE option<string>;

-- Statistiche fingerprint
DEFINE FIELD stats ON fingerprint TYPE option<object>;
-- {
--   "mean": 0.45,
--   "std": 0.12,
--   "min": -0.8,
--   "max": 0.92,
--   "dominant_frequency": 0.033  -- ~30 giorni
-- }

-- Metadata
DEFINE FIELD created_at ON fingerprint TYPE datetime DEFAULT time::now();
DEFINE FIELD model_config ON fingerprint TYPE option<object>;

-- Indexes
DEFINE INDEX fp_id ON fingerprint FIELDS id UNIQUE;
DEFINE INDEX fp_source ON fingerprint FIELDS source_type, source_id;
DEFINE INDEX fp_region ON fingerprint FIELDS region;

-- Vector index per similarity search (HNSW per performance, 100 dim)
DEFINE INDEX fp_embedding_hnsw ON fingerprint 
    FIELDS embedding 
    HNSW DIMENSION 100 
    DIST COSINE
    EFC 150        -- efConstruction: qualità build
    M 12;          -- max connections per node

-- Alternativa MTREE per exact search (più lento ma 100% recall)
-- DEFINE INDEX fp_embedding_mtree ON fingerprint FIELDS embedding MTREE DIMENSION 100 DIST COSINE TYPE F32;
```

**Query esempio:**
```sql
-- Trova i 5 fingerprint più simili a un query embedding
LET $query := [0.1, 0.2, ...]; -- 100 valori
SELECT id, name, source_type, source_id, 
       vector::distance::knn() AS similarity
FROM fingerprint
WHERE embedding <|5,64|> $query
ORDER BY similarity;

-- Fingerprint di eventi flood nella regione alpina
SELECT * FROM fingerprint
WHERE source_type = "event" 
  AND region = "alpine"
  AND source_id IN (SELECT id FROM event WHERE event_type = "flood");
```

---

### `alert` - Warning generati dal sistema

Alert prodotti dalla pipeline di early warning.

```sql
DEFINE TABLE alert SCHEMAFULL;

-- Identificazione
DEFINE FIELD id ON alert TYPE string;
DEFINE FIELD alert_code ON alert TYPE string;             -- CTW-2024-001-FLOOD-ALPINE

-- Classificazione
DEFINE FIELD alert_type ON alert TYPE string;             -- flood, storm_surge, drought, compound
DEFINE FIELD severity ON alert TYPE string;               -- watch, advisory, warning, emergency
DEFINE FIELD confidence ON alert TYPE float;              -- 0.0-1.0

-- Timing
DEFINE FIELD issued_at ON alert TYPE datetime;
DEFINE FIELD valid_from ON alert TYPE datetime;
DEFINE FIELD valid_until ON alert TYPE datetime;
DEFINE FIELD lead_time_hours ON alert TYPE int;           -- Anticipo previsione

-- Area interessata
DEFINE FIELD affected_region ON alert TYPE string;
DEFINE FIELD affected_area ON alert TYPE option<object>;  -- GeoJSON Polygon
DEFINE FIELD affected_population ON alert TYPE option<int>;

-- Descrizione
DEFINE FIELD headline ON alert TYPE string;
DEFINE FIELD description ON alert TYPE string;
DEFINE FIELD recommended_actions ON alert TYPE option<array<string>>;

-- Trigger: cosa ha generato l'alert
DEFINE FIELD trigger_type ON alert TYPE string;           -- fingerprint_match, threshold, pcmci_signal
DEFINE FIELD trigger_details ON alert TYPE object;
-- {
--   "matched_fingerprint_id": "fp_lago2000",
--   "similarity_score": 0.87,
--   "threshold_exceeded": ["sea_level", "precipitation"],
--   "causal_chain": ["NAO-", "IVT+", "precipitation+"]
-- }

-- Validazione post-evento
DEFINE FIELD status ON alert TYPE string DEFAULT "active"; -- active, expired, verified, false_alarm
DEFINE FIELD verification ON alert TYPE option<object>;
-- {
--   "verified_at": "2024-01-15T00:00:00Z",
--   "actual_event_id": "event_xyz",
--   "hit_rate": 0.85,
--   "false_alarm_rate": 0.12
-- }

-- Metadata
DEFINE FIELD created_at ON alert TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON alert TYPE option<datetime>;
DEFINE FIELD created_by ON alert TYPE option<string>;     -- system, analyst_name

-- Indexes
DEFINE INDEX alert_id ON alert FIELDS id UNIQUE;
DEFINE INDEX alert_code ON alert FIELDS alert_code UNIQUE;
DEFINE INDEX alert_type ON alert FIELDS alert_type;
DEFINE INDEX alert_severity ON alert FIELDS severity;
DEFINE INDEX alert_status ON alert FIELDS status;
DEFINE INDEX alert_issued ON alert FIELDS issued_at;
DEFINE INDEX alert_region ON alert FIELDS affected_region;
DEFINE INDEX alert_valid ON alert FIELDS valid_from, valid_until;
```

**Query esempio:**
```sql
-- Alert attivi per il Mediterraneo
SELECT * FROM alert
WHERE status = "active"
  AND affected_region = "mediterranean"
  AND valid_until > time::now()
ORDER BY severity DESC, confidence DESC;

-- Performance storica del sistema
SELECT 
    alert_type,
    count() AS total,
    math::mean(confidence) AS avg_confidence,
    count(IF status = "verified" THEN 1 END) AS verified,
    count(IF status = "false_alarm" THEN 1 END) AS false_alarms
FROM alert
WHERE issued_at > time::now() - 1y
GROUP BY alert_type;
```

---

## Edge Tables (Relazioni Graph)

### Relazioni esistenti

```sql
-- Paper valida un pattern
DEFINE TABLE validates SCHEMAFULL TYPE RELATION FROM paper TO pattern;
DEFINE FIELD validation_type ON validates TYPE string;
DEFINE FIELD confidence ON validates TYPE float;
DEFINE FIELD notes ON validates TYPE option<string>;
DEFINE FIELD validated_at ON validates TYPE datetime DEFAULT time::now();

-- Pattern osservato in un evento
DEFINE TABLE observed_in SCHEMAFULL TYPE RELATION FROM pattern TO event;
DEFINE FIELD correlation ON observed_in TYPE option<float>;
DEFINE FIELD lag_observed ON observed_in TYPE option<int>;
DEFINE FIELD notes ON observed_in TYPE option<string>;

-- Pattern causa altro pattern
DEFINE TABLE causes SCHEMAFULL TYPE RELATION FROM pattern TO pattern;
DEFINE FIELD mechanism ON causes TYPE option<string>;
DEFINE FIELD strength ON causes TYPE float;
DEFINE FIELD lag_days ON causes TYPE option<int>;
DEFINE FIELD evidence ON causes TYPE option<array<string>>;

-- Climate index correla con pattern
DEFINE TABLE correlates_with SCHEMAFULL TYPE RELATION FROM climate_index TO pattern;
DEFINE FIELD correlation ON correlates_with TYPE float;
DEFINE FIELD lag_months ON correlates_with TYPE option<int>;
DEFINE FIELD period ON correlates_with TYPE option<string>;
```

### Nuove relazioni CTW

```sql
-- Causal link triggera un evento
DEFINE TABLE triggers SCHEMAFULL TYPE RELATION FROM causal_link TO event;
DEFINE FIELD contribution ON triggers TYPE float;         -- Peso del contributo
DEFINE FIELD lag_observed ON triggers TYPE option<int>;
DEFINE FIELD notes ON triggers TYPE option<string>;

-- Observation genera fingerprint
DEFINE TABLE has_fingerprint SCHEMAFULL TYPE RELATION FROM observation TO fingerprint;
DEFINE FIELD extraction_params ON has_fingerprint TYPE option<object>;

-- Fingerprint matcha e genera alert
DEFINE TABLE matched_by SCHEMAFULL TYPE RELATION FROM alert TO fingerprint;
DEFINE FIELD similarity_score ON matched_by TYPE float;
DEFINE FIELD matched_at ON matched_by TYPE datetime DEFAULT time::now();

-- Alert riferito a evento (verifica post-hoc)
DEFINE TABLE refers_to SCHEMAFULL TYPE RELATION FROM alert TO event;
DEFINE FIELD was_correct ON refers_to TYPE bool;
DEFINE FIELD temporal_error_hours ON refers_to TYPE option<int>;
DEFINE FIELD spatial_error_km ON refers_to TYPE option<float>;

-- Causal link deriva da observation
DEFINE TABLE derived_from SCHEMAFULL TYPE RELATION FROM causal_link TO observation;
DEFINE FIELD analysis_id ON derived_from TYPE option<string>;
DEFINE FIELD contribution_weight ON derived_from TYPE option<float>;
```

---

## Query Patterns Comuni

### 1. Ricerca fingerprint simili (Early Warning)

```sql
-- Input: embedding corrente, trova eventi storici simili
LET $current := [...]; -- embedding 100-dim
SELECT 
    fp.id,
    fp.name,
    fp.source_id,
    e.name AS event_name,
    e.severity,
    e.start_date,
    vector::distance::knn() AS similarity
FROM fingerprint AS fp
JOIN event AS e ON fp.source_id = e.id
WHERE fp.embedding <|10,100|> $current
  AND fp.source_type = "event"
ORDER BY similarity;
```

### 2. Catena causale completa

```sql
-- Trova tutti gli effetti downstream di NAO negativo
SELECT 
    cl.driver,
    cl.target,
    cl.lag,
    cl.strength,
    ->triggers->event.name AS triggered_events
FROM causal_link AS cl
WHERE cl.driver CONTAINS "NAO"
  AND cl.strength < -0.5
  AND cl.p_value < 0.05;
```

### 3. Validazione storica alert

```sql
-- Performance degli alert per tipo
SELECT 
    alert_type,
    severity,
    count() AS total_alerts,
    count(IF status = "verified" THEN 1) AS hits,
    count(IF status = "false_alarm" THEN 1) AS false_alarms,
    math::mean(<-refers_to<-event.severity) AS avg_event_severity
FROM alert
WHERE issued_at > "2020-01-01"
GROUP BY alert_type, severity
ORDER BY alert_type, severity;
```

### 4. Observation aggregation spazio-temporale

```sql
-- Media giornaliera SST per regione
SELECT 
    time::group(timestamp, "day") AS day,
    region,
    math::mean(variables.sst) AS mean_sst,
    math::max(variables.sst) AS max_sst,
    count() AS n_obs
FROM observation
WHERE source = "cmems"
  AND timestamp > time::now() - 30d
GROUP BY day, region
ORDER BY day DESC;
```

---

## Migrazione e Seed

Per popolare il database con lo schema, eseguire:

```bash
# 1. Avvia SurrealDB (se non già attivo)
surreal start --bind 127.0.0.1:8000 --user root --pass root file:~/.surrealdb/data/ctw.db

# 2. Esegui lo script di seed
python scripts/seed_api_registry.py
```

---

## Note Implementative

### Performance Tips

1. **Complex Record IDs** per time series:
   ```sql
   CREATE observation:[era5, "2024-01-15T12:00:00Z", "mediterranean"] 
   CONTENT {...};
   ```

2. **Batch inserts** per observation:
   ```sql
   INSERT INTO observation [...array of objects...];
   ```

3. **LIVE SELECT** per real-time monitoring:
   ```sql
   LIVE SELECT * FROM alert WHERE status = "active";
   ```

### Dimensioni stimate

| Tabella | Records stimati | Storage |
|---------|-----------------|---------|
| observation | ~10M/anno | ~50GB |
| fingerprint | ~100K | ~500MB |
| causal_link | ~10K | ~50MB |
| alert | ~5K/anno | ~25MB |
| paper | ~10K | ~100MB |
| event | ~1K | ~10MB |

### Backup

```bash
# Export completo
surreal export --conn http://localhost:8000 --user root --pass root \
  --ns causal --db knowledge > backup_$(date +%Y%m%d).surql

# Import
surreal import --conn http://localhost:8000 --user root --pass root \
  --ns causal --db knowledge < backup_20240115.surql
```
