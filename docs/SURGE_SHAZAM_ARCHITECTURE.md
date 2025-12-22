# 🌊 Surge-Shazam-DK Architecture

> **Storm Surge Prediction System for Denmark**
> Hybrid approach: GNN + PINN + Fingerprinting + Causal Discovery

## Executive Summary

Sistema ibrido per previsione storm surge in Danimarca che combina:
- **Fingerprinting Shazam-like** su segnali meteo (picchi/anchor points)
- **Graph Neural Networks** con neuroni "eccitabili" solo su pattern rilevanti
- **Physics-Informed Neural Networks** con Shallow Water Equations
- **Causal Discovery** per teleconnessioni globali non ovvie
- **Pipeline probabilistica a stadi** con gate di confidenza
- **Gray Zone Patterns** per correlazioni storiche non ancora validate fisicamente

---

## 1. Filosofia Demistificata

### Le Neural Networks sono solo funzioni

Niente magia: una NN è un insieme di funzioni matematiche semplici concatenate:
- `y = peso × x + bias` (moltiplicazione + somma)
- Seguito da una curva (es. `tanh`) per non rendere tutto lineare

Non sono cervelli, non pensano: **approssimano relazioni tra numeri** addestrandosi su esempi.

### Il principio "Shazam"

Come Shazam riconosce canzoni in bar rumorosi:
1. Trasforma audio → spettrogramma (grafico frequenza/tempo)
2. Trova picchi intensi (punti caratteristici)
3. Crea "impronte" (hash) da coppie di picchi
4. Confronta con database di impronte note

**Noi facciamo lo stesso con dati meteo**: vento, pressione → "spettrogramma" spazio-temporale → picchi → impronte → match con eventi storici che hanno causato surge.

### Perché ibrido vince

| Approccio | Pro | Contro |
|-----------|-----|--------|
| **Solo ML/Data** | Veloce, trova pattern | Overfitta, inventa correlazioni spurie |
| **Solo Fisica** | Affidabile, causale | Lento, richiede mesh pesanti |
| **Ibrido** | Veloce + affidabile | Complessità implementativa |

---

## 2. Architettura Complessiva

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER (Dati Eterogenei)                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │  ERA5 Global │ │ DMI Tide     │ │ Indici NAO   │ │ CMEMS Ocean  │   │
│  │  (Wind, P)   │ │ Gauges       │ │ AO, ENSO     │ │ (SSH, SST)   │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
│         │                 │                │                │           │
│         └─────────────────┴────────────────┴────────────────┘           │
│                                    │                                     │
│                    Tensore X ∈ ℝ^{T × C × H × W}                        │
│                    [72 ore × 10 channels × lat × lon]                   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: FINGERPRINTING + CAUSAL DISCOVERY            │
│                                                                          │
│  ┌─────────────────────────┐    ┌─────────────────────────────────┐    │
│  │  FINGERPRINT EXTRACTOR  │    │     CAUSAL GRAPH (PCMCI)        │    │
│  │  ─────────────────────  │    │     ───────────────────────     │    │
│  │  Signal → STFT          │    │  Tigramite su dati storici      │    │
│  │  → Peak Detection       │    │  → Grafo causale con lag        │    │
│  │  → LSH Hashing          │    │  → Scopre teleconnessioni       │    │
│  │  → Pattern Match        │    │    (es. ENSO → NAO → Surge)     │    │
│  └─────────────────────────┘    └─────────────────────────────────┘    │
│              │                                │                         │
│              └────────────────┬───────────────┘                         │
│                               │                                         │
│                    Confidence Score + Causal Edges                      │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ Gate: confidence > 0.6?
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: GRAPH NEURAL NETWORK (GNN)                   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PHYSICS-INFORMED GNN                          │   │
│  │  ───────────────────────────────────────────────────────────    │   │
│  │  Nodi: Grid points globali + focus Denmark (multi-risoluzione)  │   │
│  │  Archi: Spaziali (vicinato) + Causali (da PCMCI con lag)        │   │
│  │                                                                  │   │
│  │  ┌───────────┐    ┌───────────────┐    ┌───────────────┐        │   │
│  │  │  Encoder  │ →  │  Processor    │ →  │   Decoder     │        │   │
│  │  │  (CNN)    │    │  (16 MPNN     │    │  (Surge h)    │        │   │
│  │  │           │    │   layers)     │    │               │        │   │
│  │  └───────────┘    └───────────────┘    └───────────────┘        │   │
│  │                          │                                       │   │
│  │              Physics Loss (SWE Residual)                         │   │
│  │              ∂h/∂t + ∇·(hu) ≈ 0                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ Gate: ensemble confidence > 0.8?
┌─────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: ENSEMBLE + VALIDATION                        │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │  Physics Check   │  │  Historical      │  │  Uncertainty     │      │
│  │  (SWE residual)  │  │  Validation      │  │  Quantification  │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│           │                    │                      │                 │
│           └────────────────────┴──────────────────────┘                 │
│                               │                                         │
│                    ┌──────────┴──────────┐                             │
│                    │                     │                              │
│              Physics OK?            Physics Weak?                       │
│                    │                     │                              │
│                    ▼                     ▼                              │
│             ┌──────────┐         ┌──────────────┐                      │
│             │ VALIDATED│         │  GRAY ZONE   │                      │
│             │ PATTERN  │         │   PATTERN    │                      │
│             └──────────┘         └──────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    COCKPIT DECISIONALE (Dashboard)                       │
│                                                                          │
│  ┌───────────────────────────┐  ┌───────────────────────────────────┐  │
│  │    📊 ESPERIENZA          │  │    🔬 SCIENZA                      │  │
│  │    (Historical Match)     │  │    (Physics Validation)           │  │
│  │  ─────────────────────    │  │  ───────────────────────────      │  │
│  │  Pattern match: 82%       │  │  Physics residual: 0.03 ✅        │  │
│  │  Eventi simili: 47/60     │  │  SWE constraint: OK               │  │
│  │  Prob storica: 70-85%     │  │  Inverse barometer: OK            │  │
│  │                           │  │  Prob fisica: 65-80%              │  │
│  │  [===========>    ] 82%   │  │  [========>       ] 72%           │  │
│  └───────────────────────────┘  └───────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ⚠️  OUTPUT COMBINATO                                           │   │
│  │  Surge previsto: +1.8m (±0.3m) a Esbjerg in 24-36h              │   │
│  │  Confidenza: 76% (storico forte, fisica ok)                      │   │
│  │  Raccomandazione: ALLERTA ARANCIONE                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Moduli Dettagliati

### 3.1 Data Layer

```
src/data/
├── loaders/
│   ├── era5_client.py       # ECMWF Climate Data Store API
│   ├── dmi_api.py           # DMI Open Data (tide gauges, meteo)
│   ├── cmems_client.py      # Copernicus Marine (SSH, currents)
│   └── climate_indices.py   # NAO, AO, ENSO from NOAA
├── preprocessors/
│   ├── tensor_builder.py    # NetCDF → Tensore multi-dim
│   ├── normalization.py     # Z-score per variable
│   └── interpolation.py     # Regrid a risoluzione comune
└── catalog.py               # Data catalog management
```

**Tensore Input**: `X ∈ ℝ^{T × C × H × W}`
- T = timesteps (es. 72 ore)
- C = channels (wind_u, wind_v, pressure, SST, ...)
- H, W = griglia lat/lon (0.25° ERA5)

### 3.2 Fingerprinting Module

```
src/fingerprinting/
├── spectrogram.py     # Signal → STFT spazio-temporale
├── peaks.py           # Peak detection (local maxima)
├── hasher.py          # LSH hashing (anchor + target pairs)
├── database.py        # Vector DB (FAISS) per impronte
└── matcher.py         # Nearest-neighbor search
```

**Algoritmo** (ispirato Dejavu/FAST):
```python
def extract_fingerprint(tensor_subgrid):
    # 1. STFT su ogni channel
    spectrograms = [stft(channel) for channel in tensor_subgrid]
    
    # 2. Stack e normalizza
    combined = np.stack(spectrograms).max(axis=0)
    
    # 3. Trova picchi (local maxima)
    peaks = find_peaks_2d(combined, threshold=0.8)
    
    # 4. Genera hash da coppie anchor-target
    hashes = []
    for anchor in peaks:
        for target in get_nearby_peaks(anchor, window=10):
            h = hash((anchor.freq, target.freq, target.time - anchor.time))
            hashes.append(h)
    
    return hashes
```

### 3.3 Causal Discovery Module

```
src/causal/
├── pcmci_runner.py    # Wrapper Tigramite
├── graph_builder.py   # Causal graph → edge list
└── teleconnections.py # Pre-computed patterns (NAO, ENSO)
```

**PCMCI** (Tigramite) scopre correlazioni causali con lag:
```python
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

# Trova: ENSO[t-90d] → NAO[t-30d] → Surge[t]
results = pcmci.run_pcmci(tau_max=90, pc_alpha=0.05)
causal_edges = extract_significant_links(results)
```

### 3.4 Physics-Informed GNN

```
src/physics/
├── swe.py             # Shallow Water Equations
├── boundary.py        # Danish coast boundaries (from gates/)
├── gnn_model.py       # PyTorch Geometric model
└── loss_functions.py  # data_loss + physics_loss
```

**Shallow Water Equations** (grounding fisico):
```
Continuità:  ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0
Momentum x:  ∂u/∂t + u∂u/∂x + v∂u/∂y = -g∂h/∂x + τ_wind/ρh - friction
Momentum y:  ∂v/∂t + u∂v/∂x + v∂v/∂y = -g∂h/∂y + τ_wind/ρh - friction
```

**Loss Function**:
```python
total_loss = λ_data * MSE(pred, obs) + λ_physics * SWE_residual
# λ_physics inizia alto (0.9) per evitare bias aleatorio
```

### 3.5 Gray Zone Pattern Buffer

```
src/buffer/
├── replay_buffer.py   # Deque con capacity limit
├── validator.py       # Physics check prima di upsert
└── gray_zone.py       # Patterns non ancora validati fisicamente
```

**Concetto chiave**: Correlazioni storiche forti ma non (ancora) spiegate fisicamente NON si buttano via. Vanno in "gray zone":
- Peso alto su esperienza storica
- Peso basso/temporaneo su fisica
- Mostrate al decisore con warning
- Pronte per validazione futura (più dati, più calcolo)

```python
class GrayZoneBuffer:
    def add_pattern(self, pattern, historical_confidence, physics_residual):
        if physics_residual > THRESHOLD:
            # Pattern storico forte ma fisica debole
            self.gray_zone.append({
                'pattern': pattern,
                'historical_conf': historical_confidence,  # es. 0.82
                'physics_conf': 1.0 - physics_residual,    # es. 0.40
                'status': 'awaiting_validation',
                'reason': 'Missing intermediate data / Compute too expensive'
            })
```

---

## 4. Pipeline Probabilistica a Stadi

### "Quarti di Finale" Concept

Come in un torneo: confidenza cresce passando stadi. Quando si arriva ai "quarti", evento è imminente.

| Stage | Gate | Azione se passa |
|-------|------|-----------------|
| 1 | Fingerprint match > 60% | Attiva GNN prediction |
| 2 | Ensemble confidence > 70% | Richiedi physics check |
| 3 | Physics residual < 0.05 | Pattern validato |
| 3b | Physics residual > 0.05 ma storico > 80% | Gray zone, mostra entrambi |
| Final | Combined > 80% | **ALLERTA** |

---

## 5. Cockpit Decisionale

Dashboard che mostra **entrambe le viste** al decisore umano:

### Vista "Esperienza" (Storica)
- Pattern match percentage
- Numero eventi simili trovati
- Timeline eventi passati con outcome
- Probabilità cruda da statistica

### Vista "Scienza" (Fisica)
- Physics residual (quanto viola equazioni)
- Motivo se alto: "Dati intermedi mancanti" / "Calcolo oneroso"
- Probabilità aggiustata se forziamo fisica

### Output Combinato
- **Non decide il sistema, decide l'umano**
- Slider manuale per peso esperienza/scienza
- Log per future validazioni retroattive

---

## 6. Previous Art & Repos Precotti

### Fingerprinting
| Repo | Descrizione | Uso |
|------|-------------|-----|
| [worldveil/dejavu](https://github.com/worldveil/dejavu) | Audio fingerprinting Python | Pattern matching base |
| [stanford-futuredata/FAST](https://github.com/stanford-futuredata/FAST) | Scalable similarity search | Earthquake fingerprinting |

### PINNs per Shallow Water
| Repo | Descrizione | Uso |
|------|-------------|-----|
| [tianyongsen/PINN_SWE_open](https://github.com/tianyongsen/PINN_SWE_open) | 2D SWE + topography + rainfall | **Best starting point** |
| [abihlo/pinnsSWE](https://github.com/abihlo/pinnsSWE) | SWE su sfera rotante | Grounding fisico avanzato |
| [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs) | Original PINN implementation | Reference |
| [saidezand/PINN](https://github.com/saidezand/PINN) | Compound flooding | Multi-forcing |

### GNN per Weather/Climate
| Repo | Descrizione | Uso |
|------|-------------|-----|
| [google-deepmind/graphcast](https://github.com/google-deepmind/graphcast) | SOTA weather prediction | Architecture reference |
| [pytorch_geometric_temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) | Spatio-temporal GNN | DCRNN, A3T-GCN |
| [NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) | Physics-informed ML | Fast inference |

### Storm Surge Specifico
| Repo | Descrizione | Uso |
|------|-------------|-----|
| [Timh37/surgeNN](https://github.com/Timh37/surgeNN) | NN per surge North Sea | Denmark-ready |
| [PatrickESA/StormSurgeCastNet](https://github.com/PatrickESA/StormSurgeCastNet) | Dataset globale + Transformer | Multi-decade data |

### Causal Discovery
| Repo | Descrizione | Uso |
|------|-------------|-----|
| [jakobrunge/tigramite](https://github.com/jakobrunge/tigramite) | PCMCI algorithm | Teleconnections |
| [py-why/causal-learn](https://github.com/py-why/causal-learn) | Causal discovery algorithms | Alternative methods |

---

## 7. Teleconnessioni Rilevanti per Danimarca

### NAO (North Atlantic Oscillation)
- **NAO+**: Westerlies più forti → più storm tracks → più surge North Sea
- **NAO-**: Storm track spostato sud → meno eventi ma più estremi
- Spiega ~30% varianza livello mare invernale

### ENSO → NAO (con lag 2-3 mesi)
- El Niño può modulare NAO con ritardo
- Pattern non ovvio: evento Pacifico → Europa

### Depressions Atlantic
- Low pressure Iberia/Biscay → migrazione NE → surge Jutland/Copenhagen
- Lag tipico: 24-48 ore

---

## 8. Struttura Directory Proposta

```
surge-shazam-dk/
├── pyproject.toml
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── config.py           # Variable mapping (fork da nico)
│   │   ├── constants.py        # g, ρ, friction coefficients
│   │   └── coordinates.py      # Danish coast regions
│   │
│   ├── data/
│   │   ├── loaders/
│   │   ├── preprocessors/
│   │   └── catalog.py
│   │
│   ├── fingerprinting/
│   │   ├── spectrogram.py
│   │   ├── peaks.py
│   │   ├── hasher.py
│   │   └── matcher.py
│   │
│   ├── causal/
│   │   ├── pcmci_runner.py
│   │   └── graph_builder.py
│   │
│   ├── physics/
│   │   ├── swe.py
│   │   ├── gnn_model.py
│   │   └── loss_functions.py
│   │
│   ├── pipeline/
│   │   ├── stages.py
│   │   ├── gates.py
│   │   └── ensemble.py
│   │
│   ├── buffer/
│   │   ├── replay_buffer.py
│   │   ├── validator.py
│   │   └── gray_zone.py
│   │
│   └── visualization/
│       ├── maps.py
│       └── dashboard.py
│
├── app/                        # Streamlit cockpit
│   ├── main.py
│   ├── components/
│   │   ├── experience_view.py
│   │   ├── science_view.py
│   │   └── combined_output.py
│   └── state.py
│
├── models/                     # Trained models
├── data/                       # Raw data cache
├── notebooks/                  # Experiments
└── tests/
```

---

## 9. Lessons Learned

1. **Fingerprinting (Shazam/FAST)**: Super-robusto al rumore, scalabile
2. **PINNs**: Migliorano generalizzazione con pochi dati, ma tuning loss critico
3. **GNNs**: Perfetti per dati sparsi irregolari (stazioni meteo)
4. **Hybrid vince**: ML puro overfitta, physics puro è lento
5. **Gray zone essenziale**: Non buttare correlazioni storiche solo perché fisica non ancora pronta
6. **Cockpit trasparente**: Sistema propone, umano decide

---

## 10. Rischi e Mitigazioni

| Rischio | Mitigazione |
|---------|-------------|
| PINNs non convergono | Multi-stage training, curriculum learning |
| Correlazioni spurie | Physics loss alto iniziale, PCMCI filtering |
| Dati DMI insufficienti | Integrare ERA5, CMEMS, tide gauges multipli |
| Computational cost | Pretraining offline, surrogate models |
| Concept drift (clima cambia) | Re-training periodico, monitoring |

---

## 11. Roadmap Implementazione

### Phase 1: Data Foundation (2-3 settimane)
- [ ] DMI API client
- [ ] ERA5 loader
- [ ] Tensor builder

### Phase 2: Fingerprinting (4-6 settimane)
- [ ] STFT-based features
- [ ] Peak detection (da Dejavu)
- [ ] FAISS database

### Phase 3: PINN Core (6-8 settimane)
- [ ] Fork tianyongsen/PINN_SWE_open
- [ ] Adapt per Danish coast
- [ ] Physics loss tuning

### Phase 4: GNN Integration (4-6 settimane)
- [ ] Causal graph da PCMCI
- [ ] PyTorch Geometric model
- [ ] Ensemble pipeline

### Phase 5: Dashboard (2-3 settimane)
- [ ] Streamlit app (pattern da nico)
- [ ] Dual view (esperienza + scienza)
- [ ] Alert system

---

*Documento creato: 22 Dicembre 2025*
*Progetto: Surge-Shazam-DK*
*Status: Architecture Design*
