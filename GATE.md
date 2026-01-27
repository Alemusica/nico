# 🚨 GATE - MANDATORY CONTEXT FOR ALL AGENTS

> ⚠️ **STOP! READ THIS ENTIRE FILE BEFORE ANY ACTION**  
> 📅 Last Updated: 2026-01-27
> 🔄 Auto-updated on each commit via pre-commit hook

---

## 🔒 MANDATORY CHECKLIST (DO BEFORE ANYTHING)

```
□ 1. Read this ENTIRE file (GATE.md)
□ 2. Run: git fetch && git pull origin $(git branch --show-current)
□ 3. Understand the architecture (Section 2)
□ 4. Check current state (Section 3)
□ 5. Review known issues (Section 4)
□ 6. THEN and ONLY THEN proceed with your task
```

---

# 📐 SECTION 1: PROJECT IDENTITY

| Field | Value |
|-------|-------|
| **Project** | NICO - Arctic Ocean Gate Analysis |
| **Purpose** | Analyze oceanographic data through Arctic gates/straits |
| **Current Branch** | `feature/gates-streamlit` |
| **Main UI** | Streamlit (port 8501) |
| **API** | FastAPI (port 8000) |
| **Database** | SurrealDB (port 8001) |
| **Python** | `.venv/bin/python` (ALWAYS use this!) |

---

# 🏗️ SECTION 2: ARCHITECTURE (MACRO VIEW)

## 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NICO ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     PRESENTATION LAYER                               │   │
│  │  ┌───────────────────┐  ┌───────────────────────────────────────┐  │   │
│  │  │   Streamlit App   │  │   React + Cosmograph (master only)    │  │   │
│  │  │   Port: 8501      │  │   Port: 5173                          │  │   │
│  │  │   streamlit_app.py│  │   frontend/                           │  │   │
│  │  └─────────┬─────────┘  └───────────────────────────────────────┘  │   │
│  └────────────┼────────────────────────────────────────────────────────┘   │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      API GATEWAY LAYER                               │   │
│  │                        FastAPI (api/)                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  /gates    /data    /analysis    /knowledge    /pipeline    │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SERVICES LAYER (src/services/)                    │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │GateService  │ │SLCCIService │ │CMEMSService │ │ExportService│   │   │
│  │  │gate_service │ │slcci_service│ │cmems_l4_svc │ │export_svc   │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │DTUService   │ │BathyService │ │TransportSvc │ │CacheService │   │   │
│  │  │dtu_service  │ │gebco_service│ │transport_svc│ │cache_service│   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       CORE LAYER (src/core/)                         │   │
│  │  ┌───────────────┬───────────────┬───────────────┬───────────────┐  │   │
│  │  │  models.py    │coordinates.py │  config.py    │logging_config │  │   │
│  │  │  (Pydantic)   │(geo utils)    │(yaml loader)  │(centralized)  │  │   │
│  │  └───────────────┴───────────────┴───────────────┴───────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DATA ACCESS LAYER                                 │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐  │   │
│  │  │ gates/*.shp  │ │ config/*.yaml│ │ data/*.nc    │ │ GEBCO     │  │   │
│  │  │ (shapefiles) │ │ (configs)    │ │ (NetCDF)     │ │ (830MB)   │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Directory Structure

```
nico/
├── GATE.md                 ← 🚨 YOU ARE HERE - READ FIRST!
├── .github/
│   └── copilot-instructions.md  ← Points to this GATE.md
│
├── config/                 # ✅ CENTRALIZED CONFIG
│   ├── gates.yaml          # Gate definitions (bbox, datasets)
│   ├── gate_passes.yaml    # Pre-computed satellite passes
│   ├── datasets.yaml       # Dataset definitions
│   └── defaults.yaml       # Default parameters
│
├── src/                    # ✅ CORE PYTHON MODULES
│   ├── core/               # Shared utilities
│   │   ├── models.py       # Pydantic models
│   │   ├── coordinates.py  # Geo utilities
│   │   └── config.py       # Config loader
│   │
│   ├── services/           # ⭐ MAIN BUSINESS LOGIC
│   │   ├── gate_service.py     # Gate operations
│   │   ├── slcci_service.py    # ESA SLCCI data
│   │   ├── cmems_l4_service.py # CMEMS L4 gridded
│   │   ├── dtu_service.py      # DTU Space data
│   │   ├── gebco_service.py    # Bathymetry
│   │   ├── transport_service.py # Volume transport
│   │   └── export_service.py   # Export ZIP/PNG
│   │
│   └── gates/              # Gate module
│       └── catalog.py      # GateCatalog class
│
├── app/                    # ✅ STREAMLIT UI
│   ├── main.py             # Entry point
│   └── components/
│       ├── sidebar.py      # Gate selector
│       └── tabs.py         # ⭐ ALL VISUALIZATIONS
│
├── api/                    # ✅ FASTAPI BACKEND
│   └── main.py             # FastAPI app
│
├── gates/                  # Shapefile data (*.shp)
├── data/                   # Data files and cache
└── .venv/                  # Python virtual environment
```

## 2.3 Key Services

| Service | File | Purpose |
|---------|------|---------|
| GateService | gate_service.py | Gate management |
| SLCCIService | slcci_service.py | ESA SLCCI data |
| CMESL4Service | cmems_l4_service.py | CMEMS L4 gridded |
| DTUService | dtu_service.py | DTU Space data |
| GEBCOService | gebco_service.py | Bathymetry |
| ExportService | export_service.py | Export ZIP/PNG |

---

# 📊 SECTION 3: CURRENT STATE

## 3.1 What Works ✅

- Gate selection (sidebar.py)
- SLCCI/CMEMS/DTU data loading
- Slope timeline, DOT profile
- Monthly analysis (3×4 grid)
- Volume transport + GEBCO bathymetry
- Basic export (6 functions)

## 3.2 In Progress 🔄

- New export functions (6 pending)
- Bathymetry visual fix (0 at top)

## 3.3 Broken ❌

- None currently

---

# ⚠️ SECTION 4: KNOWN ISSUES

## 4.1 Python Environment

```bash
# ✅ CORRECT
source .venv/bin/activate
.venv/bin/python script.py

# ❌ WRONG
python3 script.py
pip3 install package
```

## 4.2 Service Ports

| Service | Port |
|---------|------|
| Streamlit | 8501 |
| API | 8000 |
| SurrealDB | 8001 |

## 4.3 Streamlit Session State Keys

| Key | Description |
|-----|-------------|
| vt_transport_total_sv | Volume transport |
| vt_v_perp | Perpendicular velocity |
| vt_monthly_profiles | Monthly profiles |
| sf_flux_data | Salt flux |

---

# 📁 SECTION 5: KEY FILES

| File | Lines | Purpose |
|------|-------|---------|
| app/components/tabs.py | ~7000 | All visualizations |
| src/services/export_service.py | ~1000 | Export functions |
| src/services/slcci_service.py | ~600 | SLCCI loading |

---

# 🔧 SECTION 6: COMMON TASKS

## Adding Export Function

```python
def export_your_function(data, gate_name, dpi=300) -> bytes:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    # ... plot ...
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
```

## Quick Commands

```bash
# Start Streamlit
streamlit run streamlit_app.py --server.port 8501

# Verify edit
grep -n "function_name" path/to/file.py
```

---

# 📝 SECTION 7: CHANGE LOG

| Date | Changes |
|------|---------|
| 2026-01-26 | Initial GATE.md creation |

---

**🔚 END OF GATE.md - PROCEED WITH TASK**
