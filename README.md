# 🛰️ SLCCI Satellite Altimetry + Causal Discovery Platform

A comprehensive Python toolkit for **satellite altimetry analysis** and **intelligent causal discovery** with LLM-powered explanations.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-LLM-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This project combines **oceanographic data analysis** with **AI-powered causal discovery**:

### Core Features
- 🛰️ **Satellite Altimetry** - DOT, SLA, SSH analysis from Jason/CMEMS/AVISO
- 🔬 **Causal Discovery** - PCMCI algorithm to find cause-effect relationships with time lags
- 🤖 **LLM Integration** - Ollama (qwen3-coder) for automatic data interpretation
- ⚡ **Physics Validation** - Validate patterns against physical laws (wind setup, inverse barometer)
- 📊 **Pattern Detection** - tsfresh features, association rules, anomaly detection

### New: Intelligent Causal Discovery Pipeline

```
Dataset → LLM Interprets → Find Time Dimension → PCMCI Discovery → Physics Validation → LLM Explains
```

**Example**: Load flood data → LLM identifies "sea_level_anomaly" as target → PCMCI finds "precipitation → river_level (lag=2 days)" → Physics confirms wind setup mechanism → LLM explains the Atlantic storm track connection.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd nico

# Use Python 3.12 (recommended - 3.14 has compatibility issues)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For causal discovery features
pip install tigramite networkx fastapi uvicorn ollama
```

### 2. Start Ollama (for LLM features)

```bash
# Install Ollama: https://ollama.ai
ollama pull qwen3-coder:30b  # or llama3.2 for faster inference
ollama serve
```

### 3. Run the API Server

```bash
# Start FastAPI backend
uvicorn api.main:app --reload --port 8000
```

### 4. Run Headless Test

```bash
python test_headless.py
```

Expected output:
```
✅ PASS: llm (Ollama connected, data interpreted)
✅ PASS: causal (Found precipitation→river_level, wind→surge)
✅ PASS: satellite (Loaded AVISO/CMEMS data)
✅ PASS: llm_explain (Physics validation: 0.95)
```

---

## 📁 Project Structure

```
nico/
├── api/                          # 🔌 FastAPI Backend (NEW)
│   ├── main.py                   # REST endpoints
│   └── services/
│       ├── llm_service.py        # Ollama LLM integration
│       ├── causal_service.py     # PCMCI causal discovery
│       └── data_service.py       # Dataset loading/preprocessing
│
├── src/                          # 🧠 Core Analysis Modules
│   ├── analysis/                 # DOT, slope, statistics
│   ├── core/                     # Config, coordinates, resolvers
│   ├── data/                     # Loaders, filters, geoid
│   ├── visualization/            # Plotly/Matplotlib charts
│   ├── pattern_engine/           # Pattern detection (tsfresh, mlxtend)
│   │   ├── core/                 # Pattern dataclasses
│   │   ├── detection/            # ML detectors, association rules
│   │   ├── physics/              # Domain rules (flood, manufacturing)
│   │   └── output/               # Gray zone detector
│   └── surge_shazam/             # Physics-informed ML
│       ├── physics/              # Shallow water equations (PyTorch)
│       └── causal/               # PCMCI integration (stubs)
│
├── app/                          # 📱 Streamlit Dashboard
│   └── components/               # UI tabs (analysis, spatial, profiles)
│
├── data/                         # 📂 Satellite Data
│   ├── aviso/                    # AVISO altimetry
│   ├── cmems/                    # CMEMS L3/L4
│   ├── slcci/                    # SLCCI Jason-1/2
│   └── geoid/                    # TUM geoid model
│
├── gates/                        # 🌊 Strait Shapefiles
│
├── test_headless.py              # 🧪 Integration tests
└── gradio_app.py                 # Alternative Gradio UI
```

---

## 🔬 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API + Ollama status |
| `/data/files` | GET | List available data files |
| `/data/upload` | POST | Upload CSV/NetCDF |
| `/data/load/{path}` | GET | Load file from data/ |
| `/interpret` | POST | LLM interprets dataset structure |
| `/discover` | POST | Run PCMCI causal discovery |
| `/discover/correlations` | POST | Cross-correlation analysis |
| `/chat` | POST | Chat with LLM about data |
| `/hypotheses` | POST | Generate causal hypotheses |
| `/ws/chat` | WebSocket | Stream LLM responses |

### Example: Causal Discovery

```bash
curl -X POST http://localhost:8000/discover \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "flood_data",
    "max_lag": 7,
    "alpha_level": 0.05,
    "domain": "flood",
    "use_llm": true
  }'
```

Response:
```json
{
  "variables": ["precipitation", "wind_speed", "pressure", "river_level", "flood_index"],
  "links": [
    {
      "source": "precipitation",
      "target": "river_level",
      "lag": 2,
      "strength": 0.95,
      "p_value": 0.0001,
      "explanation": "Heavy precipitation causes river levels to rise with a 2-day lag...",
      "physics_valid": true,
      "physics_score": 0.92
    }
  ]
}
```

---

## 🧠 LLM Service Features

The `OllamaLLMService` provides:

### 1. Data Interpretation
```python
result = await llm.interpret_dataset(columns_info, filename)
# Returns: domain="flood", temporal_column="timestamp", suggested_targets=["sea_level"]
```

### 2. Causal Explanation
```python
explanation = await llm.explain_causal_relationship(
    source="wind_speed", target="storm_surge", lag=1, strength=0.52
)
# Returns: "Wind speed causes storm surge through the wind setup mechanism (τ ∝ U²)..."
```

### 3. Physics Validation
```python
validation = await llm.validate_pattern_physics(
    pattern="wind → surge", domain="flood", confidence=0.99
)
# Returns: {"is_valid": True, "physics_score": 0.95, "supporting_evidence": ["wind stress formula"]}
```

### 4. Hypothesis Generation
```python
hypotheses = await llm.generate_hypotheses(variables, domain="flood")
# Returns: [{"source": "NAO_index", "target": "storm_surge", "expected_lag": "3-5 days"}]
```

---

## ⚡ Physics Rules

Built-in physics validation for multiple domains:

### Flood/Storm Surge
| Rule | Formula | Typical Lag |
|------|---------|-------------|
| Wind Setup | η ∝ U²·L/(g·h) | 6-24 hours |
| Inverse Barometer | Δη ≈ -1 cm/hPa | 12-48 hours |
| Pressure Effect | Low pressure → surge | 24-72 hours |

### Manufacturing
| Rule | Effect |
|------|--------|
| Temperature | Arrhenius: rate ×2 per 10°C |
| Viscosity | Decreases with temperature |
| Speed | Optimal range for quality |

---

## 🛠️ Development

### Run Tests
```bash
# Headless integration test
python test_headless.py

# Unit tests
pytest tests/
```

### Code Quality
```bash
black src/ api/
ruff check src/ api/
```

### Known Issues

⚠️ **Python 3.14 Compatibility**: NetworkX and some libraries have issues with Python 3.14. Use Python 3.12 for now.

---

## 🗺️ Roadmap

### ✅ Completed (v1.0)
- [x] FastAPI backend with REST endpoints
- [x] Ollama LLM integration (qwen3-coder, llama3.2)
- [x] PCMCI causal discovery with correlation fallback
- [x] Physics validation rules (flood, manufacturing)
- [x] Data interpretation and explanation generation
- [x] NetCDF/CSV loading with auto-detection
- [x] Headless test pipeline
- [x] Pattern engine (tsfresh, mlxtend, pyod)

### 🚧 In Progress (v1.1)
- [ ] React frontend with PHI spacing layout
- [ ] Interactive causal graph visualization (D3.js)
- [ ] Real-time chat with WebSocket streaming
- [ ] Time series explorer with lag slider

### 📋 Planned (v2.0)
- [ ] Neo4j for causal graph persistence
- [ ] RAG with scientific papers (ChromaDB)
- [ ] Multi-dataset correlation analysis
- [ ] Export to standard causal formats (TETRAD, DOT)
- [ ] Teleconnection patterns (NAO, ENSO)
- [ ] Automated report generation

---

## 📚 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     React Frontend (TODO)                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │ Causal Graph│ │ Chat (LLM)  │ │ Time Series │                │
│  │ (D3.js)     │ │ Interface   │ │ Explorer    │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                           │ REST/WebSocket
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (/api)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                │
│  │ LLM Service │ │ Causal      │ │ Data        │                │
│  │ (Ollama)    │ │ Discovery   │ │ Service     │                │
│  └─────────────┘ └─────────────┘ └─────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Core Analysis + Pattern Engine                      │
│  (DOT analysis, tsfresh, mlxtend, physics validation)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📄 License

MIT License

---

## 🤝 Contributing

Key areas for contribution:
1. **React Frontend** - Build the PHI-spaced dashboard with D3.js graphs
2. **Physics Rules** - Add domain-specific validation rules
3. **LLM Prompts** - Improve scientific explanation quality
4. **Test Data** - Contribute synthetic/real datasets
