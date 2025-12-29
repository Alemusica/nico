# 🌊 Agent Gates - Istruzioni

> **Branch**: `feature/gates-streamlit`  
> **Focus**: Streamlit app per analisi gates + selezione dataset  
> **Last Updated**: 2025-12-29

---

## ⚠️ LEGGI PRIMA DI INIZIARE

Questo documento è per l'agent che lavora su **Streamlit + Gates + Dataset**.  
Se devi lavorare su **React/API/Knowledge Graph**, vai su branch `master` e leggi `docs/AGENT_FULLSTACK.md`.

---

## 🎯 Scope di questo Agent

### ✅ PUOI modificare:
```
streamlit_app.py    # App Streamlit principale
demo_dashboard.py   # Dashboard demo
gates/              # Shapefile dei gates oceanografici
data/               # Dataset e cache
notebooks/          # Jupyter notebooks per analisi
```

### ❌ NON modificare:
```
frontend/           # React (gestito da Agent Full Stack)
api/                # FastAPI backend (gestito da Agent Full Stack)
src/                # Core modules (gestito da Agent Full Stack)
```

---

## 📋 TODO Prioritizzati

### 🔴 ALTA PRIORITÀ
1. **Selezione Dataset interattiva**
   - UI Streamlit per scegliere regione/periodo
   - Integrazione con `catalog.yaml` (11 dataset disponibili)
   - Visualizzazione preview dati

2. **Analisi Gates**
   - Caricare shapefile da `gates/`
   - Visualizzare su mappa (Folium/PyDeck)
   - Calcolare flussi attraverso i gates

3. **Workflow completo Streamlit**
   - Select region → Load data → Show time series → Run analysis

### 🟡 MEDIA PRIORITÀ
4. **Grafici interattivi** con Plotly
5. **Export risultati** (CSV, NetCDF)
6. **Integrazione con pipeline esistenti**

### 🟢 BASSA PRIORITÀ
7. Caching avanzato
8. Multi-user support

---

## 🔧 Come avviare

```bash
cd /Users/alessioivoycazzaniga/nico
source .venv/bin/activate

# Streamlit principale
streamlit run streamlit_app.py --server.port 8501

# Dashboard demo (alternativa)
streamlit run demo_dashboard.py --server.port 8502
```

---

## 📂 Gates Disponibili

```
gates/
├── barents_sea_opening_S3_pass_481.shp
├── barents_sea-central_arctic_ocean.shp
├── barents_sea-kara_sea.shp
├── beaufort_sea-canadian_arctic_archipelago.shp
├── beaufort_sea-central_arctic_ocean.shp
├── bering_strait_TPJ_pass_076.shp
├── canadian_arctic_archipelago-central_arctic_ocean.shp
├── davis_strait.shp
└── ... (altri gates)
```

### Formato Shapefile:
- Proiezione: WGS84 (EPSG:4326)
- Geometria: LineString o Polygon
- Attributi: nome gate, regioni connesse

---

## 📊 Dataset Disponibili (catalog.yaml)

| Dataset | Provider | Latency | Variabili |
|---------|----------|---------|-----------|
| CMEMS Sea Level | Copernicus | 🟢 1 day | SSH, SLA, ADT |
| CMEMS Currents | Copernicus | 🟢 1 day | u, v velocities |
| ERA5 Reanalysis | ECMWF | 🟡 5 days | wind, pressure, precip |
| NOAA Climate Indices | NOAA | 🟢 daily | NAO, ENSO, AO |
| CYGNSS Wind | NASA | 🟢 2-24h | surface wind speed |
| SLCCI Lakes | ESA CCI | 🔴 months | lake levels |

### Caricare il catalogo:
```python
from src.data_manager.intake_bridge import get_catalog, search_catalog

catalog = get_catalog()
# Cerca per variabile
results = search_catalog(variable="sea_level")
# Cerca per latenza
fast_data = search_catalog(latency="near_real_time")
```

---

## 🗺️ Visualizzazione Mappe

### Con Folium (consigliato per gates):
```python
import folium
import geopandas as gpd
import streamlit as st
from streamlit_folium import st_folium

# Carica gate
gate = gpd.read_file("gates/bering_strait_TPJ_pass_076.shp")

# Crea mappa
m = folium.Map(location=[65, -170], zoom_start=4)
folium.GeoJson(gate).add_to(m)

# Mostra in Streamlit
st_folium(m, width=700, height=500)
```

### Con PyDeck (per grandi dataset):
```python
import pydeck as pdk
import streamlit as st

layer = pdk.Layer(
    "GeoJsonLayer",
    data=gate.__geo_interface__,
    get_fill_color=[255, 0, 0, 100],
)
st.pydeck_chart(pdk.Deck(layers=[layer]))
```

---

## 📈 Grafici Time Series

```python
import plotly.express as px
import streamlit as st

# Dati esempio
df = pd.DataFrame({
    "time": pd.date_range("2000-01-01", periods=100),
    "sea_level": np.random.randn(100).cumsum()
})

fig = px.line(df, x="time", y="sea_level", title="Sea Level Anomaly")
st.plotly_chart(fig, use_container_width=True)
```

---

## 🔗 Integrazione con API (opzionale)

Se hai bisogno di dati dal backend:
```python
import requests

# Catalogo
response = requests.get("http://localhost:8000/api/v1/data/catalog")
datasets = response.json()

# Time series
response = requests.get("http://localhost:8000/api/v1/timeseries/frames", params={
    "dataset_id": "lago_maggiore_2000",
    "variable": "precipitation"
})
```

---

## ⚠️ Note Importanti

1. **NON modificare l'API** - usa solo gli endpoint esistenti
2. **NON toccare React** - il frontend è gestito da altro agent
3. **Commit frequenti** su questo branch
4. **Test locale** prima di ogni commit

---

## 📚 File Principali da Conoscere

| File | Descrizione | Linee |
|------|-------------|-------|
| `streamlit_app.py` | App legacy SLCCI | ~500 |
| `demo_dashboard.py` | Dashboard demo nuova | ~630 |
| `catalog.yaml` | Catalogo multi-provider | ~200 |
| `src/data_manager/intake_bridge.py` | Bridge catalogo | ~180 |

---

## 🧪 Testing

```bash
# Verifica shapefile
python -c "import geopandas as gpd; print(gpd.read_file('gates/bering_strait_TPJ_pass_076.shp').head())"

# Verifica catalogo
python -c "from src.data_manager.intake_bridge import get_catalog; print(get_catalog().keys())"

# Verifica Streamlit
streamlit run streamlit_app.py --server.headless true
```

---

**Autore**: NICO Project  
**Branch**: feature/gates-streamlit  
**Ultimo aggiornamento**: 2025-12-29
