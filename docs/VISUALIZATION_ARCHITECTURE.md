# 📊 NICO Visualization Architecture

> **State of the Art** - Documentazione completa dell'architettura di visualizzazione per SLCCI e futuri dataset.

---

## 🎯 Obiettivo

Creare visualizzazioni **standardizzate** per analisi di trasporto oceanico attraverso gate (passaggi satellite).
L'architettura è progettata per essere **estendibile** ad altri dataset (CMEMS, ERA5, etc.) mantenendo le stesse forme di visualizzazione.

---

## 📐 Architettura delle 4 Tabs

### Tab 1: Slope Timeline
**Scopo**: Mostrare l'evoluzione temporale della pendenza SSH lungo il gate.

| Attributo | Tipo | Descrizione |
|-----------|------|-------------|
| `slope_series` | `np.ndarray` | Serie temporale delle pendenze (m/100km) |
| `time_array` | `np.ndarray` | Array di date/timestamp |
| `time_periods` | `list` | Etichette dei periodi (es. "2024-01") |

**X-axis**: Date (da `time_array`)
**Y-axis**: Slope (m/100km o cm/km)

```python
# Logica essenziale
slope_series = getattr(pass_data, 'slope_series', None)
time_array = getattr(pass_data, 'time_array', None)

fig.add_trace(go.Scatter(
    x=time_array,
    y=slope_series,
    mode="markers+lines"
))
```

---

### Tab 2: DOT Profile
**Scopo**: Mostrare il profilo medio del DOT attraverso il gate (da WEST a EAST).

| Attributo | Tipo | Descrizione |
|-----------|------|-------------|
| `profile_mean` | `np.ndarray` | Media temporale del DOT per ogni bin spaziale |
| `x_km` | `np.ndarray` | Distanza in km lungo la longitudine |
| `dot_matrix` | `np.ndarray` | Matrice [space, time] dei valori DOT |

**X-axis**: Distance along longitude (km) da `x_km`
**Y-axis**: DOT (m) da `profile_mean`

```python
# Logica essenziale
profile_mean = getattr(pass_data, 'profile_mean', None)
x_km = getattr(pass_data, 'x_km', None)

fig.add_trace(go.Scatter(
    x=x_km,
    y=profile_mean,
    mode="lines"
))

# Labels WEST/EAST ai lati
fig.add_annotation(x=x_km.min(), text="WEST")
fig.add_annotation(x=x_km.max(), text="EAST")
```

---

### Tab 3: Spatial Map
**Scopo**: Visualizzazione geografica delle misurazioni.

| Attributo | Tipo | Descrizione |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | DataFrame con colonne lat, lon, dot, corssh, etc. |
| `gate_lon_pts` | `np.ndarray` | Punti longitudine della linea del gate |
| `gate_lat_pts` | `np.ndarray` | Punti latitudine della linea del gate |

```python
# Logica essenziale
fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="dot",
    color_continuous_scale="viridis"
)

# Linea del gate
fig.add_trace(go.Scattermapbox(
    lat=gate_lat_pts,
    lon=gate_lon_pts,
    mode="lines",
    name="Gate"
))
```

---

### Tab 4: Monthly Analysis
**Scopo**: 12 subplot (uno per mese) con DOT vs Longitude + regressione lineare.

| Attributo | Tipo | Descrizione |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | DataFrame con colonne month, lon, lat, dot |

**Calcolo della pendenza (slope)**:
```python
# Conversione lon → km
R_earth = 6371.0  # km
lat_rad = np.deg2rad(mean_lat)
dlon_rad = np.deg2rad(lon) - np.deg2rad(lon.min())
x_km = R_earth * dlon_rad * np.cos(lat_rad)

# Regressione
slope_m_km, intercept = np.polyfit(x_km, dot, 1)
slope_m_100km = slope_m_km * 100  # m/100km
```

---

## 🔌 Interfaccia PassData Standard

Qualsiasi dataset deve fornire un oggetto con questi attributi:

```python
@dataclass
class PassData:
    """Standard interface for satellite pass data."""
    
    # Metadata
    strait_name: str
    pass_number: int
    
    # Tab 1: Slope Timeline
    slope_series: np.ndarray      # Shape: (n_periods,)
    time_array: np.ndarray        # Shape: (n_periods,) - datetime objects
    time_periods: List[str]       # ["2024-01", "2024-02", ...]
    
    # Tab 2: DOT Profile
    profile_mean: np.ndarray      # Shape: (n_lon_bins,)
    x_km: np.ndarray              # Shape: (n_lon_bins,)
    dot_matrix: np.ndarray        # Shape: (n_lon_bins, n_periods)
    
    # Tab 3 & 4: Spatial & Monthly
    df: pd.DataFrame              # Columns: lat, lon, dot, month, time, cycle, ...
    gate_lon_pts: np.ndarray      # Gate line longitude points
    gate_lat_pts: np.ndarray      # Gate line latitude points
```

---

## 🔧 Parametri di Configurazione

Da `SLCCIConfig` in sidebar:

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `lon_bin_size` | float | 0.05 | Dimensione bin longitudine (gradi) |
| `lat_tolerance` | float | 0.5 | Tolleranza latitudine per filtraggio |
| `min_points` | int | 10 | Minimo punti per periodo valido |
| `start_date` | date | 2019-01-01 | Data inizio analisi |
| `end_date` | date | 2024-12-31 | Data fine analisi |

---

## 📊 Flusso Dati

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCE                               │
│   NetCDF (SLCCI) │ CMEMS API │ ERA5 API │ Other Sources         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SERVICE LAYER                             │
│   SLCCIService.load_pass_data()                                 │
│   CMEMSService.load_pass_data()  (future)                       │
│   ERA5Service.load_pass_data()   (future)                       │
│                                                                  │
│   Output: PassData object with standard attributes              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SESSION STATE                                │
│   st.session_state["slcci_pass_data"] = PassData(...)           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TABS LAYER                                │
│   tabs.py → render_tabs()                                       │
│   ├── _render_slope_timeline()    uses: slope_series, time_array│
│   ├── _render_dot_profile()       uses: profile_mean, x_km      │
│   ├── _render_spatial_map()       uses: df, gate_*_pts          │
│   └── _render_monthly_analysis()  uses: df (with month column)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Come Aggiungere un Nuovo Dataset

### 1. Creare il Service

```python
# src/services/new_dataset_service.py
class NewDatasetService:
    def load_pass_data(self, config) -> PassData:
        """Load data and return standard PassData object."""
        
        # 1. Load raw data from source
        raw_data = self._load_from_source(config)
        
        # 2. Process into standard format
        df = self._create_dataframe(raw_data)
        
        # 3. Calculate derived attributes
        slope_series, time_array = self._calculate_slopes(df, config)
        profile_mean, x_km, dot_matrix = self._calculate_profile(df, config)
        
        # 4. Return PassData
        return PassData(
            strait_name=config.gate_name,
            pass_number=config.pass_number,
            slope_series=slope_series,
            time_array=time_array,
            time_periods=time_periods,
            profile_mean=profile_mean,
            x_km=x_km,
            dot_matrix=dot_matrix,
            df=df,
            gate_lon_pts=gate_lon,
            gate_lat_pts=gate_lat
        )
```

### 2. Registrare nel Sidebar

```python
# app/components/sidebar.py
def _render_load_button():
    if dataset_type == "NEW_DATASET":
        from src.services.new_dataset_service import NewDatasetService
        service = NewDatasetService()
        pass_data = service.load_pass_data(config)
        st.session_state["slcci_pass_data"] = pass_data  # Same key!
```

### 3. Nessuna Modifica a tabs.py!

Le tabs leggono da `st.session_state["slcci_pass_data"]` e usano `getattr()` per accedere agli attributi. Finché il nuovo dataset fornisce gli stessi attributi, le visualizzazioni funzionano automaticamente.

---

## 📝 Calcoli Chiave

### Slope Calculation (da DOT matrix)

```python
def calculate_slope_series(dot_matrix, x_km):
    """
    Calculate slope for each time period.
    
    dot_matrix: shape (n_lon_bins, n_time_periods)
    x_km: shape (n_lon_bins,)
    
    Returns: slope_series shape (n_time_periods,)
    """
    n_periods = dot_matrix.shape[1]
    slopes = np.full(n_periods, np.nan)
    
    for t in range(n_periods):
        dot_t = dot_matrix[:, t]
        valid = ~np.isnan(dot_t)
        if np.sum(valid) >= 2:
            slope, _ = np.polyfit(x_km[valid], dot_t[valid], 1)
            slopes[t] = slope * 100  # m/100km
    
    return slopes
```

### DOT Profile Mean

```python
def calculate_profile_mean(dot_matrix):
    """
    Calculate mean DOT profile across time.
    
    Returns: profile_mean shape (n_lon_bins,)
    """
    return np.nanmean(dot_matrix, axis=1)
```

### Longitude to Kilometers

```python
def lon_to_km(lon_array, reference_lat):
    """Convert longitude array to km distance."""
    R_earth = 6371.0
    lat_rad = np.deg2rad(reference_lat)
    dlon_rad = np.deg2rad(lon_array) - np.deg2rad(lon_array.min())
    return R_earth * dlon_rad * np.cos(lat_rad)
```

---

## ✅ Checklist per Nuovi Dataset

- [ ] Service implementato con `load_pass_data()` method
- [ ] Restituisce oggetto con tutti gli attributi `PassData`
- [ ] `slope_series` calcolato correttamente (m/100km)
- [ ] `x_km` calcolato da longitude (non usare latitude!)
- [ ] `df` ha colonne: lat, lon, dot, month, time
- [ ] Gate line points inclusi per la mappa
- [ ] Testato con le 4 tabs esistenti

---

## 📚 Riferimenti

- **SLCCI PLOTTER Notebook**: Workflow originale per i 3 pannelli
- **tabs.py**: Implementazione corrente delle visualizzazioni
- **slcci_service.py**: Service di riferimento per SLCCI dataset

---

*Ultimo aggiornamento: 2 Gennaio 2026*
*Branch: feature/gates-streamlit*
