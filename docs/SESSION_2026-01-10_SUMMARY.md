# 📋 RIEPILOGO SESSIONE 2026-01-10 + TODO DOMANI

## ✅ COSA ABBIAMO FATTO OGGI

### 1. Geostrophic Velocity Tab (NUOVO)
**File**: `app/components/tabs.py` - Funzione `_render_geostrophic_velocity_tab_cmems_l4()`

Creato nuovo tab per confrontare due metodi di calcolo velocità:
- **v_perp**: Velocità perpendicolare da ugos/vgos CMEMS L4
- **v_geo**: Velocità geostrofica da slope ADT

**Features implementate**:
- Monthly dropdown (seleziona 1 dei 12 mesi)
- Spatial averaging slider (media ogni 1-50 km, default 5km)
- Bathymetry profile GEBCO con dual x-axis (km + gradi)
- Velocity comparison plot (v_perp vs v_geo)
- Transport bar chart mensile (×10⁶ m³/s)
- Time series comparison
- Statistics: correlazione, bias, RMSE

### 2. Volume Transport Tab (REWRITE)
**File**: `app/components/tabs.py` - Funzione `_render_volume_transport_tab_cmems_l4()`

Completamente riscritta con nuovo layout:
1. **Bathymetry FIRST** (in cima come richiesto)
2. Transport bar chart (monthly climatology)
3. Time series completa
4. Statistics (mean, std, min, max)

**Cambiamenti**:
- Units: ×10⁶ m³/s invece di Sv/mSv
- Dual x-axis su tutti i plot spaziali (km + gradi)
- Blue/red colors per valori positivi/negativi
- Always GEBCO but cap at 250m

### 3. Transport Service (NUOVO)
**File**: `src/services/transport_service.py`

Nuove funzioni per calcoli velocità:
```python
compute_perpendicular_velocity(ugos, vgos, gate_angle)
compute_gate_angles(gate_lons, gate_lats)
compute_monthly_along_gate_profile(matrix, time_dt, x_km, month)
compute_spatial_average(x_km, values, bin_width_km)
```

### 4. Fix Bug Divided Gates
**File**: `app/components/loaders/base.py` - Funzione `apply_longitude_filter()`

**BUG**: Longitude filtering NON filtrava le matrici ugos/vgos!

**Fix**: Aggiunto filtering per velocity matrices
```python
if ugos_matrix is not None and len(ugos_matrix) > 0:
    new_ugos_matrix = ugos_matrix[mask, :]
if vgos_matrix is not None and len(vgos_matrix) > 0:
    new_vgos_matrix = vgos_matrix[mask, :]
```

Ora divided gates (fram_strait_west, davis_strait_east) funzionano correttamente.

### 5. Correzione Formula Geostrofica
**File**: `app/components/tabs.py` - Line 3925

**PROBLEMA**: v_perp e v_geo avevano segni opposti

**Before**: `v_geo_ts = -g / f * slope_m_m`  ❌
**After**: `v_geo_ts = g / f * slope_m_m`   ✅

Formula corretta: `v_geo = +g/f × (∂η/∂x)`

### 6. Visual Enhancement System
**Nuovo file**: `app/components/chart_style.py`

Modulo centralizzato per styling consistente:
```python
# Color palette
NAVY_BLUE = "#1E3A5F"    # Primary
CORAL = "#E07B53"         # Comparison
SKY_BLUE = "#3498DB"      # Positive
SOFT_RED = "#E74C3C"      # Negative
WHITE_BG = "#FFFFFF"      # Background
LIGHT_GRAY = "#E8E8E8"    # Gridlines

# Helper functions
get_chart_layout(title, xaxis_title, yaxis_title)
get_navy_line()
get_coral_line()
```

**File NON committato** (escluso da .gitignore): `.streamlit/config.toml`
- White background theme
- Navy primary color
- Vedi `docs/ISSUES/STREAMLIT_CONFIG_MANUAL.md` per setup

### 7. Aggiornamento Documentazione
- `docs/PROGRESS.md` - Riepilogo completo sessione
- `docs/CHAT_HISTORY.md` - Context per prossima sessione
- `docs/ISSUES/ISSUE_2026-01-10_CRITICAL_FIXES.md` - Bug dettagliati

---

## 🐛 PROBLEMI RISCONTRATI

### 🔴 CRITICO: Plotly `secondary_x` Error
**Status**: ❌ NON RISOLTO - APP CRASHA

**Error**:
```
ValueError: Invalid key specified in an element of the 'specs' argument to make_subplots: 'secondary_x'
Valid keys include: ['type', 'secondary_y', 'colspan', 'rowspan', 'l', 'r', 'b', 't']
```

**Location**: `app/components/tabs.py` line ~3981
```python
fig_profile = make_subplots(specs=[[{"secondary_x": True}]])  # ❌ SBAGLIATO!
```

**Impact**: Geostrophic Velocity tab NON si apre, app crasha

**Causa**: Plotly NON supporta `secondary_x`, solo `secondary_y`

### 🟡 Monthly Analysis: Missing Slope/R²
**Status**: ⚠️ DA VERIFICARE

Nel tab "🟣 CMEMS L4 - Monthly Analysis", i subplot mensili (Sep, Oct, Nov, Dec) mostrano il linear fit ma **mancano i valori numerici**.

**Expected**: Annotations con "Slope: X.XX cm/100km | R²: 0.XXX"
**Actual**: Solo scatter + fit line, no text

### 🟢 Deprecation Warning
**Status**: ⚠️ DA FIXARE (low priority)

Console piena di warnings:
```
Please replace `use_container_width` with `width`.
For use_container_width=True, use width='stretch'
```

Circa 10-15 occorrenze in `app/components/tabs.py`

---

## 📋 TODO DOMANI - PRIORITY ORDER

### Priority 0: Fix Crashes (BLOCKING)
**Tempo stimato**: 30-60 minuti

- [ ] **Fix `secondary_x` error** in Geostrophic Velocity tab
  - Location: `app/components/tabs.py` line ~3981
  - Soluzione: Usare dual x-axis manuale (vedi Issue doc)
  - Test: Tab si apre senza crash

### Priority 1: Missing Data
**Tempo stimato**: 20-30 minuti

- [ ] **Aggiungere slope/R² annotations** in Monthly Analysis
  - Location: funzione `_render_unified_monthly_analysis()`
  - Aggiungere text annotations su ogni subplot
  - Test: Verificare tutti i 12 mesi hanno valori visibili

### Priority 2: Code Quality
**Tempo stimato**: 10 minuti

- [ ] **Replace `use_container_width`** con `width='stretch'`
  - Batch replace in `app/components/tabs.py`
  - Cerca anche in altri files `app/components/`
  - Test: No warnings in console

### Priority 3: Consistency
**Tempo stimato**: 30 minuti

- [ ] **Standardizza dual x-axis** su tutti i plot spaziali
  - Bathymetry plots (Volume + Geostrophic tabs)
  - Velocity profile
  - Transport bar charts (se ha senso)

### Priority 4: Testing Completo
**Tempo stimato**: 20 minuti

- [ ] Test con diversi gates (bering, fram, davis)
- [ ] Test divided gates (West/East)
- [ ] Verifica v_perp vs v_geo signs coerenti
- [ ] Test spatial averaging diversi bin widths
- [ ] Edge cases: empty data, NaN values

---

## 📂 FILES MODIFICATI OGGI

| File | Status | Lines | Descrizione |
|------|--------|-------|-------------|
| `app/components/tabs.py` | ✅ Committed | ~600 | Rewrite Volume + new Geostrophic tabs |
| `app/components/loaders/base.py` | ✅ Committed | ~15 | Fix divided gates velocity filtering |
| `src/services/transport_service.py` | ✅ Committed | ~150 | New velocity calculation functions |
| `app/components/chart_style.py` | ✅ Committed | NEW | Centralized styling module |
| `.streamlit/config.toml` | ❌ NOT committed | NEW | Theme config (excluded by .gitignore) |
| `docs/PROGRESS.md` | ✅ Committed | ~300 | Session summary |
| `docs/CHAT_HISTORY.md` | ✅ Committed | ~50 | Context update |
| `docs/ISSUES/ISSUE_2026-01-10_CRITICAL_FIXES.md` | ✅ Committed | NEW | Detailed bug report |

---

## 🎯 COME PROCEDERE DOMANI

### Step 1: Setup Environment (5 min)
```bash
cd /Users/nicolocaron/Documents/GitHub/nico
git pull origin feature/gates-streamlit
source .venv/bin/activate

# Setup .streamlit config (se non esiste)
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
base = "light"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
primaryColor = "#1E3A5F"
textColor = "#2C3E50"
font = "sans serif"
EOF
```

### Step 2: Fix Critical Bug (30-60 min)
1. Apri `app/components/tabs.py`
2. Cerca line ~3981: `make_subplots(specs=[[{"secondary_x": True}]])`
3. Implementa soluzione dual x-axis manuale (vedi Issue doc)
4. Test: `streamlit run streamlit_app.py --server.port 8501`
5. Verifica tab Geostrophic Velocity si apre

### Step 3: Add Missing Annotations (20-30 min)
1. Apri `app/components/tabs.py`
2. Cerca funzione `_render_unified_monthly_analysis()`
3. Aggiungi `fig.add_annotation()` per ogni subplot con slope/R²
4. Test: Verifica valori visibili su tutti i 12 mesi

### Step 4: Batch Fixes (10 min)
```bash
# Replace use_container_width
cd /Users/nicolocaron/Documents/GitHub/nico
sed -i '' 's/use_container_width=True/width="stretch"/g' app/components/tabs.py
sed -i '' 's/use_container_width=False/width="content"/g' app/components/tabs.py
```

### Step 5: Testing & Commit (20 min)
```bash
# Test different gates
# - bering_strait
# - fram_strait_west (divided)
# - davis_strait_east (divided)

# If all OK:
git add -A
git commit -m "🐛 Fix critical bugs: secondary_x error + missing annotations"
git push origin feature/gates-streamlit
```

---

## 📚 REFERENCE DOCS

### Per Domani
- `docs/ISSUES/ISSUE_2026-01-10_CRITICAL_FIXES.md` - Bug dettagliati + soluzioni
- `docs/ISSUES/STREAMLIT_CONFIG_MANUAL.md` - Setup theme config
- `docs/PROGRESS.md` - Full session history

### Plotly Docs
- Subplots: https://plotly.com/python/subplots/
- Multiple axes: https://plotly.com/python/multiple-axes/
- Dual x-axis example:
  ```python
  fig.update_layout(
      xaxis2=dict(
          title="Secondary X",
          overlaying="x",
          side="top"
      )
  )
  ```

### Formulas Reference
```python
# Perpendicular velocity
v_perp = vN * cos(θ) + vE * sin(θ)

# Geostrophic velocity (CORRECTED SIGN)
v_geo = +g/f × (∂η/∂x)
where:
  g = 9.81 m/s²
  f = 2Ω sin(φ)  [Coriolis]
  Ω = 7.2921e-5 rad/s

# Volume transport
Q = Σ v_perp × h × Δx
Output: ×10⁶ m³/s
```

---

## 🎨 STYLE GUIDE REFERENCE

**Colors**:
- Primary: Navy Blue #1E3A5F
- Secondary: Coral #E07B53
- Positive: Sky Blue #3498DB
- Negative: Soft Red #E74C3C
- Background: White #FFFFFF
- Grid: Light Gray #E8E8E8

**Fonts**:
- Family: Inter, sans-serif
- Title: 16pt bold
- Axis labels: 12pt
- Hover text: 11pt

**Layout Rules**:
- Always white background
- No rounded corners
- Subtle gridlines (#E8E8E8)
- Dual x-axis on spatial plots (km + degrees)
- Units: ×10⁶ m³/s for transport, cm/s for velocity

---

## 💾 COMMIT INFO

**Commit hash**: `d40d6d0`
**Branch**: `feature/gates-streamlit`
**Date**: 2026-01-10 22:00

**Message**:
```
🎨 Visual enhancement + Geostrophic Velocity comparison (WIP - has bugs)

✅ Completato:
- Rewrite Volume Transport tab: bathymetry first, dual x-axis, m³/s units
- New Geostrophic Velocity tab: v_perp vs v_geo comparison, spatial averaging
- Fix divided gates: ugos/vgos matrices now filtered correctly
- Sign correction: v_geo formula changed from -g/f to +g/f
- Visual enhancement: white theme, navy/coral colors, elegant styling
- New chart_style.py module for centralized styling

🐛 Problemi da sistemare domani:
- CRITICO: Plotly secondary_x error (tab Geostrophic Velocity crasha)
- Monthly Analysis: mancano valori slope/R² sui grafici
- Deprecation: use_container_width → width='stretch'
```

**Remote**: https://github.com/Alemusica/nico.git
