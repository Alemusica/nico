# 🔴 CRITICAL ISSUES - 2026-01-10

## Issue #1: Plotly `secondary_x` Error (CRITICO)

### Descrizione
L'app crasha quando si apre il tab "Geostrophic Velocity" con errore:
```
ValueError: Invalid key specified in an element of the 'specs' argument to make_subplots: 'secondary_x'
Valid keys include: ['type', 'secondary_y', 'colspan', 'rowspan', 'l', 'r', 'b', 't']
```

### Location
- **File**: `app/components/tabs.py`
- **Line**: ~3981
- **Function**: `_render_geostrophic_velocity_tab_cmems_l4()`

### Codice Problematico
```python
fig_profile = make_subplots(specs=[[{"secondary_x": True}]])  # ❌ NON SUPPORTATO!
```

### Causa Root
Plotly `make_subplots()` NON supporta `secondary_x`. Solo `secondary_y` è valido.

### Soluzione Proposta

#### Opzione A: Dual X-Axis Manuale (RACCOMANDATO)
```python
from plotly.graph_objects import Figure, Scatter

fig_profile = Figure()

# Primary x-axis (km)
fig_profile.add_trace(
    Scatter(
        x=x_km,
        y=velocity_values,
        name="Velocity",
        line=dict(color=NAVY_BLUE),
        xaxis="x"  # Primary
    )
)

# Update layout con secondary x-axis
fig_profile.update_layout(
    xaxis=dict(
        title="Distance (km)",
        side="bottom"
    ),
    xaxis2=dict(
        title="Longitude (°E)",
        side="top",
        overlaying="x",
        tickmode="sync"  # Sincronizza con primary
    )
)

# Per aggiungere trace su secondary x:
fig_profile.add_trace(
    Scatter(
        x=gate_lons,
        y=velocity_values,
        xaxis="x2",  # Secondary x-axis
        showlegend=False
    )
)
```

#### Opzione B: Due Subplot Side-by-Side
```python
from plotly.subplots import make_subplots

fig_profile = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Distance (km)", "Longitude (°E)"),
    horizontal_spacing=0.05
)

fig_profile.add_trace(
    Scatter(x=x_km, y=velocity_values, ...),
    row=1, col=1
)

fig_profile.add_trace(
    Scatter(x=gate_lons, y=velocity_values, ...),
    row=1, col=2
)
```

### Files da Modificare
1. `app/components/tabs.py` - Line ~3981
2. Verificare anche altre occorrenze di `secondary_x` nel file

### Testing
- [ ] Aprire tab Geostrophic Velocity senza crash
- [ ] Verificare dual x-axis funziona correttamente
- [ ] Testare con diversi gates (bering_strait, fram_strait, etc.)

---

## Issue #2: Missing Slope/R² Values in Monthly Analysis

### Descrizione
Nel tab "🟣 CMEMS L4 - Monthly Analysis", i subplot mensili mostrano il linear fit ma **mancano i valori numerici** di slope e R² sui grafici.

### Location
- **File**: `app/components/tabs.py`
- **Function**: `_render_unified_monthly_analysis()`
- **Visible in**: Screenshot attachment - mesi Sep, Oct, Nov, Dec

### Problema
I grafici mostrano:
- ✅ DOT data points (scatter)
- ✅ Linear fit line (red)
- ❌ Slope value annotation (MISSING!)
- ❌ R² value annotation (MISSING!)

### Codice da Controllare
Cercare nella funzione `_render_unified_monthly_analysis()`:
```python
# Dovrebbe esserci qualcosa tipo:
fig.add_annotation(
    text=f"Slope: {slope:.2f} cm/100km<br>R²: {r2:.3f}",
    xref=f"x{idx}",  # Per subplot specifico
    yref=f"y{idx}",
    x=0.95,  # Posizione relativa
    y=0.95,
    xanchor="right",
    yanchor="top",
    showarrow=False,
    font=dict(size=10, color="#2C3E50")
)
```

### Possibili Cause
1. **Annotations mancanti**: Non vengono aggiunte al subplot
2. **Posizionamento sbagliato**: Fuori dal viewport
3. **Calcolo R² non fatto**: Slope calcolato ma R² no
4. **Variabile non passata**: Funzione calcola ma non usa valori

### Fix Richiesto
1. Identificare dove vengono creati i 12 subplot mensili
2. Per ogni subplot, aggiungere annotation con:
   - Slope in cm/100km
   - R² con 3 decimali
3. Posizionare top-right di ogni subplot
4. Usare font size 10-11pt per leggibilità

### Testing
- [ ] Verificare tutti i 12 mesi hanno slope/R² visibili
- [ ] Controllare posizionamento non sovrappone dati
- [ ] Verificare valori sono corretti (cross-check con slope_series)

---

## Issue #3: Deprecation Warning `use_container_width`

### Descrizione
Warning ripetuto in console ogni volta che si renderizza un chart:
```
Please replace `use_container_width` with `width`.
use_container_width will be removed after 2025-12-31.
For use_container_width=True, use width='stretch'
For use_container_width=False, use width='content'
```

### Location
- **File**: `app/components/tabs.py` (principalmente)
- **Occorrenze**: ~10-15 volte

### Pattern da Cercare
```python
# OLD (deprecated):
st.plotly_chart(fig, use_container_width=True)
st.plotly_chart(fig, use_container_width=False)

# NEW:
st.plotly_chart(fig, width='stretch')
st.plotly_chart(fig, width='content')
```

### Comando per Trovare Tutte le Occorrenze
```bash
cd /Users/nicolocaron/Documents/GitHub/nico
grep -rn "use_container_width" app/components/tabs.py
```

### Fix Batch
Usare search & replace con VSCode o sed:
```bash
sed -i '' 's/use_container_width=True/width="stretch"/g' app/components/tabs.py
sed -i '' 's/use_container_width=False/width="content"/g' app/components/tabs.py
```

### Files da Verificare
1. `app/components/tabs.py` (main)
2. `app/components/sidebar.py`
3. Altri file in `app/components/`

### Testing
- [ ] Nessun warning in console
- [ ] Charts si visualizzano correttamente
- [ ] Width responsive funziona

---

## Issue #4: Dual X-Axis Implementation Consistency

### Descrizione
Alcuni plot hanno dual x-axis (km + degrees), altri no. Serve **consistency across all spatial plots**.

### Files da Verificare
- Volume Transport tab: bathymetry, transport bar chart
- Geostrophic Velocity tab: bathymetry, velocity profile, transport bars

### Pattern da Implementare
```python
# Layout con dual x-axis
fig.update_layout(
    xaxis=dict(
        title="Distance (km)",
        side="bottom",
        showgrid=True,
        gridcolor=LIGHT_GRAY,
    ),
    xaxis2=dict(
        title="Longitude (°E)",
        side="top",
        overlaying="x",
        showgrid=False,
        tickmode="linear",
        dtick=2.0  # Ogni 2 gradi
    )
)

# Aggiungere trace invisibile per secondary x
fig.add_trace(
    go.Scatter(
        x=gate_lons,
        y=values,
        xaxis="x2",
        showlegend=False,
        opacity=0  # Invisibile, solo per scale
    )
)
```

### Checklist
- [ ] Bathymetry plot (Volume Transport) - dual x-axis
- [ ] Bathymetry plot (Geostrophic Velocity) - dual x-axis
- [ ] Velocity profile plot - dual x-axis
- [ ] Transport bar chart - dual x-axis se ha sense

---

## Priority Matrix

| Issue | Priority | Impact | Effort | Status |
|-------|----------|--------|--------|--------|
| #1 secondary_x error | 🔴 P0 | App crash | Medium | TODO |
| #2 Missing slope/R² | 🟡 P1 | Data missing | Low | TODO |
| #3 Deprecation warning | 🟢 P2 | Console noise | Low | TODO |
| #4 Dual x-axis consistency | 🟢 P2 | UX inconsistency | Medium | TODO |

---

## Testing Checklist

### Before Fixes
- [x] Document all errors and warnings
- [x] Take screenshots of problematic UI
- [x] Identify exact line numbers

### After Fixes
- [ ] No crashes on any tab
- [ ] No warnings in console
- [ ] All slope/R² values visible
- [ ] Dual x-axis on all spatial plots
- [ ] Test with multiple gates
- [ ] Test with divided gates (West/East)
- [ ] Verify v_perp vs v_geo signs are correct

### Edge Cases
- [ ] Empty data handling
- [ ] Single data point
- [ ] All NaN values
- [ ] Very large/small values

---

## Notes per Domani

### Approccio Raccomandato
1. **Fix #1 FIRST** (blocking): Risolvi secondary_x error
2. **Test basic functionality**: Verifica tab si apre
3. **Fix #2**: Aggiungi slope/R² annotations
4. **Fix #3**: Batch replace use_container_width
5. **Fix #4**: Standardizza dual x-axis
6. **Full testing**: Test completo tutti i gates

### Tools Utili
```bash
# Restart Streamlit dopo modifiche
pkill -f streamlit
source .venv/bin/activate
streamlit run streamlit_app.py --server.port 8501

# Check logs
tail -f logs/nico.log

# Find patterns
grep -rn "secondary_x" app/
grep -rn "use_container_width" app/
```

### Reference
- Plotly subplots docs: https://plotly.com/python/subplots/
- Dual axis: https://plotly.com/python/multiple-axes/
- Streamlit charts: https://docs.streamlit.io/library/api-reference/charts
