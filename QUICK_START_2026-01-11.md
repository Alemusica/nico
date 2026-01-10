# 🚀 QUICK START - 2026-01-11

## ⚡ START HERE

```bash
cd /Users/nicolocaron/Documents/GitHub/nico
git pull origin feature/gates-streamlit
source .venv/bin/activate
```

## 🔴 PROBLEMA #1: App Crasha (PRIORITY 0)

**Errore**: `ValueError: Invalid key 'secondary_x'`
**File**: `app/components/tabs.py` line ~3981
**Tab**: Geostrophic Velocity

### Fix veloce:
```python
# BEFORE (line ~3981):
fig_profile = make_subplots(specs=[[{"secondary_x": True}]])  # ❌

# AFTER - Opzione A (dual x-axis manuale):
fig_profile = go.Figure()
fig_profile.update_layout(
    xaxis=dict(title="Distance (km)", side="bottom"),
    xaxis2=dict(title="Longitude (°E)", side="top", overlaying="x")
)

# AFTER - Opzione B (rimuovi secondary):
fig_profile = make_subplots(rows=1, cols=1)  # ✅
```

Vedi `docs/ISSUES/ISSUE_2026-01-10_CRITICAL_FIXES.md` per soluzione dettagliata.

## 🟡 PROBLEMA #2: Missing Slope/R²

**Tab**: Monthly Analysis
**File**: `app/components/tabs.py` - funzione `_render_unified_monthly_analysis()`

Aggiungi annotations:
```python
fig.add_annotation(
    text=f"Slope: {slope:.2f} cm/100km<br>R²: {r2:.3f}",
    xref=f"x{idx}", yref=f"y{idx}",
    x=0.95, y=0.95,
    xanchor="right", yanchor="top",
    showarrow=False
)
```

## 🟢 PROBLEMA #3: Deprecation Warning

**Batch fix**:
```bash
cd /Users/nicolocaron/Documents/GitHub/nico
sed -i '' 's/use_container_width=True/width="stretch"/g' app/components/tabs.py
```

## 📚 DOCS COMPLETI

- **Bug details**: `docs/ISSUES/ISSUE_2026-01-10_CRITICAL_FIXES.md`
- **Full summary**: `docs/SESSION_2026-01-10_SUMMARY.md`
- **Progress**: `docs/PROGRESS.md`

## ✅ CHECKLIST

- [ ] Fix secondary_x error (30 min)
- [ ] Add slope/R² annotations (20 min)
- [ ] Replace use_container_width (5 min)
- [ ] Test all gates (15 min)
- [ ] Commit & push

## 🎯 Expected Outcome

- ✅ Geostrophic Velocity tab si apre
- ✅ Monthly Analysis mostra slope/R² su ogni subplot
- ✅ No warnings in console
- ✅ v_perp e v_geo hanno segni coerenti
- ✅ Dual x-axis su tutti i plot spaziali

---

**Last commit**: `f327c2a`
**Time estimate**: ~1.5 hours total
