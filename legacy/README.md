# 📜 Legacy Code

Codice originale mantenuto per riferimento e retrocompatibilità.

## File

### j2_utils.py (768 linee)
Utilities originali per l'analisi Jason-2, ora refactorizzate in moduli separati:

| Funzione Originale | Nuovo Modulo |
|-------------------|--------------|
| `load_filtered_cycles_serial_J2()` | `src/data/loaders.py` |
| `interpolate_geoid()` | `src/data/geoid.py` |
| `add_geoid_to_cycles()` | `src/data/geoid.py` |
| `filter_by_pass()` | `src/data/filters.py` |
| coordinate utilities | `src/core/coordinates.py` |

## ⚠️ Deprecazione

Questo codice è **deprecato**. Usa i moduli in `src/` per nuovo sviluppo:

```python
# ❌ Vecchio modo
from legacy.j2_utils import load_filtered_cycles_serial_J2

# ✅ Nuovo modo
from src.data.loaders import load_filtered_cycles
```

## Mantenimento

Il file è mantenuto per:
1. Riferimento durante la migrazione
2. Retrocompatibilità con script esistenti
3. Documentazione delle scelte implementative originali
