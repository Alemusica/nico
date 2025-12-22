# 📓 Notebooks

Jupyter notebooks per analisi esplorative e sviluppo.

## File

### SLCCI PLOTTER SINGLE STRAIT.ipynb
Notebook originale per l'analisi DOT di singoli stretti:
- Caricamento dati SLCCI
- Filtraggio per area geografica
- Calcolo DOT (SSH - MSS)
- Binning per longitudine
- Regressione lineare per slope
- Visualizzazione 12-subplot mensile

## Utilizzo

```bash
# Avvia Jupyter
jupyter notebook notebooks/

# Oppure in VS Code
# Apri direttamente il file .ipynb
```

## Note

- I notebook sono per **esplorazione** e **prototipazione**
- Il codice stabile va in `src/`
- Usa i moduli da `src/` nei notebook:

```python
import sys
sys.path.insert(0, '..')

from src.data.loaders import load_filtered_cycles
from src.analysis.slope import compute_slope
```
