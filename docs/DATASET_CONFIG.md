# 🔧 Dataset Configuration System

Sistema centralizzato per la gestione di variabili multi-formato.

## Concetto

Diversi dataset altimetrici (SLCCI, CMEMS, AVISO) usano **nomi diversi** per le stesse variabili:

| Concetto | SLCCI | CMEMS L3 | CMEMS L4 | AVISO |
|----------|-------|----------|----------|-------|
| Sea Surface Height | `corssh` | `sla_filtered` | `sla` | `sla` |
| Mean Sea Surface | `mean_sea_surface` | - | - | - |
| Dynamic Topography | (calcolato) | `adt` | `adt` | `adt` |
| Latitude | `latitude` | `latitude` | `latitude` | `lat` |

Il **Variable Resolver** permette di usare **nomi canonici** indipendenti dal formato:

```python
from src.core import VariableResolver

# Auto-detect del formato
resolver = VariableResolver.from_file("data/slcci/SLCCI_ALTDB_J1_Cycle001_V2.nc")

# Accesso con nome canonico - funziona con QUALSIASI formato!
ssh = resolver.get("ssh")  # → corssh in SLCCI, sla in CMEMS
lat, lon = resolver.get_coordinates()
```

## Architettura

```
┌──────────────────────────────────────────────────────┐
│                    User Code                          │
│   resolver.get("ssh")  resolver.get("mss")           │
└─────────────────────────┬────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│                 VariableResolver                      │
│   - Auto-detect formato                              │
│   - Mapping nomi canonici → nomi dataset             │
│   - Caching                                          │
└─────────────────────────┬────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────┐
│                 DatasetConfig                         │
│   - VariableMapping per ogni variabile               │
│   - CoordinateMapping                                │
│   - Metadati (epoch, lon_convention, flags)          │
└─────────────────────────┬────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ SLCCI_J1 │    │ CMEMS_L3 │    │  AVISO   │
    │  Config  │    │  Config  │    │  Config  │
    └──────────┘    └──────────┘    └──────────┘
```

## Formati Supportati

### SLCCI (Sea Level CCI)
- `slcci_j1` - Jason-1 Mission
- `slcci_j2` - Jason-2 Mission

Variabili: `corssh`, `mean_sea_surface`, `swh`, `wind_speed_alt`, `bathymetry`, `sigma0`

### CMEMS (Copernicus Marine)
- `cmems_l3` - Along-Track Level 3
- `cmems_l4` - Gridded Level 4

Variabili L3: `sla_filtered`, `sla_unfiltered`, `adt`, `mdt`, `dac`, `ocean_tide`
Variabili L4: `sla`, `adt`, `ugos`, `vgos` (velocità geostrofiche)

### AVISO
- `aviso` - AVISO/DUACS products

## Aggiungere un Nuovo Formato

```python
from src.core.config import (
    DatasetConfig,
    DatasetFormat,
    VariableMapping,
    CoordinateMapping,
    register_dataset_config,
)

# Definisci configurazione
MY_CONFIG = DatasetConfig(
    format=DatasetFormat.CUSTOM,
    name="My Custom Format",
    description="Description here",
    variables={
        "ssh": VariableMapping(
            canonical_name="ssh",
            dataset_name="my_ssh_variable",
            units="m",
            description="Sea Surface Height",
        ),
        # ... altre variabili
    },
    coordinates=CoordinateMapping(
        latitude="my_lat",
        longitude="my_lon",
        time="my_time",
    ),
)

# Registra nel sistema
register_dataset_config("my_format", MY_CONFIG)

# Ora puoi usarlo
resolver = VariableResolver.from_dataset(ds, format_hint="my_format")
```

## API Reference

### VariableResolver

```python
class VariableResolver:
    # Factory methods
    @classmethod
    def from_dataset(cls, dataset, format_hint=None) -> VariableResolver
    
    @classmethod
    def from_file(cls, filepath) -> VariableResolver
    
    # Accesso variabili
    def get(self, canonical_name, as_array=True) -> np.ndarray
    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]
    def get_time(self) -> np.ndarray
    
    # Utilità
    def has_variable(self, canonical_name) -> bool
    def list_available(self) -> list[str]
    def compute_dot(self, reference="mss") -> np.ndarray
    def get_quality_mask(self) -> np.ndarray | None
```

### DatasetConfig

```python
@dataclass
class DatasetConfig:
    format: DatasetFormat
    name: str
    description: str
    variables: dict[str, VariableMapping]
    coordinates: CoordinateMapping
    reference_epoch: str
    longitude_convention: str  # "0_360" o "-180_180"
    quality_flag_var: str | None
    quality_flag_valid: int | tuple[int, ...]
```

## Esempio Completo

```python
from src.core import VariableResolver, compare_formats

# Mostra formati supportati
compare_formats()

# Carica dati SLCCI
resolver_slcci = VariableResolver.from_file("data/slcci/SLCCI_ALTDB_J1_Cycle001_V2.nc")
print(f"Formato: {resolver_slcci.format_name}")
print(f"Variabili: {resolver_slcci.list_available()}")

ssh_slcci = resolver_slcci.get("ssh")
dot_slcci = resolver_slcci.compute_dot()

# Stesso codice funziona con CMEMS!
resolver_cmems = VariableResolver.from_file("data/cmems/some_cmems_file.nc")
ssh_cmems = resolver_cmems.get("ssh")  # Usa automaticamente "sla_filtered"

# Accesso unificato alle coordinate
lat, lon = resolver_slcci.get_coordinates()  # Funziona con qualsiasi formato
```
