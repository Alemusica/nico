# 📊 Data Directory

Questa cartella contiene i dati NetCDF per l'analisi altimetrica satellitare.

## Struttura

```
data/
├── slcci/          # SLCCI Altimeter Database (Jason-1, Jason-2)
│   ├── SLCCI_ALTDB_J1_Cycle001_V2.nc
│   ├── SLCCI_ALTDB_J1_Cycle002_V2.nc
│   └── ...
├── geoid/          # Dati geoide per calcolo DOT
│   └── TUM_ogmoc.nc
└── cmems/          # (futuro) Dati CMEMS
```

## SLCCI Data

### Formato Nome File
```
SLCCI_ALTDB_{SATELLITE}_Cycle{NNN}_V2.nc
```
- `J1` = Jason-1
- `J2` = Jason-2
- `NNN` = Numero ciclo (001-XXX)

### Variabili Principali
| Variabile | Descrizione | Unità |
|-----------|-------------|-------|
| `corssh` | Corrected Sea Surface Height | m |
| `mean_sea_surface` | Mean Sea Surface | m |
| `latitude` | Latitudine | gradi |
| `longitude` | Longitudine | gradi |
| `TimeDay` | Giorni dal 2000-01-01 | giorni |
| `validation_flag` | Flag qualità (0=valido) | - |
| `swh` | Significant Wave Height | m |
| `bathymetry` | Batimetria | m |

### Copertura
- **Latitudine**: ~-66° a +66° (limite orbitale Jason)
- **Longitudine**: 0° a 360°
- **Temporale**: Varia per missione

## Geoid Data

### TUM_ogmoc.nc
Geoide **TUM (Technical University of Munich)** per il calcolo del DOT:

```
DOT = SSH - Geoid
```

## ⚠️ Note

1. **File grandi** - I file .nc sono esclusi da Git (vedi .gitignore)
2. **Download** - Contatta il team per accesso ai dati
3. **Backup** - Mantieni backup locali dei dati originali
