# Next Steps: Graph UX & Multi-Resolution Data Analysis

> **Data**: 24 Dicembre 2024  
> **Riferimento**: [EarthKit ECMWF](https://earthkit.ecmwf.int/)

---

## 🎯 Obiettivo

Costruire un'esperienza utente centrata sul **grafo causale** che permetta di:
- Comprendere le correlazioni (anche non ovvie) tra variabili
- Gestire dati con **risoluzioni eterogenee** (spaziali e temporali)
- Selezionare il **dataset ottimale** tra diverse fonti satellite
- Esplorare le relazioni attraverso **drill-down interattivo**

---

## 📊 1. Gestione Griglie Multi-Risoluzione

### Problema
I dati provengono da fonti con risoluzioni diverse:
- **Temperatura**: griglia ~1 km² (es. 3 stazioni nello stesso punto)
- **Precipitazioni**: griglia ~7 km² (es. 1 sola stazione nell'area)

### Soluzione Proposta
| Componente | Descrizione |
|------------|-------------|
| **Resolution Metadata** | Ogni variabile nel grafo porta con sé la sua risoluzione nativa |
| **Grid Alignment Layer** | Sistema di interpolazione/aggregazione per confrontare dati |
| **Visual Indicator** | Badge sulla node del grafo che indica la risoluzione |

### Issue da Creare
- [ ] `feat: Grid resolution metadata per ogni variabile nel grafo`
- [ ] `feat: Interpolazione automatica per allineare griglie diverse`
- [ ] `ui: Badge visuale risoluzione su nodi del grafo`

---

## 🔍 2. Graph-Centric UX (Hypothesis Department)

### Interazioni Pianificate

#### 2.1 Hover → Help Snippet
> ✅ Già implementato (bottom-right)

Mostra informazioni contestuali sulla connessione/nodo.

#### 2.2 Double-Click → Drill-Down Dimensionale
**Espande il grafo** aggiungendo una dimensione:
- Click su edge → mostra i **lag temporali** della correlazione
- Click su nodo → mostra le **sotto-variabili** o la **provenienza dati**
- Visualizzazione gerarchica delle dipendenze

```
[Temperatura] ─────── [Precipitazioni]
                │
          double-click
                ↓
[Temperatura]         [Precipitazioni]
    ├── ERA5 (1km)        ├── GPM (10km)
    ├── MERRA2 (0.5°)     └── Station (point)
    └── Station (point)
```

### Issue da Creare
- [ ] `feat: Double-click drill-down su nodi del grafo causale`
- [ ] `feat: Espansione edge per mostrare lag temporali`
- [ ] `feat: Sotto-grafo per provenienza dati per variabile`

---

## 🛰️ 3. Dataset Selection Cockpit

### Workflow
```
┌─────────────────────────────────────────────────────────────┐
│                     COCKPIT INTERFACE                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │         AREA SELECTION (Map + Search Bar)           │   │
│  │  • LLM suggerisce coordinate                        │   │
│  │  • Oppure: search bar per area di interesse         │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         AVAILABLE DATASETS FOR AREA                  │   │
│  │  ☑ ERA5          (0.25° | hourly | 1979-now)       │   │
│  │  ☐ MERRA-2       (0.5°  | hourly | 1980-now)       │   │
│  │  ☐ GPM IMERG     (0.1°  | 30min  | 2000-now)       │   │
│  │  ☑ CMEMS         (0.08° | daily  | 1993-now)       │   │
│  │  ☐ Custom Sat... (var)                              │   │
│  │                                                      │   │
│  │  💡 LLM Suggestion: "Per eventi alluvionali,       │   │
│  │     consiglio ERA5 + GPM per la precipitazione"    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              HYPOTHESIS GRAPH VIEW                   │   │
│  │         (Causal Graph - già implementato)           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Fonti Dataset Previste
- **ERA5** - Reanalysis ECMWF
- **MERRA-2** - NASA Reanalysis
- **GPM IMERG** - Precipitazioni globali
- **CMEMS** - Dati oceanografici
- **Stazioni locali** - Dati puntuali ad alta precisione
- **Custom satellite** - EMSAT, RIOSAT, altri...

### Issue da Creare
- [ ] `feat: Dataset selector panel nel cockpit`
- [ ] `feat: LLM suggestions per dataset ottimale dato il caso d'uso`
- [ ] `feat: Area selection via search bar + map click`
- [ ] `feat: Dataset metadata display (resolution, coverage, temporal range)`

---

## ⏱️ 4. Timeline & Geographic Controls

### 4.1 Timeline Temporale
Controllo per esplorare **elementi precursori**:
- Slider temporale con "pin" draggabile
- Animazione play/pause per vedere evoluzione
- Window temporale selezionabile (es. -7 giorni → +3 giorni)

```
◄──────────●────────────────►
   -30d    NOW           +7d
   
   [▶ Play] [⏸ Pause] [⏹ Reset]
```

### 4.2 Geographic Zoom Control
Espandere l'area per trovare **correlazioni esterne**:
- Zoom in/out sulla mappa
- Buffer geografico configurabile
- Visualizzazione heatmap delle correlazioni spaziali

```
┌─────────────────────────────────┐
│         MAP VIEW                │
│    ┌───────────────────┐        │
│    │   Original Area   │        │
│    │    (selected)     │        │
│    └───────────────────┘        │
│  ┌─────────────────────────┐    │
│  │     Expanded Buffer     │    │
│  │  (+50km for precursors) │    │
│  └─────────────────────────┘    │
│                                 │
│  [─────●─────] Buffer: 50km     │
│  [Zoom In] [Zoom Out] [Reset]   │
└─────────────────────────────────┘
```

### Issue da Creare
- [ ] `feat: Timeline slider con pin per esplorazione temporale`
- [ ] `feat: Play/pause animation per evoluzione temporale`
- [ ] `feat: Geographic buffer control per espansione area`
- [ ] `feat: Map view integrata con grafo causale`
- [ ] `feat: Heatmap correlazioni spaziali su mappa`

---

## 🗺️ 5. Ispirazione: EarthKit ECMWF

Riferimento: https://earthkit.ecmwf.int/

Elementi da considerare:
- Interfaccia data exploration
- Selezione multi-dataset
- Visualizzazione su mappa
- API per accesso dati

---

## 📋 Riepilogo Issue da Creare

### Priority 1 - Core Graph UX
1. `feat: Double-click drill-down su nodi del grafo causale`
2. `feat: Grid resolution metadata per ogni variabile nel grafo`
3. `ui: Badge visuale risoluzione su nodi del grafo`

### Priority 2 - Dataset Selection
4. `feat: Dataset selector panel nel cockpit`
5. `feat: LLM suggestions per dataset ottimale`
6. `feat: Area selection via search bar + map click`

### Priority 3 - Temporal/Spatial Controls
7. `feat: Timeline slider con pin per esplorazione temporale`
8. `feat: Geographic buffer control per espansione area`
9. `feat: Map view integrata con grafo causale`

### Priority 4 - Advanced
10. `feat: Interpolazione automatica per allineare griglie diverse`
11. `feat: Heatmap correlazioni spaziali su mappa`
12. `feat: Play/pause animation per evoluzione temporale`

---

## 🔗 Collegamenti Interni

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Architettura sistema
- [AGENT_LAYER_ARCHITECTURE.md](./AGENT_LAYER_ARCHITECTURE.md) - Layer agente
- [DATASET_CONFIG.md](./DATASET_CONFIG.md) - Configurazione dataset

---

*Documento creato per tracciare i prossimi passi di sviluppo del sistema di analisi causale multi-risoluzione.*
