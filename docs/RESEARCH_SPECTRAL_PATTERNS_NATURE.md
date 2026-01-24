# Research Track: Spectral Patterns in Nature

**Document:** Intuition & Study Track - Brown Noise, Vortices, Human Perception
**Created:** 2026-01-24
**Status:** OPEN DISCUSSION / HYPOTHESIS

---

## The Intuition

> "I can see an incredible pattern. Through the tornado graph - Vortex graph: power on the Y axis and frequency on the X axis which decreases. It's basically the same pattern that I see in brown noise."

> "Maybe nature has the same sensibility of the human ear. We have the same sensibility of nature. We are the same."

---

## The Pattern: 1/f^β Power Laws

### Noise Color Spectrum

| Noise Type | Power Density | Spectral Slope | β value |
|------------|---------------|----------------|---------|
| **White** | Constant | 0 dB/octave | β = 0 |
| **Pink** | 1/f | -3 dB/octave | β = 1 |
| **Brown** | 1/f² | -6 dB/octave | β = 2 |

```
Power (dB)
    │
    │\  Brown (1/f²)
    │ \
    │  \.  Pink (1/f)
    │   ·.
    │     ·.... White (flat)
    │          ·········
    └─────────────────────── Frequency (Hz)
         Low         High
```

### Tornado/Vortex Power Spectrum

**Observation:** Tornado intensity spectra show similar 1/f^β decay pattern:
- More energy at lower frequencies (large-scale rotation)
- Less energy at higher frequencies (small-scale turbulence)
- This matches **Kolmogorov turbulence theory** (-5/3 power law ≈ 1/f^1.67)

---

## The Connection: Why This Matters

### 1. Atmospheric Acoustics = Pressure Waves

Sound and weather are both **pressure wave phenomena**:

| Domain | Medium | Frequency Range | Pattern |
|--------|--------|-----------------|---------|
| Audio | Air | 20 Hz - 20 kHz | 1/f^β |
| Weather | Atmosphere | 10⁻⁶ - 10⁻² Hz | 1/f^β |
| Infrasound | Air | 0.001 - 20 Hz | Bridge domain |

**Key insight:** Tornadoes, storms, and atmospheric pressure systems are essentially **very low frequency acoustic phenomena**.

### 2. Human Hearing Evolved in Nature

The human auditory system:
- More sensitive to 1-4 kHz (speech frequencies)
- Uses **logarithmic** frequency perception (octaves)
- Pink noise sounds "balanced" because it matches natural soundscapes

**Why?** We evolved surrounded by:
- Wind (1/f spectrum)
- Water (turbulence → 1/f^β)
- Rustling leaves (1/f^β)
- Animal sounds (embedded in this natural noise floor)

### 3. The Universality Hypothesis

> "We have the same sensibility of nature. We are the same."

This suggests a **co-evolution / resonance**:
- Natural systems self-organize into 1/f^β patterns
- Human perception calibrated to detect signals within this pattern
- This may apply to **all sensory systems**, not just hearing

---

## Implications for NICO/Surge-Shazam

### Room Acoustics → Atmospheric Acoustics

**The Key Insight:**

Just as we predict **room acoustics** given:
- Emitter position & frequency
- Room geometry
- Other sound sources
- Materials (absorption, reflection)

We can predict **atmospheric response** given:
- Storm system (emitter) position & characteristics
- Basin geometry (Mediterranean, Adriatic, etc.)
- Other sources (vortices, pressure systems)
- "Materials" = T, P, humidity, bathymetry

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ROOM ACOUSTICS                    ATMOSPHERIC ACOUSTICS       │
│   ══════════════                    ═════════════════════       │
│                                                                 │
│   🔊 Speaker (emitter)         →    🌀 Storm system (emitter)   │
│   📐 Room geometry             →    🗺️  Basin geometry          │
│   🔉 Other sources             →    🌡️  T, P, other vortices    │
│   🧱 Materials (absorption)    →    💨 Atmosphere properties    │
│   ═══════════════════════════════════════════════════════       │
│   🎵 Predicted sound field     →    🌊 Predicted surge field    │
│                                                                 │
│           TRANSFER FUNCTION / IMPULSE RESPONSE                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The "Song" of the Atmosphere

**Room acoustics predicts:** How a sound (frequency, amplitude) will propagate, reflect, resonate in a space.

**Atmospheric acoustics predicts:** How a pressure perturbation (storm) will propagate, interact, amplify in the atmosphere-ocean system.

The **storm surge IS the music** - the atmospheric "song" that results from:
- Source: cyclone, pressure drop, wind stress
- Medium: atmosphere + ocean
- Receiver: coastline (where we "hear" the surge)

### Mathematical Framework

In room acoustics:
```
Output(f) = Input(f) × H(f)
```
Where H(f) is the **transfer function** of the room.

In atmospheric acoustics:
```
Surge(f) = Storm(f) × H_atm(f) × H_ocean(f) × H_coast(f)
```

Each component has its own **frequency response**:
- H_atm: atmospheric propagation (wind, pressure waves)
- H_ocean: ocean response (barotropic waves, seiches)
- H_coast: coastal geometry (amplification, resonance)

### Shazam Analogy Complete

| Shazam (Audio) | NICO (Atmosphere) |
|----------------|-------------------|
| Microphone captures song | Sensors capture atmospheric state |
| Spectrogram fingerprint | Spectral fingerprint of storm |
| Database of known songs | Database of known storm patterns |
| Match → identify song | Match → predict surge response |
| "This is Bohemian Rhapsody" | "This pattern → 1.5m surge in 12h" |

### Storm Surge as "Geospatial Acoustics"

Storm surge events can be analyzed as **low-frequency pressure waves**:

```
Traditional view:  Sea Level = f(wind, pressure, tide)
Spectral view:     Sea Level = Σ(frequency components with 1/f^β weighting)
Acoustic view:     Sea Level = Convolution(Storm_signal, Basin_impulse_response)
```

### Fingerprinting Application

Just as audio fingerprinting (Shazam) identifies songs by spectral patterns:
- We can fingerprint **storm events** by their spectral signature
- Brown/pink noise patterns could reveal **precursor signals**
- Deviations from expected 1/f^β may indicate **anomalies/extreme events**

### Sicily Tornado Pattern

The observation that Sicily tornado events show brown noise spectral decay:
- Suggests **universal pattern** across atmospheric phenomena
- Could enable **cross-domain learning**: what we know about audio → weather
- Infrasound monitoring could detect tornado formation

---

## Research Questions

1. **Is the Kolmogorov -5/3 law equivalent to pink/brown noise in atmospheric data?**
   - Compare turbulence spectra with audio noise spectra

2. **Can we use audio signal processing for storm prediction?**
   - FFT analysis of pressure time series
   - Spectrogram visualization of storm events
   - Wavelet decomposition for multi-scale analysis

3. **Do extreme events deviate from 1/f^β patterns?**
   - "White noise" bursts = unpredictable extreme events?
   - Changes in β value as precursor signal?

4. **Can human intuition "hear" weather patterns?**
   - Sonification of atmospheric data
   - Perceptual testing with calibrated noise

5. **Can we compute basin "impulse responses"?**
   - Like measuring room acoustics with a clap/sweep
   - Historical storms as "test signals"
   - Build transfer function database per basin

6. **Is the Mediterranean a "resonant cavity" for atmospheric pressure?**
   - Seiches as standing waves (acoustic room modes)
   - Adriatic as organ pipe?
   - Natural frequencies of enclosed basins

---

## Deep Insight: The Organ Pipe Effect

### What Happens When Pressure + Wind Push Water?

```
LOW PRESSURE                           WIND (tailwind)
     ↓                                      →
┌──────────────────────────────────────────────────────┐
│                                                      │
│  ═══════════════════════════════════════════════    │
│     ↑ water rises      →→→ pushed →→→        ↑↑    │
│    (inv. barometer)    (wind setup)       SURGE!   │
│                                                      │
│                                    Venice/Trieste   │
└──────────────────────────────────────────────────────┘
        ADRIATIC = ORGAN PIPE / RESONANT CAVITY
```

### The Physics

1. **Inverse Barometer Effect**
   - ~1 cm rise per 1 hPa pressure drop
   - 30 hPa drop (strong storm) → 30 cm static rise

2. **Wind Setup**
   - Wind stress τ = ρ_air × C_d × U²
   - Pushes water toward downwind coast
   - Shallow water amplifies effect

3. **SEICHE: The Resonance**
   - Water oscillates like in a bathtub
   - Adriatic natural period: ~21-22 hours
   - If storm "plays" this frequency → RESONANCE → AMPLIFICATION

### The Acoustic Analogy

| Organ Pipe | Adriatic Sea |
|------------|--------------|
| Air blown across opening | Wind/pressure forcing |
| Pipe length determines pitch | Basin length determines seiche period |
| Resonance amplifies sound | Resonance amplifies surge |
| Harmonic overtones | Higher seiche modes |
| Standing waves | Standing water waves |

### Formula: Basin Natural Period

```
T = 2L / √(gH)

Where:
- T = natural period (seconds)
- L = basin length (m)
- g = gravity (9.81 m/s²)
- H = average depth (m)

Adriatic example:
L ≈ 800 km = 800,000 m
H ≈ 40 m (average)
T = 2 × 800,000 / √(9.81 × 40)
T ≈ 80,000 seconds ≈ 22 hours ✓
```

### Implication for Prediction

If we know:
- Basin geometry → natural frequencies (like measuring a room's acoustics)
- Storm characteristics → forcing frequencies
- **When forcing frequency ≈ natural frequency → DANGER!**

This is exactly like acoustic resonance destroying a wine glass with the right frequency!

---

## The Vision: NICO as "Atmospheric Acoustician"

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│                    NICO: GEOSPATIAL ACOUSTICS                      │
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │   EMITTERS   │    │    MEDIUM    │    │   RECEIVER   │         │
│  │              │    │              │    │              │         │
│  │ 🌀 Cyclones  │ →  │ 🌊 Atm+Ocean │ →  │ 🏖️ Coastline │         │
│  │ 💨 Wind      │    │              │    │              │         │
│  │ 📉 Pressure  │    │ H(f) = ?     │    │ Surge = ?    │         │
│  └──────────────┘    └──────────────┘    └──────────────┘         │
│         ↓                   ↓                   ↓                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │  FINGERPRINT │    │   TRANSFER   │    │   PREDICT    │         │
│  │              │    │   FUNCTION   │    │              │         │
│  │ Spectral     │  × │ Basin-       │  = │ Expected     │         │
│  │ signature    │    │ specific     │    │ surge        │         │
│  └──────────────┘    └──────────────┘    └──────────────┘         │
│                                                                    │
│  "Every storm plays a song. Every basin is an instrument.          │
│   We learn to read the music before it reaches the shore."         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Existing Research: Tornado Infrasound

**Key Discovery:** Tornadoes emit infrasound that can be detected **up to 1 hour BEFORE formation!**

| Finding | Source |
|---------|--------|
| Tornado infrasound: 0.5-20 Hz (below human hearing) | [Oklahoma State Research](https://news.okstate.edu/articles/engineering-architecture-technology/2022/elbing-receives-experimental-physics-investigator-allowing-continued-research-on-infrasound-signatures-of-tornadoes.html) |
| Characteristic frequency = 2× vortex rotation frequency | [Journal of Acoustical Society](https://pubs.aip.org/asa/jasa/article/156/2/1214/3308426/On-the-vortex-dynamical-contribution-to-the) |
| 100 mbar pressure drop measured in tornado core | [ScienceDaily](https://www.sciencedaily.com/releases/2018/05/180508081511.htm) |
| Hurricane sonification project at Penn State | [Smithsonian](https://www.smithsonianmag.com/innovation/turning-hurricane-data-into-music-180967414/) |
| GLINDA system: detects tornadogenesis via infrasound | [Electronic Design](https://www.electronicdesign.com/technologies/test-measurement/article/55126732/electronic-design-tornadogenesis-identified-with-infrasoundnew-predictions-possible) |

### The Math: Frequency Mapping

Tornado frequency → Human audible range:

```
Tornado rotation:     ~0.5 Hz (one rotation every 2 seconds)
Infrasound emission:  ~1-15 Hz
Human hearing:        20 Hz - 20,000 Hz

Mapping approach:
- Speed up time series 100-1000x
- 1 Hz tornado signal → 100-1000 Hz (audible!)
- Or: pitch-shift while preserving spectral shape
```

### Penn State Hurricane Sonification

Researchers mapped hurricane data to music:
- **Air pressure** → pitch
- **Latitude/longitude** → stereo position
- **Asymmetry** → timbre/texture

> "Our ears are better at sensing properties that change and fluctuate than our eyes.
> The ears are also better than the eyes at following multiple patterns simultaneously."

---

## Experiment: Sonify a Vortex

### Data Requirements

For a good sonification we need:
- [ ] Pressure time series (high temporal resolution)
- [ ] Velocity field (radial/tangential)
- [ ] Duration: at least minutes of data

### Potential Data Sources

| Source | Resolution | Variables | Access |
|--------|------------|-----------|--------|
| ERA5 | Hourly | P, wind | ✅ Have client |
| CMEMS | 6-hourly | Sea level, currents | ✅ Have client |
| Radar (NEXRAD) | ~5 min | Velocity | Need to add |
| Infrasound sensors | Real-time | Pressure | Research data |

### Sonification Algorithm

```python
# Pseudo-code for vortex sonification

def sonify_vortex(pressure_series, velocity_series, time_speedup=500):
    """
    Convert atmospheric data to audible sound.

    Args:
        pressure_series: Pressure time series (Pa)
        velocity_series: Wind velocity (m/s)
        time_speedup: Factor to bring into audible range

    Returns:
        audio: WAV file
    """
    # 1. Normalize data
    p_norm = normalize(pressure_series, -1, 1)
    v_norm = normalize(velocity_series, 0, 1)

    # 2. Resample to audio rate (44100 Hz)
    # Original: 1 sample/hour → need interpolation
    audio_rate = 44100
    original_rate = 1/3600  # hourly
    target_rate = original_rate * time_speedup

    # 3. Map to audio parameters
    # Pressure → base frequency (pitch)
    # Velocity → amplitude (loudness)
    # Pressure derivative → frequency modulation (vibrato)

    base_freq = 220  # A3
    freq = base_freq * (1 + p_norm * 0.5)  # ±50% pitch variation
    amplitude = 0.3 + v_norm * 0.7

    # 4. Synthesize
    audio = synthesize_fm(freq, amplitude, duration)

    return audio
```

### What We Might Hear

| Pattern | Sound | Meaning |
|---------|-------|---------|
| Falling pitch | Storm intensifying (pressure drop) |
| Rising pitch | Storm weakening |
| Increasing volume | Wind increasing |
| Rapid fluctuations | Turbulence, vortex structure |
| Harmonic content | Multi-scale structure |
| Brown noise character | Natural atmospheric state |
| White noise bursts | Extreme/chaotic events |

---

---

## Case Study: Sicily Storm Harry (January 2026)

### The Event

**Date:** 20-21 January 2026
**Location:** Sicily, Southern Italy
**Cause:** Cyclone Harry

| Measurement | Value | Note |
|-------------|-------|------|
| Max wave height | **12+ meters** | Unprecedented for Mediterranean |
| Mazara del Vallo | 8 meters (26 ft) | Historical meteotsunami location |
| Catania | 5+ meters (16 ft) | |
| Damage estimate | **€1+ billion** | Public + private |

> "During the night between 20 and 21 January, off eastern Sicily, waves exceeded 12 metres in height — an unprecedented phenomenon for Italy and the Mediterranean Sea."

### Historical Context: Mazara del Vallo

The Arabs (9th century) named the river "Mazaro" meaning **"possessed"** because of recurring tsunami-like waves! This location is historically prone to **meteotsunamis**.

### The Acoustic Interpretation

```
CYCLONE HARRY = THE EMITTER (Low frequency forcing)
         ↓
    MEDITERRANEAN = THE INSTRUMENT (Resonant basin)
         ↓
    12m WAVES = THE MUSIC (Amplified response)
```

**Questions for analysis:**
- Did Storm Harry's pressure oscillations match Mediterranean resonant frequencies?
- Was there seiche amplification in the Strait of Sicily?
- Can we detect precursor signals in ERA5 data?

### Data to Collect

- [ ] ERA5 MSLP for Jan 18-22, 2026 (Mediterranean domain)
- [ ] Tide gauge records (Catania, Mazara del Vallo, Taormina)
- [ ] Compute power spectrum of pressure field
- [ ] Compare with basin natural frequencies

**Sources:**
- [Euronews: Taormina mayor on €1bn damage](https://www.euronews.com/business/2026/01/23/taormina-mayor-says-urgent-action-needed-after-1bn-storm-damage-in-sicily)
- [Surfer: 26ft waves from Cyclone Harry](https://www.surfer.com/news/italy-big-waves-cyclone-harry-video)
- [Meteotsunami - NOAA](https://oceanservice.noaa.gov/facts/meteotsunami.html)

---

## Next Steps

- [ ] Analyze historical tornado data → compute power spectra
- [ ] Compare with ERA5 pressure time series spectra
- [ ] **Create sonification tool for atmospheric data** ✅ DONE
- [ ] Literature review: Kolmogorov turbulence, atmospheric acoustics
- [ ] **Experiment: Sonify the October 2000 Piedmont flood**
- [ ] **Analyze Storm Harry (Sicily Jan 2026) as case study**
- [ ] Investigate GLINDA-style infrasound detection
- [ ] Contact Penn State team re: hurricane sonification

---

---

## Deep Theory: Self-Organized Criticality (SOC)

### Why is 1/f Noise EVERYWHERE?

> "One of the classic problems in physics is the existence of the ubiquitous 1/f noise which has been detected in systems as diverse as resistors, the hourglass, the flow of the river Nile, and the luminosity of stars."

**Answer:** Self-Organized Criticality (Bak, Tang, Wiesenfeld, 1987)

### What is SOC?

Systems naturally evolve toward a **critical point** - a state of minimal stability where:
- Perturbations cascade through all scales
- No "tuning" required - the system tunes itself
- Results in **scale-invariance** (fractals in space, 1/f in time)

```
SANDPILE MODEL (Original SOC example)
═══════════════════════════════════════

Add grains one by one...
        ·
       ·↓·
      ·····        → System self-organizes to critical slope
     ·······
    ·········      → One grain can trigger avalanche of ANY size
   ···········     → Power-law distribution of avalanche sizes
  ·············    → 1/f temporal fluctuations

"The pile tunes itself to the edge of stability"
```

### Where 1/f Noise Appears

| Domain | Example | Reference |
|--------|---------|-----------|
| **Physics** | Current in resistors, quasar luminosity | Classic |
| **Geophysics** | River flow (Nile), earthquakes | Bak et al. |
| **Biology** | Heartbeat rhythms, neural firing | PMC |
| **Music** | Melodic structure across cultures | Voss & Clarke |
| **Brain** | EEG, MEG signals | Quanta Magazine |
| **Human behavior** | Reaction times, temporal production | Gilden (1995) |
| **Atmosphere** | Wind turbulence, pressure fluctuations | Kolmogorov |

### The Profound Implication

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   NATURE operates at Self-Organized Criticality                 │
│          ↓                                                      │
│   Everything exhibits 1/f^β spectra                            │
│          ↓                                                      │
│   HUMANS evolved IN this environment                            │
│          ↓                                                      │
│   Our BRAINS exhibit 1/f^β activity                            │
│          ↓                                                      │
│   Our PERCEPTION is calibrated to 1/f^β                        │
│          ↓                                                      │
│   We experience 1/f^β as COMFORTABLE/NATURAL                   │
│          ↓                                                      │
│   Engineers discover "sound masking" = 1/f^β                   │
│          ↓                                                      │
│   THEY REDISCOVERED WHAT NATURE ALREADY KNEW!                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Evidence for Evolved Perception

1. **Music follows 1/f patterns**
   - Voss & Clarke found that melodic structure across ALL cultures tends toward 1/f
   - "Pink noise music" sounds natural, white noise music sounds random

2. **Human response times are 1/f**
   - Gilden (1995): reaction times form pure 1/f time series
   - Our cognitive processes ARE 1/f processes

3. **Brain activity is 1/f**
   - EEG/MEG show 1/f background "noise"
   - This "noise" may be the signature of critical brain dynamics

4. **Natural sound recognition**
   - [PMC Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC10219008/): Human amplitude modulation sensitivity matches patterns in natural sounds
   - Evolution optimized our hearing FOR the 1/f world

### Connection to Atmospheric Acoustics

If **everything** operates at SOC:
- Atmosphere → 1/f^β pressure fluctuations (Kolmogorov turbulence)
- Ocean → 1/f^β wave spectra
- Storm systems → 1/f^β energy cascade
- **We can apply the same signal processing everywhere!**

### The Sandpile ↔ Storm Surge Analogy

| Sandpile | Storm Surge |
|----------|-------------|
| Add sand grains | Add atmospheric forcing |
| Pile self-organizes to critical slope | Ocean basin reaches equilibrium |
| One grain → avalanche of any size | One storm → surge depends on resonance |
| Power-law avalanche distribution | Power-law extreme event distribution |
| 1/f temporal correlations | 1/f sea level fluctuations |

**Implication:** Extreme events (like Sicily 12m waves) may be intrinsic to the SOC dynamics of the atmosphere-ocean system, not "anomalies" but expected tail events of a power-law distribution!

---

## Key Insight: We ARE the Noise

> "We evolved in 1/f noise. Our brains operate with 1/f dynamics.
> When we hear 1/f noise, we hear ourselves. That's why it's comfortable."

This explains:
- Why brown/pink noise helps people sleep
- Why "sound masking" works
- Why nature sounds are calming
- Why we can potentially "hear" storm patterns - we're made of the same statistics!

---

## References & Sources

### Noise & Acoustics
- [Colors of Noise - Wikipedia](https://en.wikipedia.org/wiki/Colors_of_noise)
- [Noise Modeling - Engineering LibreTexts](https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Chemical_Process_Dynamics_and_Controls_(Woolf)/02:_Modeling_Basics/2.05:_Noise_modeling-_more_detailed_information_on_noise_modeling-_white_pink_and_brown_noise_pops_and_crackles)
- [Pink Noise - Wikipedia](https://en.wikipedia.org/wiki/Pink_noise)

### Self-Organized Criticality
- Bak, P., Tang, C. & Wiesenfeld, K. (1987) [Self-organized criticality: An explanation of 1/f noise](https://link.aps.org/doi/10.1103/PhysRevLett.59.381). Phys. Rev. Lett. 59, 381-384
- [Self-organized criticality - Wikipedia](https://en.wikipedia.org/wiki/Self-organized_criticality)
- [1/f noise - Scholarpedia](http://www.scholarpedia.org/article/1/f_noise)

### Evolution & Perception
- [How Sound Shaped The Evolution Of Your Brain - NPR](https://www.npr.org/sections/health-shots/2015/09/10/436342537/how-sound-shaped-the-evolution-of-your-brain)
- [Human-Like Modulation Sensitivity - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10219008/)
- [Brain's 'Background Noise' - Quanta Magazine](https://www.quantamagazine.org/brains-background-noise-may-hold-clues-to-persistent-mysteries-20210208/)

### Turbulence
- Kolmogorov, A. N. (1941) - Turbulence theory
- Original intuition from tornado analysis with brother (2026)

---

## Visual Concept

```
                    THE UNIVERSAL 1/f PATTERN

    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │   TORNADO VORTEX        BROWN NOISE             │
    │   Power                 Power                   │
    │    │\                    │\                     │
    │    │ \                   │ \                    │
    │    │  \                  │  \                   │
    │    └───── Freq           └───── Freq           │
    │                                                 │
    │              SAME PATTERN                       │
    │                   ↓                             │
    │                                                 │
    │         HUMAN EAR SENSITIVITY                   │
    │    Sensitivity                                  │
    │    │\                                           │
    │    │ \                                          │
    │    │  ·····                                     │
    │    └─────────── Freq                           │
    │                                                 │
    │    "Nature and humans share the same           │
    │     sensibility - we evolved together"         │
    │                                                 │
    └─────────────────────────────────────────────────┘
```

---

*"Sound is pressure waves. Weather is pressure waves. We evolved to perceive one; perhaps we can learn to perceive the other."*
