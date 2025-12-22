"""
Dataset Configuration - Variable Mapping System

Sistema centralizzato per la gestione delle variabili attraverso diversi formati di dati
altimetrici (SLCCI, CMEMS, etc.). Simile al pattern "alias" usato nell'audio DSP.

Questo modulo permette di:
1. Definire alias per variabili con nomi diversi in dataset diversi
2. Rilevare automaticamente il formato del dataset
3. Accedere ai dati con nomi standardizzati

Esempio:
    config = get_dataset_config("cmems")
    ssh_var = config.get_variable("ssh")  # Ritorna "sla" per CMEMS, "corssh" per SLCCI
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class DatasetFormat(Enum):
    """Formati dataset supportati."""
    SLCCI_J1 = "slcci_j1"      # Sea Level CCI - Jason-1
    SLCCI_J2 = "slcci_j2"      # Sea Level CCI - Jason-2
    CMEMS_L3 = "cmems_l3"      # Copernicus Marine - Level 3
    CMEMS_L4 = "cmems_l4"      # Copernicus Marine - Level 4
    AVISO = "aviso"            # AVISO/DUACS
    CUSTOM = "custom"          # User-defined


@dataclass
class VariableMapping:
    """
    Mappatura di una singola variabile con alias e metadati.
    
    Attributes:
        canonical_name: Nome standard interno (es. "ssh")
        dataset_name: Nome nel dataset specifico (es. "corssh", "sla")
        units: Unità di misura
        description: Descrizione della variabile
        scale_factor: Fattore di scala per conversione (default 1.0)
        add_offset: Offset per conversione (default 0.0)
        valid_range: Range valido (min, max) o None
    """
    canonical_name: str
    dataset_name: str
    units: str = ""
    description: str = ""
    scale_factor: float = 1.0
    add_offset: float = 0.0
    valid_range: tuple[float, float] | None = None


@dataclass
class CoordinateMapping:
    """Mappatura delle coordinate spaziali/temporali."""
    latitude: str = "latitude"
    longitude: str = "longitude"
    time: str = "time"
    # Coordinate alternative
    lat_aliases: tuple[str, ...] = ("lat", "LAT", "Latitude", "LATITUDE")
    lon_aliases: tuple[str, ...] = ("lon", "LON", "Longitude", "LONGITUDE", "long")
    time_aliases: tuple[str, ...] = ("TIME", "Time", "datetime", "date")


@dataclass
class DatasetConfig:
    """
    Configurazione completa per un formato di dataset.
    
    Questa classe centralizza tutte le informazioni necessarie per
    lavorare con un tipo specifico di dataset altimetrico.
    """
    format: DatasetFormat
    name: str
    description: str
    variables: dict[str, VariableMapping] = field(default_factory=dict)
    coordinates: CoordinateMapping = field(default_factory=CoordinateMapping)
    
    # Metadati addizionali
    reference_epoch: str = "2000-01-01"  # Epoca di riferimento temporale
    longitude_convention: str = "0_360"   # "0_360" o "-180_180"
    quality_flag_var: str | None = None   # Nome variabile flag qualità
    quality_flag_valid: int | tuple[int, ...] = 0  # Valori flag validi
    
    def get_variable(self, canonical_name: str) -> str | None:
        """
        Ottieni il nome della variabile nel dataset dato il nome canonico.
        
        Args:
            canonical_name: Nome standard (es. "ssh", "mss", "dot")
            
        Returns:
            Nome variabile nel dataset o None se non trovata
        """
        if canonical_name in self.variables:
            return self.variables[canonical_name].dataset_name
        return None
    
    def get_mapping(self, canonical_name: str) -> VariableMapping | None:
        """Ottieni il mapping completo per una variabile."""
        return self.variables.get(canonical_name)
    
    def list_variables(self) -> list[str]:
        """Lista tutti i nomi canonici delle variabili disponibili."""
        return list(self.variables.keys())
    
    def get_coordinate(self, coord_type: str) -> str:
        """
        Ottieni il nome della coordinata.
        
        Args:
            coord_type: "latitude", "longitude", o "time"
        """
        return getattr(self.coordinates, coord_type, coord_type)


# =============================================================================
# CONFIGURAZIONI PREDEFINITE PER DATASET
# =============================================================================

SLCCI_J1_CONFIG = DatasetConfig(
    format=DatasetFormat.SLCCI_J1,
    name="SLCCI Jason-1",
    description="Sea Level CCI Altimeter Database - Jason-1 Mission",
    variables={
        # Sea Surface Height
        "ssh": VariableMapping(
            canonical_name="ssh",
            dataset_name="corssh",
            units="m",
            description="Corrected Sea Surface Height",
        ),
        # Mean Sea Surface
        "mss": VariableMapping(
            canonical_name="mss",
            dataset_name="mean_sea_surface",
            units="m",
            description="Mean Sea Surface Height",
        ),
        # Significant Wave Height
        "swh": VariableMapping(
            canonical_name="swh",
            dataset_name="swh",
            units="m",
            description="Significant Wave Height",
        ),
        # Wind Speed
        "wind": VariableMapping(
            canonical_name="wind",
            dataset_name="wind_speed_alt",
            units="m/s",
            description="Altimeter-derived Wind Speed",
        ),
        # Bathymetry
        "bathymetry": VariableMapping(
            canonical_name="bathymetry",
            dataset_name="bathymetry",
            units="m",
            description="Ocean Depth (negative values)",
        ),
        # Backscatter
        "sigma0": VariableMapping(
            canonical_name="sigma0",
            dataset_name="sigma0",
            units="dB",
            description="Radar Backscatter Coefficient",
        ),
        # Ionospheric Correction
        "iono": VariableMapping(
            canonical_name="iono",
            dataset_name="iono_corr",
            units="m",
            description="Ionospheric Correction",
        ),
        # Dry Tropospheric Correction
        "dry_tropo": VariableMapping(
            canonical_name="dry_tropo",
            dataset_name="dry_tropo_corr",
            units="m",
            description="Dry Tropospheric Correction",
        ),
        # Wet Tropospheric Correction
        "wet_tropo": VariableMapping(
            canonical_name="wet_tropo",
            dataset_name="wet_tropo_corr",
            units="m",
            description="Wet Tropospheric Correction",
        ),
    },
    coordinates=CoordinateMapping(
        latitude="latitude",
        longitude="longitude",
        time="TimeDay",
    ),
    reference_epoch="2000-01-01",
    longitude_convention="0_360",
    quality_flag_var="validation_flag",
    quality_flag_valid=0,
)


SLCCI_J2_CONFIG = DatasetConfig(
    format=DatasetFormat.SLCCI_J2,
    name="SLCCI Jason-2",
    description="Sea Level CCI Altimeter Database - Jason-2 Mission",
    variables=SLCCI_J1_CONFIG.variables.copy(),  # Stesse variabili di J1
    coordinates=SLCCI_J1_CONFIG.coordinates,
    reference_epoch="2000-01-01",
    longitude_convention="0_360",
    quality_flag_var="validation_flag",
    quality_flag_valid=0,
)


CMEMS_L3_CONFIG = DatasetConfig(
    format=DatasetFormat.CMEMS_L3,
    name="CMEMS Level-3",
    description="Copernicus Marine Service - Along-Track L3 Products",
    variables={
        # Sea Level Anomaly (principale variabile CMEMS)
        "ssh": VariableMapping(
            canonical_name="ssh",
            dataset_name="sla_filtered",  # o "sla_unfiltered"
            units="m",
            description="Sea Level Anomaly (filtered)",
        ),
        # ADT - Absolute Dynamic Topography
        "adt": VariableMapping(
            canonical_name="adt",
            dataset_name="adt",
            units="m",
            description="Absolute Dynamic Topography",
        ),
        # MDT - Mean Dynamic Topography
        "mdt": VariableMapping(
            canonical_name="mdt",
            dataset_name="mdt",
            units="m",
            description="Mean Dynamic Topography",
        ),
        # SLA unfiltered
        "sla": VariableMapping(
            canonical_name="sla",
            dataset_name="sla_unfiltered",
            units="m",
            description="Sea Level Anomaly (unfiltered)",
        ),
        # DAC - Dynamic Atmospheric Correction
        "dac": VariableMapping(
            canonical_name="dac",
            dataset_name="dac",
            units="m",
            description="Dynamic Atmospheric Correction",
        ),
        # Ocean Tide
        "ocean_tide": VariableMapping(
            canonical_name="ocean_tide",
            dataset_name="ocean_tide",
            units="m",
            description="Ocean Tide Height",
        ),
        # LWE - Long Wavelength Error
        "lwe": VariableMapping(
            canonical_name="lwe",
            dataset_name="lwe",
            units="m",
            description="Long Wavelength Error",
        ),
    },
    coordinates=CoordinateMapping(
        latitude="latitude",
        longitude="longitude",
        time="time",
    ),
    reference_epoch="1950-01-01",  # CMEMS usa epoca diversa
    longitude_convention="-180_180",
    quality_flag_var=None,  # CMEMS L3 non ha flag esplicito
    quality_flag_valid=0,
)


CMEMS_L4_CONFIG = DatasetConfig(
    format=DatasetFormat.CMEMS_L4,
    name="CMEMS Level-4",
    description="Copernicus Marine Service - Gridded L4 Products",
    variables={
        "ssh": VariableMapping(
            canonical_name="ssh",
            dataset_name="sla",
            units="m",
            description="Sea Level Anomaly (gridded)",
        ),
        "adt": VariableMapping(
            canonical_name="adt",
            dataset_name="adt",
            units="m",
            description="Absolute Dynamic Topography",
        ),
        "ugos": VariableMapping(
            canonical_name="ugos",
            dataset_name="ugos",
            units="m/s",
            description="Geostrophic Velocity (U component)",
        ),
        "vgos": VariableMapping(
            canonical_name="vgos",
            dataset_name="vgos",
            units="m/s",
            description="Geostrophic Velocity (V component)",
        ),
    },
    coordinates=CoordinateMapping(
        latitude="latitude",
        longitude="longitude",
        time="time",
    ),
    reference_epoch="1950-01-01",
    longitude_convention="-180_180",
)


AVISO_CONFIG = DatasetConfig(
    format=DatasetFormat.AVISO,
    name="AVISO/DUACS",
    description="AVISO+ Delayed-Time Products",
    variables={
        "ssh": VariableMapping(
            canonical_name="ssh",
            dataset_name="sla",
            units="m",
            description="Sea Level Anomaly",
        ),
        "adt": VariableMapping(
            canonical_name="adt",
            dataset_name="adt",
            units="m",
            description="Absolute Dynamic Topography",
        ),
    },
    coordinates=CoordinateMapping(
        latitude="lat",
        longitude="lon",
        time="time",
    ),
    longitude_convention="-180_180",
)


# =============================================================================
# REGISTRY GLOBALE DELLE CONFIGURAZIONI
# =============================================================================

DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "slcci_j1": SLCCI_J1_CONFIG,
    "slcci_j2": SLCCI_J2_CONFIG,
    "cmems_l3": CMEMS_L3_CONFIG,
    "cmems_l4": CMEMS_L4_CONFIG,
    "aviso": AVISO_CONFIG,
}


def get_dataset_config(format_name: str) -> DatasetConfig:
    """
    Ottieni la configurazione per un formato di dataset.
    
    Args:
        format_name: Nome del formato ("slcci_j1", "cmems_l3", etc.)
        
    Returns:
        DatasetConfig per il formato richiesto
        
    Raises:
        KeyError: Se il formato non è supportato
    """
    format_lower = format_name.lower()
    if format_lower not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise KeyError(f"Formato '{format_name}' non supportato. Disponibili: {available}")
    return DATASET_CONFIGS[format_lower]


def register_dataset_config(name: str, config: DatasetConfig) -> None:
    """
    Registra una nuova configurazione dataset.
    
    Utile per aggiungere formati custom a runtime.
    
    Args:
        name: Nome identificativo (es. "my_custom_format")
        config: DatasetConfig con la configurazione
    """
    DATASET_CONFIGS[name.lower()] = config


def list_supported_formats() -> list[str]:
    """Lista tutti i formati di dataset supportati."""
    return list(DATASET_CONFIGS.keys())
