"""
Variable Resolver - Accesso unificato ai dati multi-formato

Questo modulo fornisce un'interfaccia unificata per accedere a variabili
in dataset con naming conventions diverse (SLCCI, CMEMS, AVISO, etc.).

Pattern: Adapter + Strategy (simile al routing audio nel DSP)

Esempio:
    resolver = VariableResolver.from_dataset(ds)  # Auto-detect formato
    ssh = resolver.get("ssh")  # Funziona con qualsiasi formato
    lat, lon = resolver.get_coordinates()
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import (
    DATASET_CONFIGS,
    DatasetConfig,
    DatasetFormat,
    get_dataset_config,
    VariableMapping,
)

if TYPE_CHECKING:
    import xarray as xr


class VariableResolver:
    """
    Resolver per accesso unificato a variabili multi-dataset.
    
    Questa classe agisce come un "router" che mappa nomi canonici
    (ssh, mss, dot, etc.) ai nomi specifici del dataset corrente.
    
    Attributes:
        dataset: xarray.Dataset attivo
        config: Configurazione del formato rilevato
    """
    
    def __init__(self, dataset: xr.Dataset, config: DatasetConfig):
        """
        Inizializza il resolver.
        
        Args:
            dataset: xarray.Dataset da interrogare
            config: DatasetConfig con la mappatura variabili
        """
        self.dataset = dataset
        self.config = config
        self._cache: dict[str, np.ndarray] = {}
    
    @classmethod
    def from_dataset(cls, dataset: xr.Dataset, format_hint: str | None = None) -> VariableResolver:
        """
        Factory method con auto-detection del formato.
        
        Args:
            dataset: xarray.Dataset da analizzare
            format_hint: Hint opzionale per il formato
            
        Returns:
            VariableResolver configurato per il dataset
        """
        if format_hint:
            config = get_dataset_config(format_hint)
        else:
            config = cls._detect_format(dataset)
        
        return cls(dataset, config)
    
    @classmethod
    def from_file(cls, filepath: str | Path) -> VariableResolver:
        """
        Crea resolver da file NetCDF con auto-detection.
        
        Args:
            filepath: Percorso al file NetCDF
            
        Returns:
            VariableResolver configurato
        """
        import xarray as xr
        
        filepath = Path(filepath)
        ds = xr.open_dataset(filepath)
        
        # Usa il nome file come hint
        format_hint = cls._detect_format_from_filename(filepath.name)
        
        return cls.from_dataset(ds, format_hint)
    
    @staticmethod
    def _detect_format(dataset: xr.Dataset) -> DatasetConfig:
        """
        Rileva automaticamente il formato del dataset.
        
        Strategia:
        1. Controlla variabili chiave (corssh → SLCCI, sla → CMEMS)
        2. Controlla attributi globali
        3. Fallback a SLCCI_J1
        """
        var_names = set(dataset.data_vars.keys())
        
        # SLCCI detection
        if "corssh" in var_names:
            return get_dataset_config("slcci_j1")
        
        # CMEMS L3 detection
        if "sla_filtered" in var_names or "sla_unfiltered" in var_names:
            return get_dataset_config("cmems_l3")
        
        # CMEMS L4 detection (gridded)
        if "ugos" in var_names or "vgos" in var_names:
            return get_dataset_config("cmems_l4")
        
        # AVISO detection
        if "sla" in var_names and "lat" in dataset.coords:
            return get_dataset_config("aviso")
        
        # Controlla attributi globali
        attrs = dataset.attrs
        source = attrs.get("source", "").lower()
        institution = attrs.get("institution", "").lower()
        
        if "cmems" in source or "copernicus" in source:
            return get_dataset_config("cmems_l3")
        if "aviso" in source or "duacs" in source:
            return get_dataset_config("aviso")
        if "cci" in source or "slcci" in source:
            return get_dataset_config("slcci_j1")
        
        # Default fallback
        return get_dataset_config("slcci_j1")
    
    @staticmethod
    def _detect_format_from_filename(filename: str) -> str | None:
        """Rileva formato dal nome file."""
        filename_lower = filename.lower()
        
        if "slcci" in filename_lower:
            if "_j1_" in filename_lower:
                return "slcci_j1"
            elif "_j2_" in filename_lower:
                return "slcci_j2"
            return "slcci_j1"
        
        if "cmems" in filename_lower:
            if "l3" in filename_lower:
                return "cmems_l3"
            elif "l4" in filename_lower:
                return "cmems_l4"
            return "cmems_l3"
        
        if "aviso" in filename_lower or "duacs" in filename_lower:
            return "aviso"
        
        return None
    
    def get(self, canonical_name: str, as_array: bool = True) -> np.ndarray | xr.DataArray:
        """
        Ottieni una variabile usando il nome canonico.
        
        Args:
            canonical_name: Nome standard (ssh, mss, swh, etc.)
            as_array: Se True ritorna numpy array, altrimenti DataArray
            
        Returns:
            Dati della variabile
            
        Raises:
            KeyError: Se la variabile non esiste nel dataset
        """
        # Check cache
        if canonical_name in self._cache and as_array:
            return self._cache[canonical_name]
        
        # Ottieni nome nel dataset
        dataset_name = self.config.get_variable(canonical_name)
        
        if dataset_name is None:
            # Prova accesso diretto (per variabili non mappate)
            if canonical_name in self.dataset.data_vars:
                dataset_name = canonical_name
            else:
                available = self.list_available()
                raise KeyError(
                    f"Variabile '{canonical_name}' non trovata. "
                    f"Disponibili: {available}"
                )
        
        # Verifica esistenza nel dataset
        if dataset_name not in self.dataset.data_vars:
            raise KeyError(
                f"Variabile '{dataset_name}' (mappata da '{canonical_name}') "
                f"non presente nel dataset"
            )
        
        data = self.dataset[dataset_name]
        
        if as_array:
            arr = data.values
            self._cache[canonical_name] = arr
            return arr
        
        return data
    
    def get_with_metadata(self, canonical_name: str) -> tuple[np.ndarray, VariableMapping]:
        """
        Ottieni variabile con i suoi metadati.
        
        Returns:
            Tuple di (dati, mapping con metadati)
        """
        data = self.get(canonical_name)
        mapping = self.config.get_mapping(canonical_name)
        return data, mapping
    
    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Ottieni coordinate (lat, lon) usando la mappatura corretta.
        
        Returns:
            Tuple di (latitude, longitude) arrays
        """
        lat_name = self.config.get_coordinate("latitude")
        lon_name = self.config.get_coordinate("longitude")
        
        # Cerca con alias se non trovato
        lat = self._get_coord_with_fallback(lat_name, self.config.coordinates.lat_aliases)
        lon = self._get_coord_with_fallback(lon_name, self.config.coordinates.lon_aliases)
        
        return lat, lon
    
    def get_time(self) -> np.ndarray:
        """Ottieni coordinata temporale."""
        time_name = self.config.get_coordinate("time")
        return self._get_coord_with_fallback(time_name, self.config.coordinates.time_aliases)
    
    def _get_coord_with_fallback(self, primary: str, aliases: tuple[str, ...]) -> np.ndarray:
        """Cerca coordinata con fallback su alias."""
        # Cerca in coords
        if primary in self.dataset.coords:
            return self.dataset[primary].values
        
        # Cerca in data_vars (alcuni dataset hanno coordinate come variabili)
        if primary in self.dataset.data_vars:
            return self.dataset[primary].values
        
        # Prova alias
        for alias in aliases:
            if alias in self.dataset.coords:
                return self.dataset[alias].values
            if alias in self.dataset.data_vars:
                return self.dataset[alias].values
        
        raise KeyError(f"Coordinata '{primary}' non trovata (alias provati: {aliases})")
    
    def has_variable(self, canonical_name: str) -> bool:
        """Verifica se una variabile è disponibile."""
        dataset_name = self.config.get_variable(canonical_name)
        if dataset_name is None:
            return canonical_name in self.dataset.data_vars
        return dataset_name in self.dataset.data_vars
    
    def list_available(self) -> list[str]:
        """Lista variabili disponibili (nomi canonici)."""
        available = []
        for canonical in self.config.list_variables():
            if self.has_variable(canonical):
                available.append(canonical)
        return available
    
    def list_raw_variables(self) -> list[str]:
        """Lista tutte le variabili nel dataset (nomi originali)."""
        return list(self.dataset.data_vars.keys())
    
    def compute_dot(self, reference: str = "mss") -> np.ndarray:
        """
        Calcola DOT (Dynamic Ocean Topography).
        
        DOT = SSH - Reference Surface
        
        Args:
            reference: "mss" (Mean Sea Surface) o "mdt" (Mean Dynamic Topography)
            
        Returns:
            DOT values
        """
        ssh = self.get("ssh")
        ref = self.get(reference)
        return ssh - ref
    
    def get_quality_mask(self) -> np.ndarray | None:
        """
        Ottieni maschera di qualità.
        
        Returns:
            Boolean mask (True = valido) o None se non disponibile
        """
        flag_var = self.config.quality_flag_var
        if flag_var is None or flag_var not in self.dataset.data_vars:
            return None
        
        flags = self.dataset[flag_var].values
        valid_values = self.config.quality_flag_valid
        
        if isinstance(valid_values, int):
            return flags == valid_values
        else:
            return np.isin(flags, valid_values)
    
    @property
    def format_name(self) -> str:
        """Nome del formato rilevato."""
        return self.config.name
    
    @property
    def format_type(self) -> DatasetFormat:
        """Tipo del formato rilevato."""
        return self.config.format
    
    def __repr__(self) -> str:
        return f"VariableResolver(format={self.config.name}, vars={len(self.list_available())})"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def auto_load(filepath: str | Path) -> tuple[xr.Dataset, VariableResolver]:
    """
    Carica dataset e crea resolver automaticamente.
    
    Args:
        filepath: Percorso al file NetCDF
        
    Returns:
        Tuple di (dataset, resolver)
    """
    import xarray as xr
    
    ds = xr.open_dataset(filepath)
    resolver = VariableResolver.from_dataset(ds)
    
    return ds, resolver


def compare_formats() -> None:
    """Stampa tabella comparativa dei formati supportati."""
    print("\n" + "=" * 80)
    print("FORMATI DATASET SUPPORTATI")
    print("=" * 80)
    
    for name, config in DATASET_CONFIGS.items():
        print(f"\n{config.name} ({name})")
        print("-" * 40)
        print(f"  Descrizione: {config.description}")
        print(f"  Epoca riferimento: {config.reference_epoch}")
        print(f"  Convenzione lon: {config.longitude_convention}")
        print(f"  Variabili:")
        for canonical, mapping in config.variables.items():
            print(f"    {canonical:12} → {mapping.dataset_name:20} [{mapping.units}]")
