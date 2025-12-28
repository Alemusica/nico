"""
IntakeCatalogBridge - Bridge tra Intake YAML e client esistenti.

NON sostituisce src/data_manager/catalog.py (CopernicusCatalog) - lo usa!
"""
import intake
from pathlib import Path
from typing import List, Dict, Any, Optional
import xarray as xr
import importlib

# Paths
ROOT = Path(__file__).parent.parent.parent
CATALOG_PATH = ROOT / "catalog.yaml"


class IntakeCatalogBridge:
    """
    Bridge pattern: Intake YAML come index unificato,
    delega ai client esistenti per il download effettivo.
    
    Aggiunge:
    - Latency metadata (🟢, 🟡, 🔴)
    - Multi-provider (ERA5, NOAA, NASA, oltre a CMEMS)
    - Ricerca unificata
    """
    
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        self.catalog_path = catalog_path
        self._catalog = None
        self._client_cache = {}
    
    @property
    def catalog(self):
        """Lazy load catalog."""
        if self._catalog is None:
            self._catalog = intake.open_catalog(str(self.catalog_path))
        return self._catalog
    
    def list_datasets(self) -> List[str]:
        """Lista tutti i dataset disponibili."""
        return list(self.catalog)
    
    def get_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Ottieni metadata da catalog (latency, variables, etc.)."""
        entry = self.catalog._entries.get(dataset_id)
        if entry is None:
            raise KeyError(f"Dataset '{dataset_id}' not found")
        return dict(entry._metadata)
    
    def search(
        self,
        variables: List[str] = None,
        latency_badge: str = None,
        status: str = None,
        provider: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Cerca dataset per criteri.
        
        Args:
            variables: Lista variabili richieste (es. ["sla", "sst"])
            latency_badge: Filtro latency ("🟢", "🟡", "🔴", "⚫")
            status: Filtro status ("available", "to_implement")
            provider: Filtro provider ("Copernicus Marine", "ECMWF", etc.)
        
        Returns:
            Lista di dict con id e metadata per ogni match
        """
        results = []
        
        for name in self.catalog._entries:
            meta = self.get_metadata(name)
            
            # Filter by status
            if status and meta.get("status") != status:
                continue
            
            # Filter by latency
            if latency_badge and meta.get("latency_badge") != latency_badge:
                continue
            
            # Filter by provider
            if provider and meta.get("provider") != provider:
                continue
            
            # Filter by variables
            if variables:
                ds_vars = meta.get("variables", [])
                if not any(v in ds_vars for v in variables):
                    continue
            
            results.append({
                "id": name,
                **meta
            })
        
        return results
    
    def search_by_latency(self, max_latency: str = "🟡") -> List[Dict[str, Any]]:
        """
        Cerca dataset con latency <= max_latency.
        
        Ordine: 🟢 (real-time) < 🟡 (daily) < 🔴 (weekly+) < ⚫ (historical)
        """
        latency_order = {"🟢": 1, "🟡": 2, "🔴": 3, "⚫": 4, "⏳": 5}
        max_level = latency_order.get(max_latency, 5)
        
        results = []
        for name in self.catalog._entries:
            meta = self.get_metadata(name)
            badge = meta.get("latency_badge", "⏳")
            if latency_order.get(badge, 5) <= max_level:
                results.append({"id": name, **meta})
        
        return sorted(results, key=lambda x: latency_order.get(x.get("latency_badge", "⏳"), 5))
    
    def get_client(self, dataset_id: str):
        """
        Ottieni il client appropriato per un dataset.
        
        Returns:
            Istanza del client o None se non disponibile
        """
        if dataset_id in self._client_cache:
            return self._client_cache[dataset_id]
        
        meta = self.get_metadata(dataset_id)
        client_path = meta.get("client")
        
        if not client_path:
            return None
        
        try:
            # Import module
            module_path, class_name = client_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            client_class = getattr(module, class_name)
            
            # Instantiate
            client = client_class()
            self._client_cache[dataset_id] = client
            return client
            
        except (ImportError, AttributeError) as e:
            print(f"⚠️ Cannot load client for {dataset_id}: {e}")
            return None
    
    async def load(
        self,
        dataset_id: str,
        time_range: tuple = None,
        bbox: tuple = None,
        variables: List[str] = None,
    ) -> xr.Dataset:
        """
        Carica dataset usando il client appropriato.
        
        Args:
            dataset_id: ID dataset dal catalog
            time_range: (start, end) come stringhe ISO
            bbox: (lat_min, lat_max, lon_min, lon_max)
            variables: Lista variabili da caricare
        
        Returns:
            xarray.Dataset
        """
        meta = self.get_metadata(dataset_id)
        client = self.get_client(dataset_id)
        
        if client is not None:
            # Usa client esistente
            if hasattr(client, "download"):
                return await client.download(
                    time_range=time_range,
                    bbox=bbox,
                    variables=variables,
                )
            elif hasattr(client, "get_download_config"):
                # CopernicusCatalog pattern
                config = client.get_download_config(
                    product_id=meta.get("product_id"),
                    lat_range=(bbox[0], bbox[1]) if bbox else None,
                    lon_range=(bbox[2], bbox[3]) if bbox else None,
                    time_range=time_range,
                    variables=variables,
                )
                # TODO: actual download
                return xr.Dataset(attrs={"config": config})
        
        # Fallback: carica direttamente da Intake
        source = self.catalog[dataset_id]
        return source.to_dask()
    
    def summary(self) -> Dict[str, Any]:
        """Riepilogo del catalog."""
        datasets = self.list_datasets()
        
        by_status = {}
        by_latency = {}
        by_provider = {}
        
        for name in datasets:
            meta = self.get_metadata(name)
            
            status = meta.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            
            badge = meta.get("latency_badge", "N/A")
            by_latency[badge] = by_latency.get(badge, 0) + 1
            
            provider = meta.get("provider", "unknown")
            by_provider[provider] = by_provider.get(provider, 0) + 1
        
        return {
            "total_datasets": len(datasets),
            "by_status": by_status,
            "by_latency": by_latency,
            "by_provider": by_provider,
        }


# Singleton instance
_bridge = None


def get_catalog() -> IntakeCatalogBridge:
    """Get singleton catalog bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = IntakeCatalogBridge()
    return _bridge


# CLI test
if __name__ == "__main__":
    bridge = IntakeCatalogBridge()
    
    print("=== Intake Catalog Bridge ===\n")
    
    print("Summary:")
    summary = bridge.summary()
    print(f"  Total datasets: {summary['total_datasets']}")
    print(f"  By status: {summary['by_status']}")
    print(f"  By latency: {summary['by_latency']}")
    print(f"  By provider: {summary['by_provider']}")
    
    print("\nReal-time datasets (🟢):")
    for ds in bridge.search(latency_badge="🟢"):
        print(f"  - {ds['id']}: {ds.get('latency')}")
    
    print("\nAvailable datasets with SLA variable:")
    for ds in bridge.search(variables=["sla"], status="available"):
        print(f"  - {ds['id']}")
