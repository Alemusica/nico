"""CYGNSS Wind Speed Client - NASA PO.DAAC (2-24h latency)."""
import os
from datetime import datetime
from typing import Tuple, List, Optional
import xarray as xr
import numpy as np

try:
    import earthaccess
    HAS_EARTHACCESS = True
except ImportError:
    HAS_EARTHACCESS = False

CYGNSS_L3 = "CYGNSS_L3_GLOBAL_DAILY_V3.1"


class CYGNSSClient:
    def __init__(self):
        if not HAS_EARTHACCESS:
            raise ImportError("pip install earthaccess")
        self.auth = earthaccess.login(strategy="environment")
    
    def search_granules(
        self,
        time_range: Tuple[datetime, datetime],
        bbox: Tuple[float, float, float, float] = None,
        max_results: int = 100,
    ) -> List:
        return earthaccess.search_data(
            short_name=CYGNSS_L3,
            temporal=time_range,
            bounding_box=bbox,
            count=max_results,
        )
    
    def download(
        self,
        time_range: Tuple[datetime, datetime] = None,
        bbox: Tuple[float, float, float, float] = None,
        variables: List[str] = None,
    ) -> xr.Dataset:
        granules = self.search_granules(time_range, bbox)
        
        if not granules:
            return xr.Dataset(attrs={"note": "No data found"})
        
        files = earthaccess.download(granules, local_path="/tmp/cygnss")
        ds = xr.open_mfdataset(files, combine="by_coords")
        
        if variables:
            ds = ds[variables]
        
        ds.attrs["source"] = "CYGNSS"
        ds.attrs["latency"] = "2-24h"
        return ds


def load(time_range=None, bbox=None, variables=None) -> xr.Dataset:
    """Called by CatalogLoader."""
    return CYGNSSClient().download(time_range, bbox, variables)

Client = CYGNSSClient
