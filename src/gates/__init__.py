"""
Gates Module
============
Module for handling ocean gate definitions, shapefiles, and spatial operations.

This module provides:
- GateCatalog: Central registry for ocean gates
- Shapefile loading with geopandas
- Buffer calculations for data selection
- Satellite pass filtering

Usage:
    from src.gates import GateCatalog, GateLoader
    
    catalog = GateCatalog()
    gates = catalog.list_all()
    
    loader = GateLoader()
    geometry = loader.load("fram_strait")
"""

from src.gates.catalog import GateCatalog
from src.gates.loader import GateLoader
from src.gates.buffer import GateBuffer
from src.gates.passes import PassFilter

__all__ = [
    "GateCatalog",
    "GateLoader", 
    "GateBuffer",
    "PassFilter",
]
