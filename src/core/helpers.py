"""
Helper Functions
================
General utility functions for file handling and data extraction.
"""

import re
from pathlib import Path


def extract_cycle_number(filename: str) -> int | None:
    """
    Extract cycle number from SLCCI filename.
    
    Parameters
    ----------
    filename : str
        Filename like 'SLCCI_ALTDB_J1_Cycle001_V2.nc'
        
    Returns
    -------
    int or None
        Extracted cycle number, or None if not found
        
    Examples
    --------
    >>> extract_cycle_number("SLCCI_ALTDB_J1_Cycle042_V2.nc")
    42
    >>> extract_cycle_number("random_file.nc")
    None
    """
    match = re.search(r'Cycle(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def extract_strait_info(path: str | Path) -> tuple[str, int | None]:
    """
    Extract strait name and pass number from a gate shapefile path.
    
    Parameters
    ----------
    path : str or Path
        Path to gate shapefile
        
    Returns
    -------
    tuple[str, int | None]
        (strait_name, pass_number) where pass_number may be None
        
    Examples
    --------
    >>> extract_strait_info("davis_strait_TPJ_pass_248.shp")
    ('Davis Strait Tpj Pass 248', 248)
    """
    filename = Path(path).stem
    strait_name = filename.replace("_", " ").replace("-", " ").title()
    
    match = re.search(r'pass[_\s]*(\d+)', filename, re.IGNORECASE)
    pass_from_filename = int(match.group(1)) if match else None
    
    return strait_name, pass_from_filename


def format_time_range(start, end) -> str:
    """Format a time range for display."""
    try:
        return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    except AttributeError:
        return f"{start} to {end}"


def validate_netcdf_file(filepath: str | Path) -> bool:
    """
    Check if a file is a valid NetCDF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to file to validate
        
    Returns
    -------
    bool
        True if file exists and has .nc extension
    """
    path = Path(filepath)
    return path.exists() and path.suffix.lower() in ['.nc', '.nc4', '.netcdf']
