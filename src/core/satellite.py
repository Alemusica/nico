"""
Satellite Detection and Configuration
=====================================
Functions for detecting satellite type and mission parameters.
"""

import os
from pathlib import Path


def detect_satellite_type(base_dir: str | Path) -> str:
    """
    Detect satellite type (J1 or J2) from the base directory name.
    
    Parameters
    ----------
    base_dir : str or Path
        Base directory path containing the satellite data
        
    Returns
    -------
    str
        'J1' or 'J2' based on directory name, defaults to 'J1' if unclear
        
    Examples
    --------
    >>> detect_satellite_type("/data/Jason2")
    'J2'
    >>> detect_satellite_type("/data/J1_cycles")
    'J1'
    """
    base_dir_str = os.fspath(base_dir)
    dir_name = os.path.basename(base_dir_str.rstrip("/"))
    
    if "J2" in dir_name.upper():
        return "J2"
    return "J1"


def get_cycle_info(satellite_type: str) -> dict:
    """
    Get mission-specific cycle information.
    
    Parameters
    ----------
    satellite_type : str
        'J1' for Jason-1 or 'J2' for Jason-2
        
    Returns
    -------
    dict
        Dictionary with 'max_cycles', 'repeat_period', 'mission_name'
    """
    missions = {
        "J1": {
            "max_cycles": 537,
            "repeat_period": 9.9156,  # days
            "mission_name": "Jason-1",
            "launch_date": "2001-12-07",
            "end_date": "2013-07-01",
        },
        "J2": {
            "max_cycles": 327,
            "repeat_period": 9.9156,  # days
            "mission_name": "Jason-2/OSTM",
            "launch_date": "2008-06-20",
            "end_date": "2019-10-01",
        },
    }
    
    return missions.get(satellite_type.upper(), missions["J1"])
