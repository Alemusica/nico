"""
Cache Service
=============
Unified persistent cache for processed satellite data.

Saves PassData objects to disk for instant loading on subsequent runs.
Supports all datasets: SLCCI, CMEMS L3, CMEMS L4, DTUSpace.

Cache Structure:
    data/cache/processed/
    ├── index.json                    # Index of all cached items
    ├── slcci/
    │   ├── fram_strait_pass_248.pkl  # PassData for Fram Strait pass 248
    │   └── denmark_strait_pass_85.pkl
    ├── cmems_l3/
    │   └── barents_sea_track_123.pkl
    ├── cmems_l4/
    │   └── fram_strait.pkl
    └── dtuspace/
        └── fram_strait.pkl

Usage:
    from src.services.cache_service import DataCache
    
    cache = DataCache()
    
    # Save data
    cache.save("slcci", "fram_strait", pass_data, pass_number=248)
    
    # Load data (returns None if not cached)
    pass_data = cache.load("slcci", "fram_strait", pass_number=248)
    
    # Check if cached
    if cache.exists("slcci", "fram_strait", pass_number=248):
        print("Data is cached!")
    
    # List cached items
    items = cache.list_cached("slcci")
    
    # Clear cache
    cache.clear("slcci", "fram_strait")  # Clear specific
    cache.clear_all()  # Clear everything
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict, is_dataclass
import numpy as np
import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Index file for tracking all cached items
INDEX_FILE = CACHE_DIR / "index.json"


@dataclass
class CacheEntry:
    """Metadata for a cached item."""
    dataset: str           # slcci, cmems_l3, cmems_l4, dtuspace
    gate_name: str         # e.g., "fram_strait"
    pass_number: Optional[int]  # For along-track datasets
    track_number: Optional[int]  # For CMEMS L3
    file_path: str         # Relative path to pickle file
    created_at: str        # ISO timestamp
    size_bytes: int        # File size
    n_observations: int    # Number of data points
    date_range: str        # e.g., "2002-01 to 2021-12"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "CacheEntry":
        return cls(**d)


class DataCache:
    """
    Unified persistent cache for processed satellite data.
    
    Saves PassData objects as pickle files for instant loading.
    Maintains an index.json for quick lookup without loading data.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache with optional custom directory."""
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = self._load_index()
    
    # ==========================================================================
    # PUBLIC API
    # ==========================================================================
    
    def save(
        self, 
        dataset: str, 
        gate_name: str, 
        data: Any,
        pass_number: Optional[int] = None,
        track_number: Optional[int] = None,
    ) -> bool:
        """
        Save PassData to cache.
        
        Args:
            dataset: Dataset type (slcci, cmems_l3, cmems_l4, dtuspace)
            gate_name: Gate identifier (e.g., "fram_strait")
            data: PassData object to cache
            pass_number: Pass number for along-track datasets
            track_number: Track number for CMEMS L3
            
        Returns:
            True if saved successfully
        """
        try:
            # Create dataset subdirectory
            dataset_dir = self.cache_dir / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            filename = self._generate_filename(gate_name, pass_number, track_number)
            file_path = dataset_dir / filename
            
            # Save data
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Extract metadata
            n_obs = 0
            date_range = "unknown"
            
            if hasattr(data, 'df') and isinstance(data.df, pd.DataFrame):
                n_obs = len(data.df)
                if 'time' in data.df.columns:
                    min_date = data.df['time'].min()
                    max_date = data.df['time'].max()
                    date_range = f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}"
            elif hasattr(data, 'time_periods') and len(data.time_periods) > 0:
                date_range = f"{data.time_periods[0]} to {data.time_periods[-1]}"
            
            # Create index entry
            entry = CacheEntry(
                dataset=dataset,
                gate_name=gate_name,
                pass_number=pass_number,
                track_number=track_number,
                file_path=str(file_path.relative_to(self.cache_dir)),
                created_at=datetime.now().isoformat(),
                size_bytes=file_path.stat().st_size,
                n_observations=n_obs,
                date_range=date_range,
            )
            
            # Update index
            key = self._make_key(dataset, gate_name, pass_number, track_number)
            self._index[key] = entry.to_dict()
            self._save_index()
            
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"💾 Cached {dataset}/{gate_name}: {n_obs:,} obs, {size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
            return False
    
    def load(
        self, 
        dataset: str, 
        gate_name: str,
        pass_number: Optional[int] = None,
        track_number: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Load PassData from cache.
        
        Args:
            dataset: Dataset type
            gate_name: Gate identifier
            pass_number: Pass number (for along-track)
            track_number: Track number (for CMEMS L3)
            
        Returns:
            PassData object or None if not cached
        """
        key = self._make_key(dataset, gate_name, pass_number, track_number)
        
        if key not in self._index:
            logger.debug(f"Cache miss: {key}")
            return None
        
        entry = self._index[key]
        file_path = self.cache_dir / entry['file_path']
        
        if not file_path.exists():
            # Remove stale entry
            del self._index[key]
            self._save_index()
            logger.warning(f"Cache file missing, removed entry: {key}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"📦 Loaded from cache: {dataset}/{gate_name} ({size_mb:.1f} MB)")
            
            return data
            
        except Exception as e:
            logger.error(f"Cache load failed: {e}")
            return None
    
    def exists(
        self, 
        dataset: str, 
        gate_name: str,
        pass_number: Optional[int] = None,
        track_number: Optional[int] = None,
    ) -> bool:
        """Check if data is cached."""
        key = self._make_key(dataset, gate_name, pass_number, track_number)
        
        if key not in self._index:
            return False
        
        # Verify file still exists
        entry = self._index[key]
        file_path = self.cache_dir / entry['file_path']
        
        if not file_path.exists():
            del self._index[key]
            self._save_index()
            return False
        
        return True
    
    def get_info(
        self, 
        dataset: str, 
        gate_name: str,
        pass_number: Optional[int] = None,
        track_number: Optional[int] = None,
    ) -> Optional[Dict]:
        """Get cache entry metadata without loading data."""
        key = self._make_key(dataset, gate_name, pass_number, track_number)
        return self._index.get(key)
    
    def list_cached(self, dataset: Optional[str] = None) -> List[Dict]:
        """
        List all cached items, optionally filtered by dataset.
        
        Returns:
            List of cache entry dictionaries
        """
        entries = []
        
        for key, entry in self._index.items():
            if dataset is None or entry['dataset'] == dataset:
                entries.append(entry)
        
        return entries
    
    def clear(
        self, 
        dataset: str, 
        gate_name: str,
        pass_number: Optional[int] = None,
        track_number: Optional[int] = None,
    ) -> bool:
        """Clear specific cached item."""
        key = self._make_key(dataset, gate_name, pass_number, track_number)
        
        if key not in self._index:
            return False
        
        entry = self._index[key]
        file_path = self.cache_dir / entry['file_path']
        
        try:
            if file_path.exists():
                file_path.unlink()
            del self._index[key]
            self._save_index()
            logger.info(f"🗑️ Cleared cache: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    def clear_dataset(self, dataset: str) -> int:
        """Clear all cached items for a dataset."""
        keys_to_remove = [
            key for key, entry in self._index.items() 
            if entry['dataset'] == dataset
        ]
        
        count = 0
        for key in keys_to_remove:
            entry = self._index[key]
            file_path = self.cache_dir / entry['file_path']
            try:
                if file_path.exists():
                    file_path.unlink()
                del self._index[key]
                count += 1
            except Exception:
                pass
        
        self._save_index()
        logger.info(f"🗑️ Cleared {count} items from {dataset} cache")
        return count
    
    def clear_all(self) -> int:
        """Clear entire cache."""
        count = 0
        
        for key, entry in list(self._index.items()):
            file_path = self.cache_dir / entry['file_path']
            try:
                if file_path.exists():
                    file_path.unlink()
                count += 1
            except Exception:
                pass
        
        self._index = {}
        self._save_index()
        
        logger.info(f"🗑️ Cleared entire cache: {count} items")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        by_dataset = {}
        
        for entry in self._index.values():
            dataset = entry['dataset']
            size = entry.get('size_bytes', 0)
            total_size += size
            
            if dataset not in by_dataset:
                by_dataset[dataset] = {'count': 0, 'size_bytes': 0}
            
            by_dataset[dataset]['count'] += 1
            by_dataset[dataset]['size_bytes'] += size
        
        return {
            'total_items': len(self._index),
            'total_size_mb': total_size / (1024 * 1024),
            'by_dataset': by_dataset,
        }
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _make_key(
        self, 
        dataset: str, 
        gate_name: str,
        pass_number: Optional[int] = None,
        track_number: Optional[int] = None,
    ) -> str:
        """Generate unique cache key."""
        parts = [dataset, gate_name.lower().replace(' ', '_')]
        
        if pass_number is not None:
            parts.append(f"pass_{pass_number}")
        
        if track_number is not None:
            parts.append(f"track_{track_number}")
        
        return "/".join(parts)
    
    def _generate_filename(
        self, 
        gate_name: str,
        pass_number: Optional[int] = None,
        track_number: Optional[int] = None,
    ) -> str:
        """Generate cache filename."""
        clean_name = gate_name.lower().replace(' ', '_').replace('-', '_')
        
        if pass_number is not None:
            return f"{clean_name}_pass_{pass_number}.pkl"
        elif track_number is not None:
            return f"{clean_name}_track_{track_number}.pkl"
        else:
            return f"{clean_name}.pkl"
    
    def _load_index(self) -> Dict:
        """Load cache index from disk."""
        if INDEX_FILE.exists():
            try:
                with open(INDEX_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(INDEX_FILE, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

# Global cache instance
_cache: Optional[DataCache] = None


def get_cache() -> DataCache:
    """Get global cache instance (singleton)."""
    global _cache
    if _cache is None:
        _cache = DataCache()
    return _cache


def cache_pass_data(
    dataset: str,
    gate_name: str,
    pass_data: Any,
    pass_number: Optional[int] = None,
    track_number: Optional[int] = None,
) -> bool:
    """Convenience function to cache PassData."""
    return get_cache().save(dataset, gate_name, pass_data, pass_number, track_number)


def load_cached_pass_data(
    dataset: str,
    gate_name: str,
    pass_number: Optional[int] = None,
    track_number: Optional[int] = None,
) -> Optional[Any]:
    """Convenience function to load cached PassData."""
    return get_cache().load(dataset, gate_name, pass_number, track_number)


def is_cached(
    dataset: str,
    gate_name: str,
    pass_number: Optional[int] = None,
    track_number: Optional[int] = None,
) -> bool:
    """Convenience function to check if data is cached."""
    return get_cache().exists(dataset, gate_name, pass_number, track_number)
