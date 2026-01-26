"""
Intelligent Cache
=================
Generic two-level cache for data services WITH disk persistence.

This module provides a reusable cache implementation that can be used by:
- SLCCIService (already has its own, but can migrate)
- DTUService
- CMEMSL4Service
- Any other data service

Features:
- Two-level caching: L1 (raw data) + L2 (processed data)
- TTL-based expiration
- Max entries limit with LRU eviction
- Parameter-aware cache keys (different bin_size → different cache entry)
- Statistics tracking (hits, misses, invalidations)
- **Disk persistence** with auto-save/load

Usage:
    from src.services.intelligent_cache import IntelligentCache, CacheConfig
    
    cache = IntelligentCache(CacheConfig(ttl_days=14, max_entries=50))
    
    # Store raw data
    cache.set_raw("dtu", "fram_strait", df)
    
    # Retrieve raw data
    df = cache.get_raw("dtu", "fram_strait")
    
    # Store processed data with parameters
    cache.set_processed("dtu", "fram_strait", pass_data, n_gate_pts=400)
    
    # Retrieve processed data
    pass_data = cache.get_processed("dtu", "fram_strait", n_gate_pts=400)
    
    # Persistence
    cache.save_to_disk()  # Manual save
    cache.load_from_disk()  # Manual load (auto-called on init)
"""

import hashlib
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "intelligent"


@dataclass
class CacheConfig:
    """Configuration for intelligent caching."""
    enabled: bool = True
    max_entries: int = 100  # Max cached entries per level (increased from 50)
    persist_to_disk: bool = True  # Enable disk persistence
    cache_dir: Optional[Path] = None  # Custom cache directory (default: data/cache/intelligent)
    auto_save_interval: int = 5  # Auto-save after N set operations (reduced for safety)


class IntelligentCache:
    """
    Generic two-level in-memory cache for data services.
    
    Two-level architecture:
    - Level 1 (raw): Raw data after loading (before processing)
    - Level 2 (processed): Processed data objects (after processing with specific parameters)
    
    Cache keys include:
    - service_name (e.g., "dtu", "cmems_l4", "slcci")
    - entity_key (e.g., strait name, gate hash)
    - additional params (for processed cache: bin_size, n_gate_pts, etc.)
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Set cache directory
        self._cache_dir = self.config.cache_dir or DEFAULT_CACHE_DIR
        if self.config.persist_to_disk:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Level 1: Raw data cache
        # Key: "{service}_{entity_key}" → (data, timestamp)
        self._raw_cache: Dict[str, Tuple[Any, float]] = {}
        
        # Level 2: Processed data cache  
        # Key: "{service}_{entity_key}_{params_hash}" → (data, timestamp)
        self._processed_cache: Dict[str, Tuple[Any, float]] = {}
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "evictions": 0,
        }
        
        # Track set operations for auto-save
        self._set_count = 0
        
        # Load from disk on init
        if self.config.persist_to_disk:
            self._load_from_disk()
    
    # =========================================================================
    # Key Generation
    # =========================================================================
    
    @staticmethod
    def _hash_params(**kwargs) -> str:
        """Create hash from parameters dict."""
        if not kwargs:
            return "default"
        # Sort for consistency
        params_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(params_str.encode()).hexdigest()[:10]
    
    def _make_raw_key(self, service: str, entity_key: str) -> str:
        """Generate cache key for raw data."""
        return f"raw_{service}_{entity_key}"
    
    def _make_processed_key(self, service: str, entity_key: str, **params) -> str:
        """Generate cache key for processed data."""
        params_hash = self._hash_params(**params)
        return f"proc_{service}_{entity_key}_{params_hash}"
    
    # =========================================================================
    # Eviction (LRU when max entries exceeded)
    # =========================================================================
    
    def _enforce_max_entries(self, cache: dict):
        """Remove oldest entries if cache exceeds max size (LRU eviction)."""
        if len(cache) <= self.config.max_entries:
            return
        
        # Sort by timestamp (oldest first)
        sorted_keys = sorted(cache.keys(), key=lambda k: cache[k][1])
        n_to_remove = len(cache) - self.config.max_entries
        
        for key in sorted_keys[:n_to_remove]:
            del cache[key]
            self._stats["evictions"] += 1
            logger.debug(f"Cache evicted (LRU): {key}")
    
    # =========================================================================
    # Raw Data Cache (Level 1)
    # =========================================================================
    
    def get_raw(self, service: str, entity_key: str) -> Optional[Any]:
        """
        Get raw data from L1 cache.
        
        Parameters
        ----------
        service : str
            Service name (e.g., "dtu", "cmems_l4")
        entity_key : str
            Entity identifier (e.g., strait name, gate hash)
            
        Returns
        -------
        data or None
            Cached data or None if not found
        """
        if not self.config.enabled:
            return None
        
        key = self._make_raw_key(service, entity_key)
        
        if key in self._raw_cache:
            data, timestamp = self._raw_cache[key]
            self._stats["hits"] += 1
            logger.debug(f"Cache HIT (L1 raw): {key}")
            return data
        
        self._stats["misses"] += 1
        return None
    
    def set_raw(self, service: str, entity_key: str, data: Any):
        """
        Store raw data in L1 cache.
        
        Parameters
        ----------
        service : str
            Service name
        entity_key : str
            Entity identifier
        data : Any
            Data to cache
        """
        if not self.config.enabled:
            return
        
        key = self._make_raw_key(service, entity_key)
        self._raw_cache[key] = (data, time.time())
        self._enforce_max_entries(self._raw_cache)
        logger.debug(f"Cache SET (L1 raw): {key}")
        self._maybe_auto_save()
    
    # =========================================================================
    # Processed Data Cache (Level 2)
    # =========================================================================
    
    def get_processed(self, service: str, entity_key: str, **params) -> Optional[Any]:
        """
        Get processed data from L2 cache.
        
        Parameters
        ----------
        service : str
            Service name
        entity_key : str
            Entity identifier
        **params
            Processing parameters (e.g., n_gate_pts=400, bin_size=0.1)
            
        Returns
        -------
        data or None
            Cached data or None if not found
        """
        if not self.config.enabled:
            return None
        
        key = self._make_processed_key(service, entity_key, **params)
        
        if key in self._processed_cache:
            data, timestamp = self._processed_cache[key]
            self._stats["hits"] += 1
            logger.debug(f"Cache HIT (L2 processed): {key}")
            return data
        
        self._stats["misses"] += 1
        return None
    
    def set_processed(self, service: str, entity_key: str, data: Any, **params):
        """
        Store processed data in L2 cache.
        
        Parameters
        ----------
        service : str
            Service name
        entity_key : str
            Entity identifier
        data : Any
            Processed data to cache
        **params
            Processing parameters used (for cache key)
        """
        if not self.config.enabled:
            return
        
        key = self._make_processed_key(service, entity_key, **params)
        self._processed_cache[key] = (data, time.time())
        self._enforce_max_entries(self._processed_cache)
        logger.debug(f"Cache SET (L2 processed): {key}")
        self._maybe_auto_save()
    
    # =========================================================================
    # Cache Management
    # =========================================================================
    
    def invalidate(self, service: Optional[str] = None, entity_key: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Parameters
        ----------
        service : str, optional
            If provided, only invalidate entries for this service
        entity_key : str, optional
            If provided, only invalidate entries for this entity
        """
        def should_remove(key: str) -> bool:
            if service and f"_{service}_" not in key:
                return False
            if entity_key and entity_key not in key:
                return False
            return True
        
        # Remove from both caches
        raw_keys = [k for k in self._raw_cache.keys() if should_remove(k)]
        proc_keys = [k for k in self._processed_cache.keys() if should_remove(k)]
        
        for k in raw_keys:
            del self._raw_cache[k]
        for k in proc_keys:
            del self._processed_cache[k]
        
        count = len(raw_keys) + len(proc_keys)
        self._stats["invalidations"] += count
        
        if count > 0:
            scope = f"service={service}, entity={entity_key}" if service or entity_key else "all"
            logger.info(f"Cache invalidated: {count} entries ({scope})")
    
    def invalidate_processed(self, service: str, entity_key: str):
        """
        Invalidate only processed cache for an entity (keep raw).
        
        Useful when processing parameters change (e.g., bin_size, n_gate_pts).
        """
        prefix = f"proc_{service}_{entity_key}"
        keys_to_remove = [k for k in self._processed_cache.keys() if k.startswith(prefix)]
        
        for key in keys_to_remove:
            del self._processed_cache[key]
            self._stats["invalidations"] += 1
        
        if keys_to_remove:
            logger.info(f"Cache invalidated {len(keys_to_remove)} processed entries for {service}/{entity_key}")
    
    def clear(self):
        """Clear all cache entries."""
        count = len(self._raw_cache) + len(self._processed_cache)
        self._raw_cache.clear()
        self._processed_cache.clear()
        self._stats["invalidations"] += count
        logger.info(f"Cache cleared: {count} entries removed")
        
        # Also clear disk cache
        if self.config.persist_to_disk:
            self._clear_disk_cache()
    
    # =========================================================================
    # Disk Persistence
    # =========================================================================
    
    def _get_raw_cache_file(self) -> Path:
        """Get path to L1 cache file."""
        return self._cache_dir / "cache_l1_raw.pkl"
    
    def _get_processed_cache_file(self) -> Path:
        """Get path to L2 cache file."""
        return self._cache_dir / "cache_l2_processed.pkl"
    
    def _maybe_auto_save(self):
        """Auto-save to disk after N set operations."""
        if not self.config.persist_to_disk:
            return
        
        self._set_count += 1
        if self._set_count >= self.config.auto_save_interval:
            self.save_to_disk()
            self._set_count = 0
    
    def save_to_disk(self):
        """
        Save cache to disk.
        
        Creates two pickle files:
        - cache_l1_raw.pkl: Raw data cache
        - cache_l2_processed.pkl: Processed data cache
        """
        if not self.config.persist_to_disk:
            logger.debug("Disk persistence disabled, skipping save")
            return
        
        try:
            # Save L1 (raw) cache
            raw_file = self._get_raw_cache_file()
            with open(raw_file, 'wb') as f:
                pickle.dump(self._raw_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save L2 (processed) cache
            proc_file = self._get_processed_cache_file()
            with open(proc_file, 'wb') as f:
                pickle.dump(self._processed_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            raw_size = raw_file.stat().st_size / (1024 * 1024)
            proc_size = proc_file.stat().st_size / (1024 * 1024)
            
            logger.debug(
                f"Cache saved to disk: L1={len(self._raw_cache)} entries ({raw_size:.2f} MB), "
                f"L2={len(self._processed_cache)} entries ({proc_size:.2f} MB)"
            )
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")
    
    def _load_from_disk(self):
        """
        Load cache from disk.
        
        Called automatically on init if persist_to_disk=True.
        Data persists indefinitely (no TTL expiration).
        """
        try:
            # Load L1 (raw) cache
            raw_file = self._get_raw_cache_file()
            if raw_file.exists() and raw_file.stat().st_size > 0:
                with open(raw_file, 'rb') as f:
                    self._raw_cache = pickle.load(f)
                logger.info(f"L1 cache loaded: {len(self._raw_cache)} entries")
            else:
                logger.debug("L1 cache file empty or missing, starting fresh")
            
            # Load L2 (processed) cache
            proc_file = self._get_processed_cache_file()
            if proc_file.exists() and proc_file.stat().st_size > 0:
                with open(proc_file, 'rb') as f:
                    self._processed_cache = pickle.load(f)
                logger.info(f"L2 cache loaded: {len(self._processed_cache)} entries")
            else:
                logger.debug("L2 cache file empty or missing, starting fresh")
            
            logger.info(
                f"Cache loaded from disk: L1={len(self._raw_cache)} entries, "
                f"L2={len(self._processed_cache)} entries"
            )
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
            # Start fresh but don't overwrite what we loaded
            if not self._raw_cache:
                self._raw_cache = {}
            if not self._processed_cache:
                self._processed_cache = {}
    
    def _clear_disk_cache(self):
        """Remove cache files from disk."""
        try:
            raw_file = self._get_raw_cache_file()
            proc_file = self._get_processed_cache_file()
            
            if raw_file.exists():
                raw_file.unlink()
            if proc_file.exists():
                proc_file.unlink()
            
            logger.debug("Disk cache files removed")
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")
    
    def load_from_disk(self):
        """
        Public method to reload cache from disk.
        
        Useful for refreshing cache after external changes.
        """
        self._load_from_disk()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        # Calculate disk size
        disk_size_mb = 0.0
        if self.config.persist_to_disk:
            raw_file = self._get_raw_cache_file()
            proc_file = self._get_processed_cache_file()
            if raw_file.exists():
                disk_size_mb += raw_file.stat().st_size / (1024 * 1024)
            if proc_file.exists():
                disk_size_mb += proc_file.stat().st_size / (1024 * 1024)
        
        return {
            **self._stats,
            "raw_entries": len(self._raw_cache),
            "processed_entries": len(self._processed_cache),
            "total_entries": len(self._raw_cache) + len(self._processed_cache),
            "total_items": len(self._raw_cache) + len(self._processed_cache),  # Alias for UI
            "total_size_mb": round(disk_size_mb, 2),  # Alias for UI
            "hit_rate": (
                self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
            ) * 100,
            "disk_size_mb": round(disk_size_mb, 2),
            "persist_enabled": self.config.persist_to_disk,
        }
    
    def get_all_entries(self) -> List[Dict]:
        """
        Get list of all cache entries for UI display.
        
        Returns list of dicts with:
        - key: Full cache key
        - dataset: Service name (slcci, cmems_l4, dtu, etc.)
        - gate: Gate name
        - pass: Pass number (if any)
        - track: Track number (if any)
        - date_range: Date range string
        - n_obs: Number of observations
        - level: L1 (raw) or L2 (processed)
        """
        entries = []
        
        for key, (data, timestamp) in self._raw_cache.items():
            entry = self._parse_cache_key(key, data, timestamp, "L1")
            if entry:
                entries.append(entry)
        
        for key, (data, timestamp) in self._processed_cache.items():
            entry = self._parse_cache_key(key, data, timestamp, "L2")
            if entry:
                entries.append(entry)
        
        return entries
    
    def _parse_cache_key(self, key: str, data: Any, timestamp: float, level: str) -> Optional[Dict]:
        """Parse a cache key into display-friendly dict."""
        # Key format: "raw_{service}_{entity}" or "proc_{service}_{entity}_{hash}"
        parts = key.split("_")
        if len(parts) < 3:
            return None
        
        level_prefix = parts[0]  # "raw" or "proc"
        service = parts[1]  # e.g., "dtu", "cmems", "slcci"
        entity_parts = parts[2:]  # Rest is entity key
        
        # Reconstruct entity (may contain underscores)
        entity = "_".join(entity_parts)
        if level_prefix == "proc" and len(entity_parts) > 1:
            # Remove hash from end
            entity = "_".join(entity_parts[:-1])
        
        # Try to extract info from data
        n_obs = 0
        date_range = "N/A"
        pass_num = None
        track_num = None
        
        if hasattr(data, 'df') and data.df is not None:
            n_obs = len(data.df)
            if hasattr(data, 'time_array') and len(data.time_array) > 0:
                try:
                    import pandas as pd
                    times = pd.to_datetime(data.time_array)
                    date_range = f"{times.min().strftime('%Y-%m')} to {times.max().strftime('%Y-%m')}"
                except:
                    pass
            if hasattr(data, 'pass_number'):
                pass_num = data.pass_number
            if hasattr(data, 'track_number'):
                track_num = data.track_number
        elif hasattr(data, '__len__'):
            n_obs = len(data)
        
        # Map service names for display
        dataset_map = {
            "dtu": "dtuspace",
            "cmems": "cmems_l4",
            "slcci": "slcci",
            "l4": "cmems_l4",
        }
        dataset = dataset_map.get(service, service)
        
        return {
            "key": key,
            "dataset": dataset,
            "gate": entity.replace("_", " ").title(),
            "pass": pass_num,
            "track": track_num,
            "date_range": date_range,
            "n_obs": n_obs,
            "level": level,
            "age_hours": round((time.time() - timestamp) / 3600, 1),
        }
    
    def clear_by_key(self, key: str):
        """Clear a specific cache entry by key."""
        if key in self._raw_cache:
            del self._raw_cache[key]
            self._stats["invalidations"] += 1
            logger.debug(f"Cleared L1 entry: {key}")
        if key in self._processed_cache:
            del self._processed_cache[key]
            self._stats["invalidations"] += 1
            logger.debug(f"Cleared L2 entry: {key}")
        
        # Save changes to disk
        if self.config.persist_to_disk:
            self.save_to_disk()
    
    def clear_all(self):
        """Alias for clear() for UI compatibility."""
        self.clear()
    
    # =========================================================================
    # Compatibility API (for sidebar migration from DataCache)
    # =========================================================================
    
    def load(
        self, 
        dataset: str, 
        gate_name: str, 
        pass_number: Optional[int] = None,
        time_range: Optional[Tuple[int, int]] = None,
    ) -> Optional[Any]:
        """
        Load cached data (compatibility API for DataCache migration).
        
        This wraps get_processed() with a simpler interface.
        
        Parameters
        ----------
        dataset : str
            Dataset type (e.g., "cmems_l4", "dtuspace")
        gate_name : str
            Gate identifier (normalized)
        pass_number : int, optional
            Pass number for along-track data
        time_range : tuple, optional
            (start_year, end_year) tuple
            
        Returns
        -------
        data or None
        """
        # Build entity key
        entity_key = gate_name
        
        # Build params for cache key
        params = {}
        if pass_number is not None:
            params["pass_number"] = pass_number
        if time_range is not None:
            params["time_range"] = f"{time_range[0]}_{time_range[1]}"
        
        return self.get_processed(dataset, entity_key, **params)
    
    def save(
        self,
        dataset: str,
        gate_name: str,
        data: Any,
        pass_number: Optional[int] = None,
        time_range: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Save data to cache (compatibility API for DataCache migration).
        
        This wraps set_processed() with a simpler interface.
        
        Parameters
        ----------
        dataset : str
            Dataset type (e.g., "cmems_l4", "dtuspace")
        gate_name : str
            Gate identifier (normalized)
        data : Any
            Data to cache
        pass_number : int, optional
            Pass number for along-track data
        time_range : tuple, optional
            (start_year, end_year) tuple
            
        Returns
        -------
        bool
            True if saved successfully
        """
        try:
            # Build entity key
            entity_key = gate_name
            
            # Build params for cache key
            params = {}
            if pass_number is not None:
                params["pass_number"] = pass_number
            if time_range is not None:
                params["time_range"] = f"{time_range[0]}_{time_range[1]}"
            
            self.set_processed(dataset, entity_key, data, **params)
            return True
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
            return False
    
    def exists(
        self,
        dataset: str,
        gate_name: str,
        pass_number: Optional[int] = None,
        time_range: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Check if data is cached (compatibility API)."""
        return self.load(dataset, gate_name, pass_number, time_range) is not None
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"IntelligentCache("
            f"L1={stats['raw_entries']}, "
            f"L2={stats['processed_entries']}, "
            f"hits={stats['hits']}, "
            f"misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.1f}%)"
        )


# =============================================================================
# Global Cache Instance (Singleton)
# =============================================================================

_global_cache: Optional[IntelligentCache] = None


def get_intelligent_cache(config: Optional[CacheConfig] = None) -> IntelligentCache:
    """
    Get the global intelligent cache instance (singleton).
    
    Parameters
    ----------
    config : CacheConfig, optional
        Configuration (only used on first call)
        
    Returns
    -------
    IntelligentCache
        The global cache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = IntelligentCache(config)
        logger.info(f"Initialized global IntelligentCache: {_global_cache.config}")
    
    return _global_cache


def clear_global_cache():
    """Clear the global cache (memory and disk)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()


def save_global_cache():
    """Explicitly save the global cache to disk."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.save_to_disk()
