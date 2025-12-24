"""
💾 Data Cache
=============

Persistent cache for downloaded data with indexing.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sqlite3

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


@dataclass
class CacheEntry:
    """A cached data entry."""
    id: str
    source: str  # era5, cmems_sla, etc.
    variables: List[str]
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    time_range: Tuple[str, str]
    resolution_temporal: str
    resolution_spatial: str
    file_path: str
    file_size_mb: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source,
            "variables": self.variables,
            "lat_range": self.lat_range,
            "lon_range": self.lon_range,
            "time_range": self.time_range,
            "resolution_temporal": self.resolution_temporal,
            "resolution_spatial": self.resolution_spatial,
            "file_path": self.file_path,
            "file_size_mb": self.file_size_mb,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
        }


class DataCache:
    """
    Persistent data cache with SQLite index.
    
    Stores downloaded data files and tracks metadata for fast retrieval.
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories for organization
        (self.cache_dir / "era5").mkdir(exist_ok=True)
        (self.cache_dir / "cmems").mkdir(exist_ok=True)
        (self.cache_dir / "indices").mkdir(exist_ok=True)
        
        # SQLite index
        self.db_path = self.cache_dir / "cache_index.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                variables TEXT NOT NULL,
                lat_min REAL NOT NULL,
                lat_max REAL NOT NULL,
                lon_min REAL NOT NULL,
                lon_max REAL NOT NULL,
                time_start TEXT NOT NULL,
                time_end TEXT NOT NULL,
                resolution_temporal TEXT NOT NULL,
                resolution_spatial TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size_mb REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON cache_entries(source)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_time ON cache_entries(time_start, time_end)
        """)
        conn.commit()
        conn.close()
    
    def _generate_id(
        self,
        source: str,
        variables: List[str],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        resolution_temporal: str,
        resolution_spatial: str,
    ) -> str:
        """Generate unique cache ID."""
        key = f"{source}_{sorted(variables)}_{lat_range}_{lon_range}_{time_range}_{resolution_temporal}_{resolution_spatial}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def find(
        self,
        source: str,
        variables: List[str],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        resolution_temporal: str = None,
        resolution_spatial: str = None,
    ) -> Optional[CacheEntry]:
        """
        Find cached data that covers the requested area/time.
        
        Returns the best matching entry or None.
        """
        conn = sqlite3.connect(self.db_path)
        
        # Look for exact or containing match
        query = """
            SELECT * FROM cache_entries
            WHERE source = ?
            AND lat_min <= ? AND lat_max >= ?
            AND lon_min <= ? AND lon_max >= ?
            AND time_start <= ? AND time_end >= ?
        """
        params = [
            source,
            lat_range[0], lat_range[1],
            lon_range[0], lon_range[1],
            time_range[0], time_range[1],
        ]
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        # Find best match (smallest area that contains request)
        best = None
        best_area = float('inf')
        
        for row in rows:
            entry = self._row_to_entry(row)
            
            # Check variables
            cached_vars = set(entry.variables)
            requested_vars = set(variables)
            if not requested_vars.issubset(cached_vars):
                continue
            
            # Calculate area
            area = (entry.lat_range[1] - entry.lat_range[0]) * (entry.lon_range[1] - entry.lon_range[0])
            if area < best_area:
                best_area = area
                best = entry
        
        return best
    
    def _row_to_entry(self, row) -> CacheEntry:
        """Convert database row to CacheEntry."""
        return CacheEntry(
            id=row[0],
            source=row[1],
            variables=json.loads(row[2]),
            lat_range=(row[3], row[4]),
            lon_range=(row[5], row[6]),
            time_range=(row[7], row[8]),
            resolution_temporal=row[9],
            resolution_spatial=row[10],
            file_path=row[11],
            file_size_mb=row[12],
            created_at=datetime.fromisoformat(row[13]),
            last_accessed=datetime.fromisoformat(row[14]),
            access_count=row[15],
        )
    
    def add(
        self,
        source: str,
        variables: List[str],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        resolution_temporal: str,
        resolution_spatial: str,
        data: Any,  # xarray.Dataset or dict
    ) -> CacheEntry:
        """
        Add data to cache.
        
        Args:
            source: Data source name
            variables: Variables in dataset
            lat_range: (min, max) latitude
            lon_range: (min, max) longitude
            time_range: (start, end) dates
            resolution_temporal: Temporal resolution
            resolution_spatial: Spatial resolution
            data: Data to cache (xarray.Dataset or dict)
            
        Returns:
            CacheEntry for the stored data
        """
        # Generate ID
        cache_id = self._generate_id(
            source, variables, lat_range, lon_range, time_range,
            resolution_temporal, resolution_spatial
        )
        
        # Determine file path
        ext = "nc" if HAS_XARRAY and hasattr(data, 'to_netcdf') else "json"
        file_path = self.cache_dir / source / f"{cache_id}.{ext}"
        
        # Save data
        if ext == "nc" and HAS_XARRAY:
            data.to_netcdf(file_path)
            file_size = file_path.stat().st_size / (1024 * 1024)
        else:
            with open(file_path, "w") as f:
                json.dump(data if isinstance(data, dict) else {"data": "stored"}, f)
            file_size = file_path.stat().st_size / (1024 * 1024)
        
        # Create entry
        now = datetime.now()
        entry = CacheEntry(
            id=cache_id,
            source=source,
            variables=variables,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            resolution_temporal=resolution_temporal,
            resolution_spatial=resolution_spatial,
            file_path=str(file_path),
            file_size_mb=file_size,
            created_at=now,
            last_accessed=now,
            access_count=1,
        )
        
        # Store in index
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO cache_entries
            (id, source, variables, lat_min, lat_max, lon_min, lon_max,
             time_start, time_end, resolution_temporal, resolution_spatial,
             file_path, file_size_mb, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id, entry.source, json.dumps(entry.variables),
            entry.lat_range[0], entry.lat_range[1],
            entry.lon_range[0], entry.lon_range[1],
            entry.time_range[0], entry.time_range[1],
            entry.resolution_temporal, entry.resolution_spatial,
            entry.file_path, entry.file_size_mb,
            entry.created_at.isoformat(), entry.last_accessed.isoformat(),
            entry.access_count,
        ))
        conn.commit()
        conn.close()
        
        print(f"💾 Cached: {source} data ({file_size:.2f} MB)")
        return entry
    
    def load(self, entry: CacheEntry) -> Any:
        """Load data from cache entry."""
        file_path = Path(entry.file_path)
        
        if not file_path.exists():
            return None
        
        # Update access stats
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE cache_entries
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """, (datetime.now().isoformat(), entry.id))
        conn.commit()
        conn.close()
        
        # Load data
        if file_path.suffix == ".nc" and HAS_XARRAY:
            return xr.open_dataset(file_path)
        else:
            with open(file_path) as f:
                return json.load(f)
    
    def list_entries(self, source: str = None) -> List[CacheEntry]:
        """List all cache entries, optionally filtered by source."""
        conn = sqlite3.connect(self.db_path)
        
        if source:
            cursor = conn.execute(
                "SELECT * FROM cache_entries WHERE source = ? ORDER BY last_accessed DESC",
                (source,)
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM cache_entries ORDER BY last_accessed DESC"
            )
        
        entries = [self._row_to_entry(row) for row in cursor.fetchall()]
        conn.close()
        return entries
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        conn = sqlite3.connect(self.db_path)
        
        cursor = conn.execute("""
            SELECT 
                source,
                COUNT(*) as count,
                SUM(file_size_mb) as total_size_mb,
                SUM(access_count) as total_accesses
            FROM cache_entries
            GROUP BY source
        """)
        
        stats = {
            "sources": {},
            "total_entries": 0,
            "total_size_mb": 0,
        }
        
        for row in cursor.fetchall():
            stats["sources"][row[0]] = {
                "count": row[1],
                "size_mb": row[2] or 0,
                "accesses": row[3] or 0,
            }
            stats["total_entries"] += row[1]
            stats["total_size_mb"] += row[2] or 0
        
        conn.close()
        return stats
    
    def clear(self, source: str = None, older_than_days: int = None):
        """Clear cache entries."""
        conn = sqlite3.connect(self.db_path)
        
        if source and older_than_days:
            cursor = conn.execute(
                "SELECT file_path FROM cache_entries WHERE source = ? AND created_at < datetime('now', ?)",
                (source, f'-{older_than_days} days')
            )
        elif source:
            cursor = conn.execute(
                "SELECT file_path FROM cache_entries WHERE source = ?",
                (source,)
            )
        elif older_than_days:
            cursor = conn.execute(
                "SELECT file_path FROM cache_entries WHERE created_at < datetime('now', ?)",
                (f'-{older_than_days} days',)
            )
        else:
            cursor = conn.execute("SELECT file_path FROM cache_entries")
        
        # Delete files
        for row in cursor.fetchall():
            try:
                Path(row[0]).unlink(missing_ok=True)
            except:
                pass
        
        # Delete from DB
        if source and older_than_days:
            conn.execute(
                "DELETE FROM cache_entries WHERE source = ? AND created_at < datetime('now', ?)",
                (source, f'-{older_than_days} days')
            )
        elif source:
            conn.execute("DELETE FROM cache_entries WHERE source = ?", (source,))
        elif older_than_days:
            conn.execute(
                "DELETE FROM cache_entries WHERE created_at < datetime('now', ?)",
                (f'-{older_than_days} days',)
            )
        else:
            conn.execute("DELETE FROM cache_entries")
        
        conn.commit()
        conn.close()
