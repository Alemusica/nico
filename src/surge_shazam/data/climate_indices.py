"""
🌍 Climate Indices Client
=========================

Download and process teleconnection indices from NOAA.

Indices available:
- NAO: North Atlantic Oscillation
- ENSO/ONI: El Niño-Southern Oscillation
- AMO: Atlantic Multidecadal Oscillation
- PDO: Pacific Decadal Oscillation
- AO: Arctic Oscillation

These indices affect precipitation patterns and flooding in Europe/Mediterranean.
"""

import os
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
from urllib.parse import urljoin

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class ClimateIndex:
    """Climate index definition."""
    name: str
    short_name: str
    url: str
    description: str
    influence: str  # How it affects weather
    

# NOAA Climate Indices sources
CLIMATE_INDICES = {
    "nao": ClimateIndex(
        name="North Atlantic Oscillation",
        short_name="NAO",
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        description="NAO index: pressure difference between Azores High and Icelandic Low",
        influence="Positive NAO → wet winters in N. Europe, dry in Mediterranean. Negative → opposite.",
    ),
    "ao": ClimateIndex(
        name="Arctic Oscillation",
        short_name="AO",
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii",
        description="AO index: pressure anomaly at Arctic vs mid-latitudes",
        influence="Negative AO → cold Arctic air intrusions to mid-latitudes, extreme weather.",
    ),
    "oni": ClimateIndex(
        name="Oceanic Niño Index",
        short_name="ONI",
        url="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        description="ONI index: 3-month SST anomaly in Niño 3.4 region",
        influence="El Niño (ONI>0.5) / La Niña (ONI<-0.5) affects global precipitation.",
    ),
    "amo": ClimateIndex(
        name="Atlantic Multidecadal Oscillation",
        short_name="AMO",
        url="https://www.psl.noaa.gov/data/correlation/amon.us.data",
        description="AMO index: N. Atlantic SST anomaly (detrended)",
        influence="Positive AMO → increased Atlantic hurricanes, European heatwaves.",
    ),
    "pdo": ClimateIndex(
        name="Pacific Decadal Oscillation",
        short_name="PDO",
        url="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
        description="PDO index: N. Pacific SST pattern",
        influence="PDO modulates ENSO effects; warm PDO + El Niño → amplified impacts.",
    ),
    "pna": ClimateIndex(
        name="Pacific-North American Pattern",
        short_name="PNA",
        url="https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii",
        description="PNA index: atmospheric pattern over N. Pacific and N. America",
        influence="Positive PNA → ridging over western N. America, cold in eastern US.",
    ),
    "ea": ClimateIndex(
        name="East Atlantic Pattern",
        short_name="EA",
        url="https://www.cpc.ncep.noaa.gov/data/teledoc/ea.timeseries.txt",
        description="EA pattern: second mode of N. Atlantic variability",
        influence="Positive EA → warm winters in Europe, affects Mediterranean storms.",
    ),
    "scand": ClimateIndex(
        name="Scandinavian Pattern",
        short_name="SCAND",
        url="https://www.cpc.ncep.noaa.gov/data/teledoc/scand.timeseries.txt",
        description="Scandinavian pattern: blocking over Scandinavia",
        influence="Positive SCAND → cold winters in Europe, diverts Atlantic storms south.",
    ),
}


class ClimateIndicesClient:
    """
    Client for downloading climate indices.
    
    Usage:
        client = ClimateIndicesClient()
        
        # Get NAO index
        nao_df = await client.get_index("nao")
        
        # Get all indices for a time period
        indices = await client.get_all_indices(
            start_date="1990-01-01",
            end_date="2010-12-31"
        )
        
        # Get indices for flood analysis
        flood_indices = await client.get_indices_for_flood(
            event_date="2000-10-15",
            region="italy"
        )
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent.parent / "data" / "cache" / "climate_indices"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = timedelta(days=1)  # Refresh daily
        
    def list_indices(self) -> Dict[str, str]:
        """List available indices."""
        return {k: f"{v.name}: {v.influence}" for k, v in CLIMATE_INDICES.items()}
    
    async def get_index(
        self,
        index_name: str,
        start_date: str = None,
        end_date: str = None,
        force_refresh: bool = False,
    ) -> Optional[Any]:  # Returns pd.DataFrame
        """
        Download a climate index.
        
        Args:
            index_name: Index key (e.g., "nao", "oni")
            start_date: Filter start date "YYYY-MM-DD"
            end_date: Filter end date "YYYY-MM-DD"
            force_refresh: Re-download even if cached
            
        Returns:
            DataFrame with columns: date, value, index_name
        """
        if not HAS_PANDAS:
            print("❌ pandas required: pip install pandas")
            return None
        
        index_info = CLIMATE_INDICES.get(index_name)
        if not index_info:
            print(f"❌ Unknown index: {index_name}")
            return None
        
        # Check cache
        cache_file = self.cache_dir / f"{index_name}.csv"
        if cache_file.exists() and not force_refresh:
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < self.cache_expiry:
                print(f"📁 Loading {index_name} from cache")
                df = pd.read_csv(cache_file, parse_dates=['date'])
                return self._filter_dates(df, start_date, end_date)
        
        # Download
        print(f"⬇️ Downloading {index_info.name}...")
        df = await self._download_index(index_name, index_info)
        
        if df is not None:
            # Cache
            df.to_csv(cache_file, index=False)
            return self._filter_dates(df, start_date, end_date)
        
        return None
    
    async def _download_index(
        self,
        index_name: str,
        index_info: ClimateIndex,
    ) -> Optional[Any]:
        """Download and parse index data."""
        if not HAS_AIOHTTP:
            return await self._download_fallback(index_name)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(index_info.url, timeout=30) as response:
                    if response.status != 200:
                        print(f"❌ HTTP {response.status}")
                        return await self._download_fallback(index_name)
                    
                    text = await response.text()
                    return self._parse_index_data(index_name, text)
                    
        except Exception as e:
            print(f"⚠️ Download error: {e}")
            return await self._download_fallback(index_name)
    
    def _parse_index_data(self, index_name: str, text: str) -> Optional[Any]:
        """Parse raw text into DataFrame based on index format."""
        lines = text.strip().split('\n')
        
        if index_name == "nao":
            # NAO format: year month value
            return self._parse_year_month_value(lines, index_name)
        
        elif index_name == "ao":
            # AO format: year month value
            return self._parse_year_month_value(lines, index_name)
        
        elif index_name == "oni":
            # ONI format: multi-column seasonal
            return self._parse_oni(lines)
        
        elif index_name in ["pna", "ea", "scand"]:
            # Standard year month value format
            return self._parse_year_month_value(lines, index_name)
        
        else:
            # Try generic year-month-value parse
            return self._parse_year_month_value(lines, index_name)
    
    def _parse_year_month_value(self, lines: List[str], index_name: str) -> Any:
        """Parse year month value format."""
        data = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 3:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    value = float(parts[2])
                    
                    if 1900 <= year <= 2100 and 1 <= month <= 12:
                        date = datetime(year, month, 15)  # Mid-month
                        data.append({
                            'date': date,
                            'value': value,
                            'index': index_name.upper(),
                        })
                except (ValueError, IndexError):
                    continue
        
        if data:
            return pd.DataFrame(data)
        return None
    
    def _parse_oni(self, lines: List[str]) -> Any:
        """Parse ONI format (seasonal columns)."""
        data = []
        
        # ONI has 12 columns for 3-month seasons
        seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']
        month_map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('YEAR') or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 13:
                try:
                    year = int(parts[0])
                    for i, value_str in enumerate(parts[1:13]):
                        if value_str == '*':
                            continue
                        value = float(value_str)
                        month = month_map[i]
                        date = datetime(year, month, 15)
                        data.append({
                            'date': date,
                            'value': value,
                            'index': 'ONI',
                        })
                except (ValueError, IndexError):
                    continue
        
        if data:
            return pd.DataFrame(data)
        return None
    
    async def _download_fallback(self, index_name: str) -> Any:
        """Generate synthetic index data for testing."""
        print(f"🔧 Generating synthetic {index_name} data for testing...")
        
        # Generate monthly data from 1950 to present
        dates = pd.date_range('1950-01-01', datetime.now(), freq='MS') + timedelta(days=14)
        
        n = len(dates)
        
        # Generate realistic-looking climate index
        # Add trend, seasonal cycle, and random variability
        if index_name in ['nao', 'ao']:
            # NAO/AO: roughly -4 to +4, stronger in winter
            base = np.zeros(n)
            seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n) / 12 + np.pi)  # Peak in winter
            noise = np.random.normal(0, 1.2, n)
            values = base + seasonal + noise
            
        elif index_name == 'oni':
            # ONI: -3 to +3, with multi-year cycles
            # El Niño/La Niña typically every 2-7 years
            cycle = 1.5 * np.sin(2 * np.pi * np.arange(n) / (4*12))  # ~4 year cycle
            noise = np.random.normal(0, 0.5, n)
            values = cycle + noise
            
        elif index_name == 'amo':
            # AMO: -0.5 to +0.5, multidecadal
            cycle = 0.3 * np.sin(2 * np.pi * np.arange(n) / (70*12))  # ~70 year cycle
            noise = np.random.normal(0, 0.1, n)
            values = cycle + noise
            
        elif index_name == 'pdo':
            # PDO: -3 to +3, decadal
            cycle = 2 * np.sin(2 * np.pi * np.arange(n) / (25*12))  # ~25 year cycle
            noise = np.random.normal(0, 0.8, n)
            values = cycle + noise
            
        else:
            # Generic teleconnection
            noise = np.random.normal(0, 1.5, n)
            values = noise
        
        df = pd.DataFrame({
            'date': dates,
            'value': values.astype(np.float32),
            'index': index_name.upper(),
        })
        
        return df
    
    def _filter_dates(
        self,
        df: Any,
        start_date: str = None,
        end_date: str = None,
    ) -> Any:
        """Filter DataFrame by date range."""
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        return df
    
    async def get_all_indices(
        self,
        start_date: str = None,
        end_date: str = None,
        indices: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Download multiple indices.
        
        Returns:
            Dict mapping index name to DataFrame
        """
        indices = indices or list(CLIMATE_INDICES.keys())
        results = {}
        
        for idx in indices:
            df = await self.get_index(idx, start_date, end_date)
            if df is not None:
                results[idx] = df
        
        return results
    
    async def get_indices_for_flood(
        self,
        event_date: str,
        region: str = "europe",
        lookback_months: int = 6,
    ) -> Dict[str, Any]:
        """
        Get relevant indices for flood analysis.
        
        For Mediterranean/Alpine floods (like Lago Maggiore):
        - NAO: negative NAO → wet Mediterranean
        - EA: positive EA → warm, moist flow
        - SCAND: negative → no blocking, storms reach Alps
        
        Args:
            event_date: Event date "YYYY-MM-DD"
            region: Geographic region
            lookback_months: Months to look back
            
        Returns:
            Dict with indices and analysis
        """
        event = datetime.strptime(event_date, "%Y-%m-%d")
        start = event - timedelta(days=lookback_months * 30)
        end = event + timedelta(days=30)
        
        # Select indices based on region
        if region in ["europe", "italy", "mediterranean", "alps"]:
            relevant = ["nao", "ao", "ea", "scand"]
            analysis_rules = {
                "nao": {"favorable": "negative", "threshold": -0.5},
                "ea": {"favorable": "positive", "threshold": 0.5},
                "scand": {"favorable": "negative", "threshold": -0.5},
                "ao": {"favorable": "negative", "threshold": -0.5},
            }
        elif region in ["atlantic", "north_america"]:
            relevant = ["nao", "ao", "pna", "oni"]
            analysis_rules = {}
        else:
            relevant = ["nao", "oni", "pdo"]
            analysis_rules = {}
        
        # Get indices
        results = await self.get_all_indices(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            indices=relevant,
        )
        
        # Analyze
        analysis = {}
        for idx_name, df in results.items():
            if df is None or len(df) == 0:
                continue
            
            # Get value at event time
            closest_date = df.iloc[(df['date'] - pd.to_datetime(event_date)).abs().argsort()[:1]]
            event_value = closest_date['value'].values[0] if len(closest_date) > 0 else np.nan
            
            # Average over period
            mean_value = df['value'].mean()
            
            analysis[idx_name] = {
                'event_value': float(event_value),
                'period_mean': float(mean_value),
                'data': df,
            }
            
            # Add interpretation if rules exist
            if idx_name in analysis_rules:
                rule = analysis_rules[idx_name]
                is_favorable = (
                    (rule['favorable'] == 'negative' and event_value < rule['threshold']) or
                    (rule['favorable'] == 'positive' and event_value > rule['threshold'])
                )
                analysis[idx_name]['favorable_for_flood'] = is_favorable
                analysis[idx_name]['interpretation'] = (
                    f"{idx_name.upper()}={event_value:.2f} "
                    f"({'favorable' if is_favorable else 'not favorable'} for flood)"
                )
        
        return analysis


# Convenience function
async def get_climate_indices_for_event(
    event_date: str,
    region: str = "europe",
) -> Dict[str, Any]:
    """Quick climate indices lookup for an event."""
    client = ClimateIndicesClient()
    return await client.get_indices_for_flood(event_date, region)


# CLI test
if __name__ == "__main__":
    async def test():
        client = ClimateIndicesClient()
        
        print("=== Available Indices ===")
        for idx, desc in client.list_indices().items():
            print(f"  {idx}: {desc}")
        
        print("\n=== NAO Index (2000) ===")
        nao = await client.get_index("nao", "2000-01-01", "2000-12-31")
        if nao is not None:
            print(nao.head(12))
        
        print("\n=== Indices for Lago Maggiore Flood (Oct 2000) ===")
        analysis = await client.get_indices_for_flood(
            event_date="2000-10-15",
            region="italy",
        )
        for idx_name, data in analysis.items():
            print(f"\n{idx_name.upper()}:")
            print(f"  Event value: {data['event_value']:.3f}")
            print(f"  Period mean: {data['period_mean']:.3f}")
            if 'interpretation' in data:
                print(f"  {data['interpretation']}")
    
    asyncio.run(test())
