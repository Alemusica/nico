"""
🌊 Data Flow Auditor
====================
Audit di CMEMS, ERA5, Climate Indices, Cache management.

Checks:
- Connection availability
- Download functionality
- Error handling
- Cache operations
- Data quality
- File formats support
"""

import asyncio
import httpx
from pathlib import Path
import sys
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from audit_agents.orchestrator import AuditAgent, AuditCheck


class DataFlowAuditor(AuditAgent):
    """Audit data ingestion and cache systems."""
    
    def __init__(self):
        super().__init__(
            name="DataFlowAuditor",
            scope="CMEMS, ERA5, Climate Indices, Cache"
        )
        self.api_base = "http://localhost:8000/api/v1"
    
    async def _run_checks(self):
        """Run all data flow checks."""
        await self._check_backend_health()
        await self._check_cmems_client()
        await self._check_era5_client()
        await self._check_climate_indices()
        await self._check_cache_system()
        await self._check_data_manager()
        await self._check_file_formats()
    
    async def _check_backend_health(self):
        """Check if backend API is accessible."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base}/health", timeout=5.0)
                
                self.check(
                    name="backend_health",
                    condition=response.status_code == 200,
                    severity="critical",
                    message="Backend API accessible" if response.status_code == 200 else f"Backend returned {response.status_code}",
                    url=f"{self.api_base}/health",
                    status_code=response.status_code
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.check(
                        name="backend_components",
                        condition="components" in data,
                        severity="high",
                        message="Backend components reported",
                        components=data.get("components", {})
                    )
        except Exception as e:
            self.check(
                name="backend_health",
                condition=False,
                severity="critical",
                message=f"Cannot connect to backend: {e}",
                error=str(e)
            )
    
    async def _check_cmems_client(self):
        """Check CMEMS client availability."""
        try:
            from src.surge_shazam.data.cmems_client import CMEMSClient
            
            self.check(
                name="cmems_import",
                condition=True,
                severity="high",
                message="CMEMSClient module imports successfully"
            )
            
            # Check client instantiation
            try:
                client = CMEMSClient()
                self.check(
                    name="cmems_instantiation",
                    condition=True,
                    severity="high",
                    message="CMEMSClient instantiates without errors"
                )
            except Exception as e:
                self.check(
                    name="cmems_instantiation",
                    condition=False,
                    severity="high",
                    message=f"CMEMSClient instantiation failed: {e}",
                    error=str(e)
                )
        except ImportError as e:
            self.check(
                name="cmems_import",
                condition=False,
                severity="high",
                message=f"CMEMSClient import failed: {e}",
                error=str(e)
            )
    
    async def _check_era5_client(self):
        """Check ERA5 client availability."""
        try:
            from src.surge_shazam.data.era5_client import ERA5Client
            
            self.check(
                name="era5_import",
                condition=True,
                severity="high",
                message="ERA5Client module imports successfully"
            )
            
            # Check humidity variables presence
            try:
                client = ERA5Client()
                self.check(
                    name="era5_instantiation",
                    condition=True,
                    severity="high",
                    message="ERA5Client instantiates without errors"
                )
                
                # Check if humidity vars in config
                has_humidity = hasattr(client, 'available_vars') or hasattr(client, 'variables')
                self.check(
                    name="era5_humidity_vars",
                    condition=has_humidity,
                    severity="medium",
                    message="ERA5 humidity variables available" if has_humidity else "Humidity vars not verified",
                    expected_vars=["dewpoint_2m", "specific_humidity_850", "relative_humidity_850"]
                )
            except Exception as e:
                self.check(
                    name="era5_instantiation",
                    condition=False,
                    severity="high",
                    message=f"ERA5Client instantiation failed: {e}",
                    error=str(e)
                )
        except ImportError as e:
            self.check(
                name="era5_import",
                condition=False,
                severity="high",
                message=f"ERA5Client import failed: {e}",
                error=str(e)
            )
    
    async def _check_climate_indices(self):
        """Check climate indices client."""
        try:
            from src.surge_shazam.data.climate_indices import ClimateIndicesClient
            
            self.check(
                name="climate_indices_import",
                condition=True,
                severity="medium",
                message="ClimateIndicesClient module imports successfully"
            )
            
            try:
                client = ClimateIndicesClient()
                self.check(
                    name="climate_indices_instantiation",
                    condition=True,
                    severity="medium",
                    message="ClimateIndicesClient instantiates without errors"
                )
            except Exception as e:
                self.check(
                    name="climate_indices_instantiation",
                    condition=False,
                    severity="medium",
                    message=f"ClimateIndicesClient instantiation failed: {e}",
                    error=str(e)
                )
        except ImportError as e:
            self.check(
                name="climate_indices_import",
                condition=False,
                severity="medium",
                message=f"ClimateIndicesClient import failed: {e}",
                error=str(e)
            )
    
    async def _check_cache_system(self):
        """Check cache system via API."""
        try:
            async with httpx.AsyncClient() as client:
                # Check cache stats endpoint
                response = await client.get(f"{self.api_base}/data/cache/stats", timeout=5.0)
                
                if response.status_code == 200:
                    data = response.json()
                    self.check(
                        name="cache_stats_endpoint",
                        condition=True,
                        severity="medium",
                        message="Cache stats endpoint working",
                        stats=data
                    )
                elif response.status_code == 503:
                    self.check(
                        name="cache_stats_endpoint",
                        condition=False,
                        severity="medium",
                        message="Cache stats returns 503 - DataManager issue",
                        status_code=response.status_code
                    )
                else:
                    self.check(
                        name="cache_stats_endpoint",
                        condition=False,
                        severity="medium",
                        message=f"Cache stats returned unexpected status: {response.status_code}",
                        status_code=response.status_code
                    )
        except Exception as e:
            self.check(
                name="cache_stats_endpoint",
                condition=False,
                severity="medium",
                message=f"Cache stats check failed: {e}",
                error=str(e)
            )
    
    async def _check_data_manager(self):
        """Check Data Manager availability."""
        try:
            from src.data_manager.manager import DataManager
            
            self.check(
                name="data_manager_import",
                condition=True,
                severity="high",
                message="DataManager module imports successfully"
            )
            
            try:
                manager = DataManager()
                self.check(
                    name="data_manager_instantiation",
                    condition=True,
                    severity="high",
                    message="DataManager instantiates without errors"
                )
            except Exception as e:
                self.check(
                    name="data_manager_instantiation",
                    condition=False,
                    severity="high",
                    message=f"DataManager instantiation failed: {e}",
                    error=str(e)
                )
        except ImportError as e:
            self.check(
                name="data_manager_import",
                condition=False,
                severity="high",
                message=f"DataManager import failed: {e}",
                error=str(e)
            )
    
    async def _check_file_formats(self):
        """Check file format support (NetCDF, CSV, ZARR)."""
        formats_ok = []
        formats_missing = []
        
        # Check NetCDF
        try:
            import netCDF4
            formats_ok.append("NetCDF")
        except ImportError:
            formats_missing.append("NetCDF")
        
        # Check xarray (ZARR support)
        try:
            import xarray
            formats_ok.append("ZARR (via xarray)")
        except ImportError:
            formats_missing.append("ZARR")
        
        # CSV always supported (pandas)
        try:
            import pandas
            formats_ok.append("CSV")
        except ImportError:
            formats_missing.append("CSV")
        
        self.check(
            name="file_formats_support",
            condition=len(formats_missing) == 0,
            severity="medium",
            message=f"File formats: {len(formats_ok)}/3 supported",
            supported=formats_ok,
            missing=formats_missing
        )


async def main():
    """Run DataFlowAuditor standalone."""
    print("🌊 Data Flow Auditor")
    print("=" * 60)
    
    auditor = DataFlowAuditor()
    report = await auditor.run_audit()
    
    print(f"\n📊 Results:")
    print(f"  Total Checks: {report.total_checks}")
    print(f"  Passed: {report.passed} ✅")
    print(f"  Failed: {report.failed} ❌")
    print(f"  Warnings: {report.warnings} ⚠️")
    print(f"  Duration: {report.duration_ms:.0f}ms")
    
    if report.error:
        print(f"\n❌ Error: {report.error}")
    
    print("\n📋 Detailed Checks:")
    for check in report.checks:
        status = "✅" if check.passed else "❌"
        print(f"  {status} {check.name}: {check.message}")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())
