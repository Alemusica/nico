#!/usr/bin/env python3
"""
Prefill CMEMS L4 Cache
======================
Pre-download CMEMS L4 data for Fram and Davis divided gates (1993-2020).

Gates to cache:
    - Fram Strait WEST (Greenland side)
    - Fram Strait EAST (Svalbard side)
    - Davis Strait WEST (Baffin Island side)
    - Davis Strait EAST (Greenland side)

Usage:
    source .venv/bin/activate
    python scripts/prefill_cmems_cache.py
    
Or with custom date range:
    python scripts/prefill_cmems_cache.py --start 1993-01-01 --end 2020-12-31

Time estimate: ~15-30 minutes total (4 gates × ~4-8 min each)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.cmems_l4_service import CMEMSL4Service, CMEMSL4Config
from src.services.gate_service import load_gate_config
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# Target gates to prefill
TARGET_GATES = [
    "fram_strait_west",
    "fram_strait_east", 
    "davis_strait_west",
    "davis_strait_east",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prefill CMEMS L4 cache for divided gates"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="1993-01-01",
        help="Start date (YYYY-MM-DD). Default: 1993-01-01"
    )
    parser.add_argument(
        "--end", 
        type=str,
        default="2020-12-31",
        help="End date (YYYY-MM-DD). Default: 2020-12-31"
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["adt", "ugos", "vgos"],
        help="Variables to download. Default: adt ugos vgos"
    )
    parser.add_argument(
        "--gates",
        type=str,
        nargs="+",
        default=TARGET_GATES,
        help=f"Gates to prefill. Default: {' '.join(TARGET_GATES)}"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=50.0,
        help="Buffer around gate in km. Default: 50.0"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading"
    )
    
    return parser.parse_args()


def prefill_gate(
    gate_name: str,
    start_date: str,
    end_date: str,
    variables: list[str],
    buffer_km: float,
    use_cache: bool = True,
    dry_run: bool = False
):
    """
    Prefill cache for a single gate.
    
    Parameters
    ----------
    gate_name : str
        Gate identifier (e.g., "fram_strait_west")
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    variables : list[str]
        Variables to download
    buffer_km : float
        Buffer around gate in km
    use_cache : bool
        Whether to use cache
    dry_run : bool
        If True, only show what would be downloaded
    """
    logger.info("=" * 80)
    logger.info(f"Processing: {gate_name}")
    logger.info("=" * 80)
    
    try:
        # Load gate config
        gate_config = load_gate_config(gate_name)
        if not gate_config:
            logger.error(f"Gate '{gate_name}' not found in gates.yaml")
            return False
            
        logger.info(f"Gate: {gate_config['name']}")
        logger.info(f"Region: {gate_config.get('region', 'Unknown')}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Variables: {', '.join(variables)}")
        logger.info(f"Buffer: {buffer_km} km")
        
        if dry_run:
            logger.info("[DRY RUN] Would download data but skipping...")
            return True
        
        # Initialize service
        service = CMEMSL4Service()
        
        # Create config
        config = CMEMSL4Config(
            gate_name=gate_name,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            buffer_km=buffer_km,
            use_cache=use_cache
        )
        
        # Load data (will download and cache if not present)
        logger.info("🔄 Loading/downloading data...")
        start_time = datetime.now()
        
        data = service.load_gate_data(config)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Completed in {elapsed:.1f}s")
        
        # Show data info
        if data:
            logger.info(f"📊 Data shape: {data.adt_matrix.shape if data.adt_matrix is not None else 'N/A'}")
            logger.info(f"📍 Gate points: {len(data.gate_lon_pts)}")
            logger.info(f"📅 Time steps: {len(data.time_array)}")
            logger.info(f"🔬 Native resolution: {data.native_resolution_km:.2f} km")
            logger.info(f"📏 Effective spacing: {data.effective_spacing_km:.2f} km")
        
        # Show cache stats
        stats = service.get_cache_stats()
        if stats:
            logger.info(f"💾 Cache stats: L1={stats['l1_hits']}/{stats['l1_total']} hits, "
                       f"L2={stats['l2_hits']}/{stats['l2_total']} hits")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process {gate_name}: {e}", exc_info=True)
        return False


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("CMEMS L4 CACHE PREFILL")
    print("=" * 80)
    print(f"Date range: {args.start} to {args.end}")
    print(f"Variables: {', '.join(args.variables)}")
    print(f"Gates: {', '.join(args.gates)}")
    print(f"Buffer: {args.buffer} km")
    print(f"Cache: {'DISABLED' if args.no_cache else 'ENABLED'}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("=" * 80 + "\n")
    
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No data will be downloaded\n")
    
    # Process each gate
    results = {}
    total_start = datetime.now()
    
    for i, gate_name in enumerate(args.gates, 1):
        print(f"\n[{i}/{len(args.gates)}] Processing {gate_name}...")
        
        success = prefill_gate(
            gate_name=gate_name,
            start_date=args.start,
            end_date=args.end,
            variables=args.variables,
            buffer_km=args.buffer,
            use_cache=not args.no_cache,
            dry_run=args.dry_run
        )
        
        results[gate_name] = "✅ SUCCESS" if success else "❌ FAILED"
    
    total_elapsed = (datetime.now() - total_start).total_seconds()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for gate_name, status in results.items():
        print(f"{status:12} {gate_name}")
    
    success_count = sum(1 for s in results.values() if "SUCCESS" in s)
    print(f"\nCompleted: {success_count}/{len(args.gates)} gates")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed:.1f}s)")
    
    if success_count == len(args.gates):
        print("\n🎉 All gates successfully cached!")
    else:
        print(f"\n⚠️  {len(args.gates) - success_count} gate(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
