#!/usr/bin/env python3
"""
🧪 Investigation Agent Test Runner
===================================

Run tests for the Investigation Agent system.

Usage:
    # Run all tests
    python tests/run_investigation_tests.py
    
    # Run specific module
    python tests/run_investigation_tests.py geo
    python tests/run_investigation_tests.py cmems
    python tests/run_investigation_tests.py era5
    python tests/run_investigation_tests.py climate
    python tests/run_investigation_tests.py literature
    python tests/run_investigation_tests.py agent
    
    # Run with verbose output
    python tests/run_investigation_tests.py -v
    
    # Run and show print statements
    python tests/run_investigation_tests.py -s
    
    # Run full integration test
    python tests/run_investigation_tests.py full
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Test modules
TEST_MODULES = {
    "geo": "test_investigation/test_geo_resolver.py",
    "cmems": "test_investigation/test_cmems_client.py",
    "era5": "test_investigation/test_era5_client.py",
    "climate": "test_investigation/test_climate_indices.py",
    "literature": "test_investigation/test_literature_scraper.py",
    "agent": "test_investigation/test_investigation_agent.py",
    "all": "test_investigation/",
}


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"🧪 {title}")
    print("=" * 60)


def run_quick_check():
    """Quick import check before running tests."""
    print_header("Quick Import Check")
    
    modules_ok = True
    
    # Check each module can be imported
    checks = [
        ("agent.tools.geo_resolver", "GeoResolver"),
        ("surge_shazam.data.cmems_client", "CMEMSClient"),
        ("surge_shazam.data.era5_client", "ERA5Client"),
        ("surge_shazam.data.climate_indices", "ClimateIndicesClient"),
        ("agent.tools.literature_scraper", "LiteratureScraper"),
        ("agent.investigation_agent", "InvestigationAgent"),
    ]
    
    for module_path, class_name in checks:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✅ {module_path}.{class_name}")
        except ImportError as e:
            print(f"  ❌ {module_path}.{class_name}: {e}")
            modules_ok = False
        except Exception as e:
            print(f"  ⚠️ {module_path}.{class_name}: {e}")
            modules_ok = False
    
    return modules_ok


def run_tests(module: str = "all", verbose: bool = False, show_output: bool = False):
    """Run tests with pytest."""
    test_path = TEST_MODULES.get(module, module)
    
    if not test_path.startswith("test_investigation"):
        test_path = f"test_investigation/{test_path}"
    
    full_path = PROJECT_ROOT / "tests" / test_path
    
    print_header(f"Running Tests: {test_path}")
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", str(full_path)]
    
    if verbose:
        cmd.append("-v")
    
    if show_output:
        cmd.append("-s")
    
    # Add color output
    cmd.append("--tb=short")
    
    # Run pytest
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def run_full_integration():
    """Run full integration test with Lago Maggiore example."""
    print_header("Full Integration Test: Lago Maggiore 2000")
    
    import asyncio
    
    async def test_integration():
        from agent.investigation_agent import InvestigationAgent
        
        agent = InvestigationAgent()
        
        print("\n🔍 Running investigation: 'alluvioni Lago Maggiore 2000'")
        print("-" * 50)
        
        result = await agent.investigate(
            "analizza le alluvioni del Lago Maggiore nell'ottobre 2000",
            collect_satellite=True,
            collect_reanalysis=True,
            collect_climate_indices=True,
            collect_papers=True,
            run_correlation=True,
            expand_search=True,
        )
        
        print("\n" + "=" * 50)
        print("📋 INVESTIGATION RESULTS")
        print("=" * 50)
        
        print(f"\n📍 Location: {result.event_context.location_name}")
        print(f"📅 Period: {result.event_context.start_date} to {result.event_context.end_date}")
        
        print(f"\n📊 Data Sources ({len(result.data_sources)}):")
        for source in result.data_sources:
            print(f"   - {source.name} ({source.source_type}, quality: {source.quality})")
        
        print(f"\n🔗 Correlations ({len(result.correlations)}):")
        for corr in result.correlations:
            if corr.get('type') == 'climate_index':
                print(f"   - {corr.get('interpretation', corr)}")
        
        print(f"\n📚 Papers Found: {len(result.papers)}")
        for paper in result.papers[:3]:
            print(f"   - {paper.get('title', 'Unknown')[:60]}...")
        
        print(f"\n📍 Related Events: {len(result.related_events)}")
        for event in result.related_events[:5]:
            print(f"   - {event}")
        
        print(f"\n💡 Key Findings ({len(result.key_findings)}):")
        for finding in result.key_findings:
            print(f"   • {finding}")
        
        print(f"\n📝 Recommendations ({len(result.recommendations)}):")
        for rec in result.recommendations:
            print(f"   → {rec}")
        
        print(f"\n🎯 Confidence: {result.confidence:.0%}")
        
        return result
    
    try:
        result = asyncio.run(test_integration())
        return 0 if result else 1
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Investigation Agent tests")
    parser.add_argument(
        "module",
        nargs="?",
        default="all",
        choices=list(TEST_MODULES.keys()) + ["full", "check"],
        help="Test module to run (default: all)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-s", "--show-output", action="store_true", help="Show print statements")
    
    args = parser.parse_args()
    
    print("\n" + "🧪" * 30)
    print("   INVESTIGATION AGENT TEST SUITE")
    print("🧪" * 30)
    
    # Quick import check
    if not run_quick_check():
        print("\n⚠️ Some imports failed. Check module paths.")
        # Continue anyway to see detailed errors
    
    # Run requested tests
    if args.module == "check":
        return 0  # Just the import check
    elif args.module == "full":
        return run_full_integration()
    else:
        return run_tests(args.module, args.verbose, args.show_output)


if __name__ == "__main__":
    sys.exit(main())
