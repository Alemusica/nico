#!/usr/bin/env python3
"""
🔬 Knowledge Base Backend Comparison Test

Compares Neo4j and SurrealDB for:
- Write performance (papers, events, patterns)
- Read performance (single, bulk)
- Search performance (vector, text, graph traversal)
- Connection stability

Run: python tests/test_backends.py
"""

import asyncio
import time
import random
import statistics
from dataclasses import dataclass
from typing import Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.knowledge_service import (
    create_knowledge_service,
    Paper,
    HistoricalEvent,
    ClimateIndex,
    CausalPattern,
)


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    operation: str
    backend: str
    count: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    ops_per_second: float
    success: bool
    error: Optional[str] = None


def generate_test_papers(n: int) -> list[Paper]:
    """Generate test paper data."""
    journals = [
        "Nature", "Science", "JGR Oceans", "Deep-Sea Research", 
        "Ocean Modelling", "Climate Dynamics", "GRL"
    ]
    keywords_pool = [
        "NAO", "AMOC", "SST", "SSH", "Arctic", "Atlantic",
        "warming", "circulation", "variability", "teleconnection"
    ]
    
    papers = []
    for i in range(n):
        papers.append(Paper(
            title=f"Test Paper {i}: Oceanographic Analysis of Pattern {i % 10}",
            authors=[f"Author {j}" for j in range(random.randint(1, 5))],
            abstract=f"This paper investigates the causal relationship between variable {i % 5} and outcome {i % 3}. "
                     f"We use advanced statistical methods including PCMCI to identify time-lagged dependencies.",
            doi=f"10.1234/test.{i:06d}",
            year=random.randint(1990, 2024),
            journal=random.choice(journals),
            keywords=random.sample(keywords_pool, k=random.randint(2, 5)),
            embedding=[random.uniform(-1, 1) for _ in range(128)],  # Smaller for testing
        ))
    return papers


def generate_test_events(n: int) -> list[HistoricalEvent]:
    """Generate test historical events."""
    from datetime import datetime, timedelta
    
    event_types = ["flood", "drought", "storm", "heatwave", "cold_snap"]
    
    events = []
    base_date = datetime(1950, 1, 1)
    for i in range(n):
        start = base_date + timedelta(days=random.randint(0, 25000))
        events.append(HistoricalEvent(
            name=f"Event {i}: {random.choice(event_types).title()} {start.year}",
            description=f"Significant oceanographic event affecting region {i % 10}",
            event_type=random.choice(event_types),
            start_date=start,
            end_date=start + timedelta(days=random.randint(1, 30)),
            location={"region": f"Region_{i % 5}", "lat": random.uniform(-60, 80), "lon": random.uniform(-180, 180)},
            severity=random.uniform(0.3, 1.0),
            source="NOAA",
        ))
    return events


def generate_test_patterns(n: int) -> list[CausalPattern]:
    """Generate test causal patterns."""
    variables_pool = ["SST", "SSH", "NAO", "AMO", "Wind", "Precipitation", "SLA", "Current"]
    pattern_types = ["teleconnection", "feedback", "forcing", "response", "oscillation"]
    
    patterns = []
    for i in range(n):
        vars = random.sample(variables_pool, k=random.randint(2, 4))
        patterns.append(CausalPattern(
            name=f"Pattern {i}: {vars[0]} → {vars[-1]}",
            description=f"Causal relationship from {vars[0]} to {vars[-1]} with intermediate effects",
            pattern_type=random.choice(pattern_types),
            variables=vars,
            lag_days=random.randint(0, 30),
            strength=random.uniform(0.3, 0.95),
            confidence=random.uniform(0.5, 0.99),
            metadata={"method": "PCMCI", "alpha": 0.05},
        ))
    return patterns


async def benchmark_writes(
    service,
    backend: str,
    papers: list[Paper],
    events: list[HistoricalEvent],
    patterns: list[CausalPattern],
) -> list[BenchmarkResult]:
    """Benchmark write operations."""
    results = []
    
    # Papers
    print(f"  [{backend}] Writing {len(papers)} papers...")
    times = []
    try:
        for paper in papers:
            start = time.perf_counter()
            await service.add_paper(paper)
            times.append((time.perf_counter() - start) * 1000)
        
        results.append(BenchmarkResult(
            operation="write_papers",
            backend=backend,
            count=len(papers),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            ops_per_second=len(papers) / (sum(times) / 1000),
            success=True,
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            operation="write_papers",
            backend=backend,
            count=0,
            total_time_ms=0,
            avg_time_ms=0,
            min_time_ms=0,
            max_time_ms=0,
            ops_per_second=0,
            success=False,
            error=str(e),
        ))
    
    # Events
    print(f"  [{backend}] Writing {len(events)} events...")
    times = []
    try:
        for event in events:
            start = time.perf_counter()
            await service.add_event(event)
            times.append((time.perf_counter() - start) * 1000)
        
        results.append(BenchmarkResult(
            operation="write_events",
            backend=backend,
            count=len(events),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            ops_per_second=len(events) / (sum(times) / 1000),
            success=True,
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            operation="write_events",
            backend=backend,
            count=0,
            total_time_ms=0,
            avg_time_ms=0,
            min_time_ms=0,
            max_time_ms=0,
            ops_per_second=0,
            success=False,
            error=str(e),
        ))
    
    # Patterns
    print(f"  [{backend}] Writing {len(patterns)} patterns...")
    times = []
    try:
        for pattern in patterns:
            start = time.perf_counter()
            await service.add_pattern(pattern)
            times.append((time.perf_counter() - start) * 1000)
        
        results.append(BenchmarkResult(
            operation="write_patterns",
            backend=backend,
            count=len(patterns),
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            ops_per_second=len(patterns) / (sum(times) / 1000),
            success=True,
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            operation="write_patterns",
            backend=backend,
            count=0,
            total_time_ms=0,
            avg_time_ms=0,
            min_time_ms=0,
            max_time_ms=0,
            ops_per_second=0,
            success=False,
            error=str(e),
        ))
    
    return results


async def benchmark_reads(service, backend: str, n_queries: int = 20) -> list[BenchmarkResult]:
    """Benchmark read/search operations."""
    results = []
    
    # Paper search
    print(f"  [{backend}] Running {n_queries} paper searches...")
    queries = ["ocean", "climate", "NAO", "warming", "variability"]
    times = []
    try:
        for _ in range(n_queries):
            query = random.choice(queries)
            start = time.perf_counter()
            await service.search_papers(query=query, limit=10)
            times.append((time.perf_counter() - start) * 1000)
        
        results.append(BenchmarkResult(
            operation="search_papers",
            backend=backend,
            count=n_queries,
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            ops_per_second=n_queries / (sum(times) / 1000),
            success=True,
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            operation="search_papers",
            backend=backend,
            count=0,
            total_time_ms=0,
            avg_time_ms=0,
            min_time_ms=0,
            max_time_ms=0,
            ops_per_second=0,
            success=False,
            error=str(e),
        ))
    
    # Pattern search
    print(f"  [{backend}] Running {n_queries} pattern searches...")
    pattern_types = ["teleconnection", "feedback", "forcing"]
    times = []
    try:
        for _ in range(n_queries):
            start = time.perf_counter()
            await service.search_patterns(
                pattern_type=random.choice(pattern_types),
                min_confidence=0.5,
                limit=10,
            )
            times.append((time.perf_counter() - start) * 1000)
        
        results.append(BenchmarkResult(
            operation="search_patterns",
            backend=backend,
            count=n_queries,
            total_time_ms=sum(times),
            avg_time_ms=statistics.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            ops_per_second=n_queries / (sum(times) / 1000),
            success=True,
        ))
    except Exception as e:
        results.append(BenchmarkResult(
            operation="search_patterns",
            backend=backend,
            count=0,
            total_time_ms=0,
            avg_time_ms=0,
            min_time_ms=0,
            max_time_ms=0,
            ops_per_second=0,
            success=False,
            error=str(e),
        ))
    
    # Statistics
    print(f"  [{backend}] Getting statistics...")
    try:
        start = time.perf_counter()
        stats = await service.get_statistics()
        time_ms = (time.perf_counter() - start) * 1000
        
        results.append(BenchmarkResult(
            operation="get_statistics",
            backend=backend,
            count=1,
            total_time_ms=time_ms,
            avg_time_ms=time_ms,
            min_time_ms=time_ms,
            max_time_ms=time_ms,
            ops_per_second=1000 / time_ms,
            success=True,
        ))
        print(f"    Stats: {stats}")
    except Exception as e:
        results.append(BenchmarkResult(
            operation="get_statistics",
            backend=backend,
            count=0,
            total_time_ms=0,
            avg_time_ms=0,
            min_time_ms=0,
            max_time_ms=0,
            ops_per_second=0,
            success=False,
            error=str(e),
        ))
    
    return results


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a nice table."""
    print("\n" + "=" * 100)
    print("📊 BENCHMARK RESULTS")
    print("=" * 100)
    
    # Group by operation
    by_operation = {}
    for r in results:
        if r.operation not in by_operation:
            by_operation[r.operation] = {}
        by_operation[r.operation][r.backend] = r
    
    print(f"\n{'Operation':<20} {'Backend':<12} {'Count':>8} {'Total (ms)':>12} {'Avg (ms)':>10} {'Ops/sec':>10} {'Status':>10}")
    print("-" * 100)
    
    for operation, backends in by_operation.items():
        for backend, r in backends.items():
            status = "✅ OK" if r.success else f"❌ {r.error[:20] if r.error else 'Failed'}"
            print(f"{operation:<20} {backend:<12} {r.count:>8} {r.total_time_ms:>12.2f} {r.avg_time_ms:>10.2f} {r.ops_per_second:>10.1f} {status:>10}")
        print()
    
    # Winner summary
    print("\n" + "=" * 100)
    print("🏆 COMPARISON SUMMARY")
    print("=" * 100)
    
    for operation, backends in by_operation.items():
        if len(backends) == 2 and all(r.success for r in backends.values()):
            neo4j = backends.get('neo4j')
            surreal = backends.get('surrealdb')
            
            if neo4j and surreal:
                if neo4j.avg_time_ms < surreal.avg_time_ms:
                    winner = "Neo4j"
                    ratio = surreal.avg_time_ms / neo4j.avg_time_ms
                else:
                    winner = "SurrealDB"
                    ratio = neo4j.avg_time_ms / surreal.avg_time_ms
                
                print(f"  {operation:<20}: {winner} is {ratio:.1f}x faster")


async def main():
    """Run the benchmark."""
    print("=" * 60)
    print("🔬 KNOWLEDGE BASE BACKEND BENCHMARK")
    print("   Neo4j vs SurrealDB for Oceanography")
    print("=" * 60)
    
    # Generate test data
    print("\n📦 Generating test data...")
    n_items = 50  # Adjust for longer/shorter tests
    papers = generate_test_papers(n_items)
    events = generate_test_events(n_items)
    patterns = generate_test_patterns(n_items)
    print(f"   Generated {len(papers)} papers, {len(events)} events, {len(patterns)} patterns")
    
    all_results = []
    
    # Test each backend
    for backend in ["neo4j", "surrealdb"]:
        print(f"\n{'='*60}")
        print(f"🔌 Testing {backend.upper()}")
        print("="*60)
        
        try:
            service = create_knowledge_service(backend)
            await service.connect()
            print(f"   Connected to {backend}")
            
            # Write benchmark
            print("\n📝 Write Operations:")
            write_results = await benchmark_writes(
                service, backend, papers, events, patterns
            )
            all_results.extend(write_results)
            
            # Read benchmark
            print("\n📖 Read Operations:")
            read_results = await benchmark_reads(service, backend)
            all_results.extend(read_results)
            
            await service.disconnect()
            print(f"   Disconnected from {backend}")
            
        except Exception as e:
            print(f"   ❌ Failed to connect: {e}")
            # Add failure results
            for op in ["write_papers", "write_events", "write_patterns", 
                       "search_papers", "search_patterns", "get_statistics"]:
                all_results.append(BenchmarkResult(
                    operation=op,
                    backend=backend,
                    count=0,
                    total_time_ms=0,
                    avg_time_ms=0,
                    min_time_ms=0,
                    max_time_ms=0,
                    ops_per_second=0,
                    success=False,
                    error=str(e),
                ))
    
    # Print results
    print_results(all_results)
    
    print("\n✨ Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(main())
