#!/usr/bin/env python
"""
🔄 Data Intelligence Pipeline - Demo Script
============================================

Demonstrates the full pipeline with sample data.
"""

from pathlib import Path
import json

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import (
    Scraper, ScrapedItem,
    Raffinatore, RefinedItem,
    Correlatore, Correlation, OceanEvent,
    KnowledgeScorer, ScoredKnowledge
)

def create_sample_items():
    """Create sample scraped items for testing."""
    return [
        ScrapedItem(
            id="sample_001",
            source_type="paper",
            url="https://example.com/paper1",
            title="Atlantic Water Intrusion Detected in Fram Strait During Summer 2023",
            content="""
            We report observations of enhanced Atlantic Water transport through Fram Strait
            during June-August 2023. Temperature anomalies of +1.5°C were recorded at 200m depth.
            This warm water pulse is expected to accelerate sea ice melt in the Arctic basin.
            The transport reached 8.5 Sv, significantly above the long-term mean of 6.8 Sv.
            Our analysis suggests this event is linked to the weakened Arctic Oscillation index.
            """,
            authors=["A. Researcher", "B. Scientist"],
            published_date="2023-06-15",
            source_name="Nature Climate",
            keywords=["Arctic", "Fram Strait", "Atlantic Water"],
            word_count=85,
            has_date=True,
            has_authors=True,
            topic_tags=["fram strait", "temperature"],
        ),
        ScrapedItem(
            id="sample_002",
            source_type="news",
            url="https://example.com/news1",
            title="Arctic Sea Ice Reaches Record Low for June",
            content="""
            Arctic sea ice extent hit a new record low for June 2023, according to satellite data.
            The ice cover measured 10.8 million km², about 1.2 million km² below the 1981-2010 average.
            Scientists attribute the rapid melt to unusually warm Atlantic water flowing into the region
            and persistent southerly winds. The Barents Sea showed particularly dramatic ice loss.
            """,
            authors=["Climate Correspondent"],
            published_date="2023-06-20",
            source_name="Science News",
            keywords=["sea ice", "Arctic", "climate"],
            word_count=72,
            has_date=True,
            has_authors=True,
            topic_tags=["sea ice", "arctic"],
        ),
        ScrapedItem(
            id="sample_003",
            source_type="paper",
            url="https://example.com/paper2",
            title="Atmospheric Forcing of Arctic Sea Ice Variability: Role of the NAO",
            content="""
            This study examines the relationship between the North Atlantic Oscillation (NAO)
            and Arctic sea ice extent over the period 1979-2023. We find that negative NAO phases
            are associated with enhanced storm activity and increased precipitation over the Arctic.
            Wind patterns during these events push sea ice away from the Siberian coast.
            The correlation coefficient between winter NAO and summer ice extent is -0.65.
            """,
            authors=["C. Atmospheric", "D. Ocean"],
            published_date="2023-05-01",
            source_name="Journal of Climate",
            keywords=["NAO", "sea ice", "atmospheric forcing"],
            word_count=78,
            has_date=True,
            has_authors=True,
            topic_tags=["atmosphere", "sea ice"],
        ),
        ScrapedItem(
            id="sample_004",
            source_type="paper",
            url="https://example.com/paper3",
            title="Heat Transport Variability in the Barents Sea: A Precursor for Arctic Warming",
            content="""
            We analyze 30 years of ocean heat transport data in the Barents Sea Opening.
            Our results show that anomalous heat transport events (>2σ) precede Arctic
            temperature anomalies by 3-6 months. The average heat transport is 73 TW,
            with extreme events reaching 120 TW. These warm pulses originate from the
            Norwegian Sea and are modulated by the Atlantic Multidecadal Oscillation.
            """,
            authors=["E. Heat", "F. Transport"],
            published_date="2023-04-15",
            source_name="GRL",
            keywords=["Barents Sea", "heat transport", "Arctic warming"],
            word_count=72,
            has_date=True,
            has_authors=True,
            topic_tags=["barents sea", "temperature"],
        ),
    ]


def main():
    print("="*70)
    print("🔄 DATA INTELLIGENCE PIPELINE - DEMO WITH SAMPLE DATA")
    print("="*70)
    
    # Create sample items
    print("\n📥 Creating sample scraped items...")
    items = create_sample_items()
    print(f"   Created {len(items)} sample items")
    
    # Stage 2: Refine
    print("\n🧹 Stage 2: RAFFINATORE")
    raffinatore = Raffinatore(min_quality_score=2.0)
    refined = raffinatore.refine_all(items)
    print(f"   Refined {len(refined)} items")
    
    for item in refined[:3]:
        print(f"   📄 {item.title[:50]}...")
        print(f"      Quality: {item.quality_score:.1f} | Topics: {item.topics}")
        print(f"      Locations: {item.locations} | Measurements: {len(item.measurements)}")
    
    # Stage 3: Correlate
    print("\n🔗 Stage 3: CORRELATORE")
    correlatore = Correlatore()
    correlations = correlatore.correlate_all(refined)
    print(f"   Found {len(correlations)} correlations")
    
    # Show some correlations
    precursors = [c for c in correlations if c.hypothesis_type == "precursor"]
    print(f"\n   🔮 Precursor signals found: {len(precursors)}")
    for p in precursors[:3]:
        print(f"      {p.item_id} → {p.event_id}")
        print(f"      Days before: {p.days_from_event} | Decay: {p.decay_weight:.2f}")
        print(f"      Confidence: {p.causal_confidence:.2f}")
    
    # Stage 4: Score
    print("\n🧠 Stage 4: KNOWLEDGE SCORER")
    scorer = KnowledgeScorer()
    knowledge = scorer.score_all()
    
    # Final results
    print("\n" + "="*70)
    print("📊 FINAL KNOWLEDGE SCORES")
    print("="*70)
    print(f"{'Event':<40} | {'Score':>6} | {'Conf':>5} | {'Pre':>4}")
    print("-"*70)
    
    for k in sorted(knowledge, key=lambda x: -x.overall_score):
        print(f"{k.event_name[:40]:<40} | {k.overall_score:>6.1f} | {k.confidence:>5.0%} | {k.precursor_count:>4}")
    
    print("\n📈 Multi-Factor Indices (best event):")
    if knowledge:
        best = max(knowledge, key=lambda x: x.overall_score)
        print(f"   Event: {best.event_name}")
        print(f"   🌡️ Thermodynamics: {best.thermodynamics:.1f}/10")
        print(f"   💨 Anemometry:     {best.anemometry:.1f}/10")
        print(f"   🌧️ Precipitation:  {best.precipitation:.1f}/10")
        print(f"   ❄️ Cryosphere:     {best.cryosphere:.1f}/10")
        print(f"   🌊 Oceanography:   {best.oceanography:.1f}/10")
        
        if best.key_findings:
            print(f"\n   🔑 Key Findings:")
            for finding in best.key_findings:
                print(f"      • {finding}")
    
    print("\n✅ Pipeline demo complete!")


if __name__ == "__main__":
    main()
