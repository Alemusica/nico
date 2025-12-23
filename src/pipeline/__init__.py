"""
🔄 Data Intelligence Pipeline
==============================

Four-stage pipeline for processing web data into scored knowledge:

1. SCRAPER → Harvest news, papers, RSS feeds
2. RAFFINATORE → Clean, extract entities, validate
3. CORRELATORE → Link to events, temporal correlation
4. KNOWLEDGE SCORER → Multi-factor indices, confidence scores

Usage:
    from src.pipeline import Scraper, Raffinatore, Correlatore, KnowledgeScorer
    
    # Stage 1: Scrape
    scraper = Scraper()
    raw_items = scraper.scrape_topics()
    
    # Stage 2: Refine
    raffinatore = Raffinatore()
    refined = raffinatore.refine_all(raw_items)
    
    # Stage 3: Correlate
    correlatore = Correlatore()
    correlations = correlatore.correlate_all(refined)
    
    # Stage 4: Score
    scorer = KnowledgeScorer()
    knowledge = scorer.score_all()
    
    # Or use the full pipeline:
    from src.pipeline import run_full_pipeline
    results = run_full_pipeline(topics=["arctic sea ice", "fram strait"])
"""

from .scraper import Scraper, ScrapedItem
from .raffinatore import Raffinatore, RefinedItem
from .correlatore import Correlatore, Correlation, OceanEvent
from .knowledge_scorer import KnowledgeScorer, ScoredKnowledge


def run_full_pipeline(
    topics: list = None,
    rss_feeds: list = None,
    paper_queries: list = None,
) -> dict:
    """
    Run the complete data intelligence pipeline.
    
    Args:
        topics: List of topics to scrape news for
        rss_feeds: List of RSS feeds to parse
        paper_queries: List of queries for paper search
        
    Returns:
        Dictionary with results from each stage
    """
    results = {
        "scraped": [],
        "refined": [],
        "correlations": [],
        "knowledge": [],
        "stats": {}
    }
    
    print("\n" + "="*70)
    print("🔄 RUNNING FULL DATA INTELLIGENCE PIPELINE")
    print("="*70)
    
    # Stage 1: Scrape
    print("\n📥 STAGE 1: SCRAPING...")
    scraper = Scraper()
    raw_items = scraper.scrape_topics(topics=topics)
    results["scraped"] = raw_items
    results["stats"]["scraped_count"] = len(raw_items)
    
    # Stage 2: Refine
    print("\n🧹 STAGE 2: REFINING...")
    raffinatore = Raffinatore()
    refined_items = raffinatore.refine_all(raw_items)
    results["refined"] = refined_items
    results["stats"]["refined_count"] = len(refined_items)
    results["stats"]["refine_stats"] = raffinatore.get_stats()
    
    # Stage 3: Correlate
    print("\n🔗 STAGE 3: CORRELATING...")
    correlatore = Correlatore()
    correlations = correlatore.correlate_all(refined_items)
    results["correlations"] = correlations
    results["stats"]["correlation_count"] = len(correlations)
    results["stats"]["correlate_stats"] = correlatore.get_stats()
    
    # Stage 4: Score
    print("\n🧠 STAGE 4: SCORING...")
    scorer = KnowledgeScorer()
    knowledge = scorer.score_all()
    results["knowledge"] = knowledge
    results["stats"]["knowledge_count"] = len(knowledge)
    results["stats"]["knowledge_stats"] = scorer.get_stats()
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"   Scraped:      {len(raw_items)} items")
    print(f"   Refined:      {len(refined_items)} items")
    print(f"   Correlations: {len(correlations)}")
    print(f"   Knowledge:    {len(knowledge)} events scored")
    
    return results


# Import enhanced scorer
from .enhanced_scorer import (
    EnhancedKnowledgeScorer,
    DynamicIndices,
    HybridScore,
    create_enhanced_scorer,
)


__all__ = [
    # Core pipeline
    "Scraper",
    "ScrapedItem",
    "Raffinatore",
    "RefinedItem", 
    "Correlatore",
    "Correlation",
    "OceanEvent",
    "KnowledgeScorer",
    "ScoredKnowledge",
    "run_full_pipeline",
    # Enhanced scoring
    "EnhancedKnowledgeScorer",
    "DynamicIndices",
    "HybridScore",
    "create_enhanced_scorer",
]

