"""
🧹 Raffinatore Module (Data Refinery)
=====================================

Transforms raw scraped data into clean, structured, validated items:

1. ENTITY EXTRACTION - Locations, dates, events, measurements
2. DEDUPLICATION - Similarity-based duplicate detection
3. QUALITY SCORING - Relevance, completeness, freshness
4. GARBAGE FILTERING - Remove low-quality, irrelevant content

Output: Refined items ready for correlation analysis.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import defaultdict
import unicodedata

# NLP
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available, using regex fallback")

from .scraper import ScrapedItem


@dataclass
class ExtractedEntity:
    """An extracted entity from text."""
    text: str
    label: str  # GPE (location), DATE, ORG, EVENT, QUANTITY
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class RefinedItem:
    """A refined, validated data item."""
    id: str
    original_id: str  # Reference to ScrapedItem
    source_type: str
    
    # Core content
    title: str
    content: str
    summary: str
    
    # Extracted metadata
    authors: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    source_name: Optional[str] = None
    url: str = ""
    
    # Extracted entities
    locations: List[str] = field(default_factory=list)  # Geographic entities
    dates_mentioned: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    measurements: List[Dict[str, Any]] = field(default_factory=list)  # {"value": 5.2, "unit": "°C"}
    events: List[str] = field(default_factory=list)
    
    # Topic classification
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Quality scores (0-10)
    quality_score: float = 0.0
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    freshness_score: float = 0.0
    
    # Processing metadata
    refined_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RefinedItem":
        return cls(**data)


class Raffinatore:
    """
    Data Refinery - Cleans and enriches raw scraped data.
    
    Pipeline:
    1. Load raw items
    2. Extract entities (locations, dates, measurements)
    3. Classify topics
    4. Score quality
    5. Detect duplicates
    6. Filter garbage
    7. Save refined items
    """
    
    # Geographic keywords for Arctic/Ocean research
    GEO_KEYWORDS = {
        "arctic": ["arctic", "polar", "north pole"],
        "fram_strait": ["fram strait", "fram", "greenland sea"],
        "barents_sea": ["barents", "barents sea", "svalbard"],
        "norwegian_sea": ["norwegian sea", "norway", "nordic"],
        "north_atlantic": ["atlantic", "north atlantic", "gulf stream"],
        "bering_strait": ["bering", "bering strait", "alaska"],
        "beaufort_sea": ["beaufort", "beaufort sea", "canada arctic"],
        "greenland": ["greenland", "greenlandic"],
        "siberian": ["siberian", "laptev", "east siberian"],
    }
    
    # Topic keywords
    TOPIC_KEYWORDS = {
        "sea_ice": ["sea ice", "ice extent", "ice cover", "ice melt", "ice loss"],
        "temperature": ["temperature", "warming", "heat", "thermal", "sst", "°c"],
        "currents": ["current", "circulation", "transport", "flow", "gyre"],
        "sea_level": ["sea level", "ssh", "sla", "tide", "surge"],
        "atmosphere": ["wind", "pressure", "storm", "cyclone", "nao", "ao"],
        "precipitation": ["precipitation", "rain", "snow", "freshwater"],
        "salinity": ["salinity", "salt", "freshening", "halocline"],
    }
    
    # Measurement patterns
    MEASUREMENT_PATTERNS = [
        (r"(-?\d+\.?\d*)\s*°C", "temperature", "°C"),
        (r"(-?\d+\.?\d*)\s*°F", "temperature", "°F"),
        (r"(-?\d+\.?\d*)\s*km²", "area", "km²"),
        (r"(-?\d+\.?\d*)\s*million\s*km²", "area", "million_km²"),
        (r"(-?\d+\.?\d*)\s*m/s", "velocity", "m/s"),
        (r"(-?\d+\.?\d*)\s*cm", "length", "cm"),
        (r"(-?\d+\.?\d*)\s*mm", "length", "mm"),
        (r"(-?\d+\.?\d*)\s*Sv", "transport", "Sv"),  # Sverdrup
        (r"(-?\d+\.?\d*)\s*%", "percentage", "%"),
        (r"(-?\d+\.?\d*)\s*days?", "time", "days"),
    ]
    
    def __init__(
        self,
        input_dir: Path = None,
        output_dir: Path = None,
        min_quality_score: float = 3.0,
        similarity_threshold: float = 0.85,
    ):
        base = Path(__file__).parent.parent.parent.parent / "data" / "pipeline"
        self.input_dir = input_dir or base / "raw"
        self.output_dir = output_dir or base / "refined"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_quality_score = min_quality_score
        self.similarity_threshold = similarity_threshold
        
        # Load spaCy model
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("   ⚠️ Downloading spaCy model...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
        
        print(f"🧹 Raffinatore initialized")
        print(f"   📂 Input: {self.input_dir}")
        print(f"   📂 Output: {self.output_dir}")
        print(f"   🔬 spaCy: {'✅' if self.nlp else '❌ (regex mode)'}")
    
    # =========================================================================
    # Entity Extraction
    # =========================================================================
    
    def extract_entities_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text[:10000])  # Limit text length
        entities = []
        
        for ent in doc.ents:
            entities.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            ))
        
        return entities
    
    def extract_entities_regex(self, text: str) -> List[ExtractedEntity]:
        """Fallback entity extraction using regex patterns."""
        entities = []
        text_lower = text.lower()
        
        # Extract geographic locations
        for region, keywords in self.GEO_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    entities.append(ExtractedEntity(
                        text=region,
                        label="GPE",
                        start=text_lower.find(kw),
                        end=text_lower.find(kw) + len(kw),
                    ))
        
        # Extract dates (simple patterns)
        date_patterns = [
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{4}\b",  # Years
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        ]
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label="DATE",
                    start=match.start(),
                    end=match.end(),
                ))
        
        return entities
    
    def extract_measurements(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical measurements with units."""
        measurements = []
        
        for pattern, mtype, unit in self.MEASUREMENT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    measurements.append({
                        "value": value,
                        "unit": unit,
                        "type": mtype,
                        "context": text[max(0, match.start()-30):match.end()+30],
                    })
                except ValueError:
                    pass
        
        return measurements
    
    def classify_topics(self, text: str) -> List[str]:
        """Classify text into predefined topics."""
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score >= 1:
                topics.append(topic)
        
        return topics
    
    def extract_locations(self, text: str) -> List[str]:
        """Extract geographic locations."""
        text_lower = text.lower()
        locations = set()
        
        for region, keywords in self.GEO_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    locations.add(region)
        
        return list(locations)
    
    # =========================================================================
    # Quality Scoring
    # =========================================================================
    
    def score_quality(self, item: ScrapedItem) -> Dict[str, float]:
        """
        Score item quality on multiple dimensions (0-10 scale).
        """
        scores = {}
        
        # Completeness: Has title, content, date, authors
        completeness = 0
        if item.title and len(item.title) > 10:
            completeness += 2.5
        if item.content and len(item.content) > 100:
            completeness += 2.5
        if item.has_date:
            completeness += 2.5
        if item.has_authors:
            completeness += 2.5
        scores["completeness"] = completeness
        
        # Content quality: Word count, no boilerplate
        content_score = 0
        word_count = item.word_count
        if word_count > 500:
            content_score += 3
        elif word_count > 200:
            content_score += 2
        elif word_count > 50:
            content_score += 1
        
        # Check for boilerplate/garbage
        garbage_indicators = [
            "cookie", "subscribe", "newsletter", "advertisement",
            "click here", "sign up", "privacy policy", "terms of service",
        ]
        garbage_count = sum(1 for g in garbage_indicators if g in item.content.lower())
        content_score += max(0, 4 - garbage_count)
        
        # Scientific indicators
        if item.source_type == "paper":
            content_score += 3
        elif "abstract" in item.content.lower() or "doi" in item.content.lower():
            content_score += 2
        
        scores["content"] = min(10, content_score)
        
        # Relevance: Topic keywords found
        text = f"{item.title} {item.content}".lower()
        relevant_topics = sum(
            1 for topic_kws in self.TOPIC_KEYWORDS.values()
            for kw in topic_kws if kw in text
        )
        scores["relevance"] = min(10, relevant_topics * 1.5)
        
        # Freshness: Recency of publication
        if item.published_date:
            try:
                pub_date = datetime.fromisoformat(item.published_date.replace("Z", "+00:00"))
                days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
                if days_old < 30:
                    scores["freshness"] = 10
                elif days_old < 90:
                    scores["freshness"] = 8
                elif days_old < 365:
                    scores["freshness"] = 6
                elif days_old < 730:
                    scores["freshness"] = 4
                else:
                    scores["freshness"] = 2
            except:
                scores["freshness"] = 5
        else:
            scores["freshness"] = 3
        
        # Overall score
        scores["overall"] = (
            scores["completeness"] * 0.2 +
            scores["content"] * 0.3 +
            scores["relevance"] * 0.3 +
            scores["freshness"] * 0.2
        )
        
        return scores
    
    # =========================================================================
    # Deduplication
    # =========================================================================
    
    def text_fingerprint(self, text: str) -> str:
        """Create a fingerprint for deduplication."""
        # Normalize text
        text = unicodedata.normalize("NFKD", text.lower())
        text = re.sub(r"[^\w\s]", "", text)
        words = text.split()
        
        # Use sorted word hashes for similarity
        word_hashes = sorted(set(hashlib.md5(w.encode()).hexdigest()[:8] for w in words))
        return hashlib.sha256("".join(word_hashes[:50]).encode()).hexdigest()[:32]
    
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def detect_duplicates(
        self, 
        items: List[RefinedItem]
    ) -> List[RefinedItem]:
        """Mark duplicate items."""
        fingerprints = {}
        
        for item in items:
            fp = self.text_fingerprint(item.content)
            
            if fp in fingerprints:
                item.is_duplicate = True
                item.duplicate_of = fingerprints[fp]
            else:
                fingerprints[fp] = item.id
                
                # Also check similarity with existing items
                for other in items:
                    if other.id == item.id or other.is_duplicate:
                        continue
                    
                    sim = self.jaccard_similarity(item.title, other.title)
                    if sim > self.similarity_threshold:
                        item.is_duplicate = True
                        item.duplicate_of = other.id
                        break
        
        return items
    
    # =========================================================================
    # Main Refinement Process
    # =========================================================================
    
    def refine_item(self, raw: ScrapedItem) -> Optional[RefinedItem]:
        """Refine a single scraped item."""
        # Score quality
        scores = self.score_quality(raw)
        
        # Filter garbage
        if scores["overall"] < self.min_quality_score:
            return None
        
        # Extract entities
        if self.nlp:
            entities = self.extract_entities_spacy(f"{raw.title} {raw.content}")
        else:
            entities = self.extract_entities_regex(f"{raw.title} {raw.content}")
        
        # Extract measurements
        measurements = self.extract_measurements(raw.content)
        
        # Classify topics
        topics = self.classify_topics(f"{raw.title} {raw.content}")
        
        # Extract locations
        locations = self.extract_locations(f"{raw.title} {raw.content}")
        
        # Filter entities by type
        dates_mentioned = [e.text for e in entities if e.label == "DATE"]
        organizations = [e.text for e in entities if e.label == "ORG"]
        
        # Create refined item
        refined = RefinedItem(
            id=f"refined_{raw.id}",
            original_id=raw.id,
            source_type=raw.source_type,
            title=raw.title,
            content=raw.content,
            summary=raw.summary or raw.content[:500],
            authors=raw.authors,
            published_date=raw.published_date,
            source_name=raw.source_name,
            url=raw.url,
            locations=locations,
            dates_mentioned=dates_mentioned[:10],
            organizations=organizations[:10],
            measurements=measurements[:20],
            topics=topics,
            keywords=raw.keywords,
            quality_score=scores["overall"],
            relevance_score=scores["relevance"],
            completeness_score=scores["completeness"],
            freshness_score=scores["freshness"],
        )
        
        return refined
    
    def refine_all(self, raw_items: List[ScrapedItem] = None) -> List[RefinedItem]:
        """
        Refine all raw items.
        """
        # Load raw items if not provided
        if raw_items is None:
            raw_items = []
            for file in self.input_dir.glob("*.json"):
                if file.name == "seen_urls.json":
                    continue
                try:
                    with open(file) as f:
                        raw_items.append(ScrapedItem.from_dict(json.load(f)))
                except:
                    pass
        
        print("\n" + "="*60)
        print("🧹 RAFFINATORE - DATA REFINEMENT")
        print("="*60)
        print(f"📥 Input items: {len(raw_items)}")
        
        # Refine each item
        refined_items = []
        garbage_count = 0
        
        for raw in raw_items:
            refined = self.refine_item(raw)
            if refined:
                refined_items.append(refined)
            else:
                garbage_count += 1
        
        print(f"   ✅ Refined: {len(refined_items)}")
        print(f"   🗑️ Filtered (garbage): {garbage_count}")
        
        # Detect duplicates
        refined_items = self.detect_duplicates(refined_items)
        duplicates = sum(1 for i in refined_items if i.is_duplicate)
        print(f"   🔄 Duplicates detected: {duplicates}")
        
        # Save refined items
        for item in refined_items:
            output_file = self.output_dir / f"{item.id}.json"
            with open(output_file, "w") as f:
                json.dump(item.to_dict(), f, indent=2)
        
        # Summary
        unique_items = [i for i in refined_items if not i.is_duplicate]
        print(f"\n📊 REFINEMENT COMPLETE")
        print(f"   📤 Unique refined items: {len(unique_items)}")
        
        # Topic distribution
        topic_counts = defaultdict(int)
        for item in unique_items:
            for topic in item.topics:
                topic_counts[topic] += 1
        
        print(f"\n📈 Topics found:")
        for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
            print(f"   {topic}: {count}")
        
        return unique_items
    
    def get_refined_items(self) -> List[RefinedItem]:
        """Load all refined items from disk."""
        items = []
        for file in self.output_dir.glob("*.json"):
            try:
                with open(file) as f:
                    items.append(RefinedItem.from_dict(json.load(f)))
            except:
                pass
        return [i for i in items if not i.is_duplicate]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get refinement statistics."""
        items = self.get_refined_items()
        
        topic_counts = defaultdict(int)
        location_counts = defaultdict(int)
        
        for item in items:
            for topic in item.topics:
                topic_counts[topic] += 1
            for loc in item.locations:
                location_counts[loc] += 1
        
        avg_quality = sum(i.quality_score for i in items) / len(items) if items else 0
        
        return {
            "total_refined": len(items),
            "avg_quality_score": round(avg_quality, 2),
            "topics": dict(topic_counts),
            "locations": dict(location_counts),
            "with_measurements": sum(1 for i in items if i.measurements),
        }
