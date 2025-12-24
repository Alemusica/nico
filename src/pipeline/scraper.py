"""
🕷️ Web Scraper Module
======================

Collects data from multiple sources:
- News articles (newspaper3k)
- Scientific papers (Semantic Scholar API - free)
- RSS feeds (feedparser)
- Custom sources

All free, no API keys required for basic usage.
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from urllib.parse import urlparse
import time

import feedparser
import requests
from bs4 import BeautifulSoup

# newspaper3k for article extraction
try:
    from newspaper import Article, Config as NewspaperConfig
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("Warning: newspaper3k not available")


@dataclass
class ScrapedItem:
    """A single scraped item (article, paper, etc.)."""
    id: str  # SHA256 hash of URL
    source_type: str  # 'news', 'paper', 'rss', 'custom'
    url: str
    title: str
    content: str
    summary: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Metadata
    source_name: Optional[str] = None  # "Reuters", "Nature", etc.
    language: str = "en"
    keywords: List[str] = field(default_factory=list)
    
    # Location/Topic tags (extracted later by Raffinatore)
    geo_tags: List[str] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)
    
    # Quality indicators
    word_count: int = 0
    has_date: bool = False
    has_authors: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScrapedItem":
        return cls(**data)


class Scraper:
    """
    Multi-source web scraper for news and scientific papers.
    
    Sources:
    - News: newspaper3k for article extraction
    - Papers: Semantic Scholar API (free, 100 req/5min)
    - RSS: feedparser for any RSS/Atom feed
    """
    
    # Semantic Scholar API (free, no key needed)
    SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
    
    # Pre-configured RSS feeds for oceanography/climate
    DEFAULT_RSS_FEEDS = {
        "nature_climate": "https://www.nature.com/nclimate.rss",
        "science_daily_earth": "https://www.sciencedaily.com/rss/earth_climate.xml",
        "phys_org_earth": "https://phys.org/rss-feed/earth-news/",
        "eos_agu": "https://eos.org/feed",
        "copernicus_ocean": "https://marine.copernicus.eu/feed",
    }
    
    # Default search topics
    DEFAULT_TOPICS = [
        "arctic sea ice",
        "fram strait",
        "atlantic water intrusion",
        "barents sea temperature",
        "ocean heat transport",
        "sea level anomaly",
        "marine heatwave",
    ]
    
    def __init__(
        self,
        output_dir: Path = None,
        rate_limit_delay: float = 1.0,
    ):
        self.output_dir = output_dir or Path(__file__).parent.parent.parent.parent / "data" / "pipeline" / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.seen_urls: Set[str] = set()
        
        # Load previously scraped URLs
        self._load_seen_urls()
        
        # Configure newspaper
        if NEWSPAPER_AVAILABLE:
            self.newspaper_config = NewspaperConfig()
            self.newspaper_config.browser_user_agent = (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self.newspaper_config.request_timeout = 15
            self.newspaper_config.fetch_images = False
        
        print(f"🕷️ Scraper initialized")
        print(f"   📁 Output: {self.output_dir}")
        print(f"   📰 Newspaper3k: {'✅' if NEWSPAPER_AVAILABLE else '❌'}")
    
    def _load_seen_urls(self):
        """Load previously scraped URLs to avoid duplicates."""
        seen_file = self.output_dir / "seen_urls.json"
        if seen_file.exists():
            with open(seen_file) as f:
                self.seen_urls = set(json.load(f))
    
    def _save_seen_urls(self):
        """Save scraped URLs."""
        seen_file = self.output_dir / "seen_urls.json"
        with open(seen_file, "w") as f:
            json.dump(list(self.seen_urls), f)
    
    def _generate_id(self, url: str) -> str:
        """Generate unique ID from URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def _save_item(self, item: ScrapedItem):
        """Save scraped item to disk."""
        item_file = self.output_dir / f"{item.id}.json"
        with open(item_file, "w") as f:
            json.dump(item.to_dict(), f, indent=2)
        self.seen_urls.add(item.url)
    
    # =========================================================================
    # News Scraping (newspaper3k)
    # =========================================================================
    
    def scrape_article(self, url: str) -> Optional[ScrapedItem]:
        """Scrape a single news article."""
        if url in self.seen_urls:
            return None
        
        if not NEWSPAPER_AVAILABLE:
            print(f"   ⚠️ newspaper3k not available, skipping: {url}")
            return None
        
        try:
            article = Article(url, config=self.newspaper_config)
            article.download()
            article.parse()
            
            # Try NLP for keywords
            try:
                article.nlp()
                keywords = article.keywords
                summary = article.summary
            except:
                keywords = []
                summary = None
            
            item = ScrapedItem(
                id=self._generate_id(url),
                source_type="news",
                url=url,
                title=article.title or "Untitled",
                content=article.text,
                summary=summary,
                authors=article.authors,
                published_date=article.publish_date.isoformat() if article.publish_date else None,
                source_name=urlparse(url).netloc,
                keywords=keywords,
                word_count=len(article.text.split()),
                has_date=article.publish_date is not None,
                has_authors=len(article.authors) > 0,
            )
            
            self._save_item(item)
            print(f"   ✅ Scraped: {item.title[:50]}...")
            return item
            
        except Exception as e:
            print(f"   ❌ Failed to scrape {url}: {e}")
            return None
    
    def scrape_news_search(
        self, 
        query: str, 
        max_results: int = 10
    ) -> List[ScrapedItem]:
        """
        Search for news articles using Google News RSS.
        Free, no API key needed.
        """
        print(f"\n📰 Searching news for: '{query}'")
        
        # Google News RSS (free)
        encoded_query = requests.utils.quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        items = []
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:max_results]:
            time.sleep(self.rate_limit_delay)
            
            # Google News links redirect, get actual URL
            actual_url = entry.get("link", "")
            
            item = self.scrape_article(actual_url)
            if item:
                item.topic_tags.append(query)
                items.append(item)
        
        print(f"   📊 Scraped {len(items)} articles")
        return items
    
    # =========================================================================
    # Scientific Papers (Semantic Scholar - FREE)
    # =========================================================================
    
    def search_papers(
        self,
        query: str,
        max_results: int = 20,
        year_start: int = None,
    ) -> List[ScrapedItem]:
        """
        Search for scientific papers via Semantic Scholar API.
        FREE, no API key required (100 requests per 5 minutes).
        """
        print(f"\n📚 Searching papers for: '{query}'")
        
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,abstract,authors,year,url,publicationDate,citationCount,fieldsOfStudy",
        }
        
        if year_start:
            params["year"] = f"{year_start}-"
        
        try:
            response = requests.get(
                f"{self.SEMANTIC_SCHOLAR_API}/paper/search",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            
        except Exception as e:
            print(f"   ❌ Semantic Scholar API error: {e}")
            return []
        
        items = []
        for paper in data.get("data", []):
            paper_id = paper.get("paperId", "")
            url = f"https://www.semanticscholar.org/paper/{paper_id}"
            
            if url in self.seen_urls:
                continue
            
            # Extract authors
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            
            # Handle None values
            abstract = paper.get("abstract") or ""
            
            item = ScrapedItem(
                id=self._generate_id(url),
                source_type="paper",
                url=url,
                title=paper.get("title", "Untitled") or "Untitled",
                content=abstract,
                summary=abstract[:500] if abstract else None,
                authors=authors,
                published_date=paper.get("publicationDate"),
                source_name="Semantic Scholar",
                keywords=paper.get("fieldsOfStudy") or [],
                word_count=len(abstract.split()),
                has_date=paper.get("publicationDate") is not None,
                has_authors=len(authors) > 0,
                topic_tags=[query],
            )
            
            self._save_item(item)
            items.append(item)
            print(f"   ✅ Paper: {item.title[:50]}...")
            
            time.sleep(self.rate_limit_delay)
        
        print(f"   📊 Found {len(items)} papers")
        return items
    
    # =========================================================================
    # RSS Feeds
    # =========================================================================
    
    def scrape_rss_feed(
        self,
        feed_url: str,
        feed_name: str = None,
        max_items: int = 20,
    ) -> List[ScrapedItem]:
        """Scrape items from an RSS feed."""
        print(f"\n📡 Scraping RSS: {feed_name or feed_url}")
        
        feed = feedparser.parse(feed_url)
        items = []
        
        for entry in feed.entries[:max_items]:
            url = entry.get("link", "")
            if url in self.seen_urls:
                continue
            
            # Get content from feed or scrape article
            content = ""
            if "content" in entry:
                content = entry.content[0].get("value", "")
            elif "summary" in entry:
                content = entry.summary
            
            # Clean HTML
            if content:
                soup = BeautifulSoup(content, "html.parser")
                content = soup.get_text(separator=" ", strip=True)
            
            # Parse date
            published = None
            if "published_parsed" in entry and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6]).isoformat()
                except:
                    pass
            
            item = ScrapedItem(
                id=self._generate_id(url),
                source_type="rss",
                url=url,
                title=entry.get("title", "Untitled"),
                content=content,
                summary=entry.get("summary", "")[:500],
                published_date=published,
                source_name=feed_name or feed.feed.get("title", "RSS"),
                word_count=len(content.split()),
                has_date=published is not None,
            )
            
            self._save_item(item)
            items.append(item)
        
        print(f"   📊 Scraped {len(items)} items from RSS")
        return items
    
    def scrape_default_feeds(self) -> List[ScrapedItem]:
        """Scrape all default RSS feeds."""
        all_items = []
        for name, url in self.DEFAULT_RSS_FEEDS.items():
            items = self.scrape_rss_feed(url, feed_name=name)
            all_items.extend(items)
            time.sleep(self.rate_limit_delay)
        return all_items
    
    # =========================================================================
    # Bulk Operations
    # =========================================================================
    
    def scrape_topics(
        self,
        topics: List[str] = None,
        include_news: bool = True,
        include_papers: bool = True,
        max_per_topic: int = 10,
    ) -> Dict[str, List[ScrapedItem]]:
        """
        Scrape multiple topics from news and papers.
        """
        topics = topics or self.DEFAULT_TOPICS
        results = {"news": [], "papers": [], "rss": []}
        
        print("\n" + "="*60)
        print("🕷️ BULK SCRAPING SESSION")
        print("="*60)
        print(f"Topics: {topics}")
        print(f"News: {'✅' if include_news else '❌'}")
        print(f"Papers: {'✅' if include_papers else '❌'}")
        
        for topic in topics:
            if include_news:
                items = self.scrape_news_search(topic, max_results=max_per_topic)
                results["news"].extend(items)
            
            if include_papers:
                items = self.search_papers(topic, max_results=max_per_topic, year_start=2020)
                results["papers"].extend(items)
            
            time.sleep(self.rate_limit_delay * 2)
        
        # Also scrape RSS feeds
        results["rss"] = self.scrape_default_feeds()
        
        # Save seen URLs
        self._save_seen_urls()
        
        # Summary
        total = sum(len(v) for v in results.values())
        print("\n" + "="*60)
        print(f"📊 SCRAPING COMPLETE: {total} items")
        print(f"   📰 News: {len(results['news'])}")
        print(f"   📚 Papers: {len(results['papers'])}")
        print(f"   📡 RSS: {len(results['rss'])}")
        print("="*60)
        
        return results
    
    def get_all_raw_items(self) -> List[ScrapedItem]:
        """Load all scraped items from disk."""
        items = []
        for file in self.output_dir.glob("*.json"):
            if file.name == "seen_urls.json":
                continue
            try:
                with open(file) as f:
                    data = json.load(f)
                    items.append(ScrapedItem.from_dict(data))
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
        return items
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        items = self.get_all_raw_items()
        
        by_type = {}
        by_source = {}
        
        for item in items:
            by_type[item.source_type] = by_type.get(item.source_type, 0) + 1
            by_source[item.source_name] = by_source.get(item.source_name, 0) + 1
        
        return {
            "total_items": len(items),
            "by_type": by_type,
            "by_source": dict(sorted(by_source.items(), key=lambda x: -x[1])[:10]),
            "unique_urls": len(self.seen_urls),
        }
