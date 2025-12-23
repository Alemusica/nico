"""
Agent Tools Module
==================

Tools for investigation:
- Geo resolution (Nominatim)
- Literature scraping (arXiv, Semantic Scholar)
- PDF parsing
"""

from .geo_resolver import GeoResolver, GeoLocation, KNOWN_LOCATIONS
from .literature_scraper import (
    LiteratureScraper,
    ArxivScraper,
    SemanticScholarScraper,
    Paper,
)
from .pdf_parser import PDFParser, ParsedPaper, PaperSection, Reference

__all__ = [
    # Geo
    "GeoResolver",
    "GeoLocation",
    "KNOWN_LOCATIONS",
    # Literature
    "LiteratureScraper",
    "ArxivScraper",
    "SemanticScholarScraper",
    "Paper",
    # PDF
    "PDFParser",
    "ParsedPaper",
    "PaperSection",
    "Reference",
]
