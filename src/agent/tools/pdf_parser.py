"""
📄 PDF Parser for Scientific Papers
===================================

Extract structured information from PDF papers:
- Abstract, sections, figures, tables
- References (for citation graph)
- Entities (locations, dates, methods)
- Key findings

Uses multiple backends:
- PyMuPDF (fitz) - fast, good quality
- pdfplumber - good for tables
- GROBID - best for scientific papers (requires server)
"""

import os
import re
import ssl
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json

try:
    import certifi
    HAS_CERTIFI = True
except ImportError:
    HAS_CERTIFI = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# SSL context for macOS certificate verification
if HAS_CERTIFI:
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
else:
    SSL_CONTEXT = ssl.create_default_context()


@dataclass
class PaperSection:
    """Parsed section from a paper."""
    title: str
    content: str
    level: int = 1  # Heading level
    page_start: int = 0
    page_end: int = 0


@dataclass
class Reference:
    """Parsed reference."""
    raw_text: str
    authors: List[str] = field(default_factory=list)
    title: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""


@dataclass
class Figure:
    """Figure/table metadata."""
    caption: str
    page: int
    figure_type: str = "figure"  # figure, table, equation
    number: str = ""


@dataclass
class ParsedPaper:
    """Fully parsed paper."""
    title: str
    authors: List[str]
    abstract: str
    full_text: str
    
    sections: List[PaperSection] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    
    # Extracted entities
    locations: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    
    # Metadata
    doi: str = ""
    keywords: List[str] = field(default_factory=list)
    page_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'sections': [{'title': s.title, 'content': s.content} for s in self.sections],
            'references_count': len(self.references),
            'figures_count': len(self.figures),
            'locations': self.locations,
            'dates': self.dates,
            'methods': self.methods,
        }


class PDFParser:
    """
    Parse scientific PDFs.
    
    Usage:
        parser = PDFParser()
        
        # Parse a paper
        paper = await parser.parse("/path/to/paper.pdf")
        
        print(paper.title)
        print(paper.abstract)
        for sec in paper.sections:
            print(f"## {sec.title}")
    """
    
    # Common section headers in scientific papers
    SECTION_PATTERNS = [
        r'^(?:1\.?\s*)?(?:INTRODUCTION|Introduction)',
        r'^(?:2\.?\s*)?(?:METHODS?|Methods?|METHODOLOGY|Methodology)',
        r'^(?:2\.?\s*)?(?:DATA|Data|DATA\s+AND\s+METHODS)',
        r'^(?:3\.?\s*)?(?:RESULTS?|Results?)',
        r'^(?:4\.?\s*)?(?:DISCUSSION|Discussion)',
        r'^(?:5\.?\s*)?(?:CONCLUSIONS?|Conclusions?)',
        r'^(?:REFERENCES?|References?|BIBLIOGRAPHY)',
        r'^(?:ACKNOWLEDGEMENTS?|Acknowledgements?)',
        r'^(?:ABSTRACT|Abstract)',
        r'^(?:APPENDIX|Appendix)',
        r'^STUDY\s+AREA',
        r'^BACKGROUND',
    ]
    
    # Patterns for entity extraction
    LOCATION_PATTERNS = [
        r'(?:Lake|Lago|River|Rio|Valley|Valle|Basin|Region|Area)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Lake|River|Valley|Basin)',
        r'(?:northern|southern|eastern|western|central)\s+(?:Italy|Europe|Alps|Mediterranean)',
    ]
    
    METHOD_PATTERNS = [
        r'(?:ERA5|ECMWF|CMEMS|Sentinel|Landsat|MODIS|GPM)',
        r'(?:LSTM|CNN|Random\s+Forest|XGBoost|Neural\s+Network)',
        r'(?:HEC-RAS|SWAT|MIKE|WRF|HYSPLIT)',
        r'(?:regression|correlation|principal\s+component|EOF)',
    ]
    
    def __init__(self, use_grobid: bool = False, grobid_url: str = None):
        self.use_grobid = use_grobid
        self.grobid_url = grobid_url or "http://localhost:8070"
    
    async def parse(self, pdf_path: Path) -> Optional[ParsedPaper]:
        """
        Parse a PDF paper.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ParsedPaper with extracted content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            return None
        
        # Try GROBID first (best for scientific papers)
        if self.use_grobid:
            result = await self._parse_grobid(pdf_path)
            if result:
                return result
        
        # Fall back to PyMuPDF
        if HAS_PYMUPDF:
            return self._parse_pymupdf(pdf_path)
        
        # Fall back to pdfplumber
        if HAS_PDFPLUMBER:
            return self._parse_pdfplumber(pdf_path)
        
        print("❌ No PDF parser available. Install: pip install pymupdf pdfplumber")
        return None
    
    def _parse_pymupdf(self, pdf_path: Path) -> ParsedPaper:
        """Parse using PyMuPDF."""
        doc = fitz.open(pdf_path)
        
        full_text = ""
        sections = []
        figures = []
        
        # Extract text from all pages
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            full_text += page_text + "\n"
            
            # Look for figures/tables
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") == 1:  # Image
                    # Try to find caption nearby
                    figures.append(Figure(
                        caption=f"Figure on page {page_num + 1}",
                        page=page_num + 1,
                        figure_type="figure"
                    ))
        
        # Extract title (usually first large text)
        title = self._extract_title(full_text)
        
        # Extract abstract
        abstract = self._extract_abstract(full_text)
        
        # Extract authors (tricky, usually between title and abstract)
        authors = self._extract_authors(full_text)
        
        # Parse sections
        sections = self._extract_sections(full_text)
        
        # Extract references
        references = self._extract_references(full_text)
        
        # Extract entities
        locations = self._extract_locations(full_text)
        dates = self._extract_dates(full_text)
        methods = self._extract_methods(full_text)
        
        doc.close()
        
        return ParsedPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            full_text=full_text,
            sections=sections,
            references=references,
            figures=figures,
            locations=locations,
            dates=dates,
            methods=methods,
            page_count=len(doc) if doc else 0,
        )
    
    def _parse_pdfplumber(self, pdf_path: Path) -> ParsedPaper:
        """Parse using pdfplumber."""
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            title = self._extract_title(full_text)
            abstract = self._extract_abstract(full_text)
            authors = self._extract_authors(full_text)
            sections = self._extract_sections(full_text)
            references = self._extract_references(full_text)
            
            return ParsedPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                full_text=full_text,
                sections=sections,
                references=references,
                locations=self._extract_locations(full_text),
                dates=self._extract_dates(full_text),
                methods=self._extract_methods(full_text),
                page_count=len(pdf.pages),
            )
    
    async def _parse_grobid(self, pdf_path: Path) -> Optional[ParsedPaper]:
        """Parse using GROBID server (best quality)."""
        if not HAS_AIOHTTP:
            return None
        
        try:
            connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT)
            async with aiohttp.ClientSession(connector=connector) as session:
                with open(pdf_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('input', f, filename=pdf_path.name)
                    
                    url = f"{self.grobid_url}/api/processFulltextDocument"
                    async with session.post(url, data=data, timeout=120) as response:
                        if response.status == 200:
                            tei_xml = await response.text()
                            return self._parse_tei(tei_xml)
                        else:
                            print(f"⚠️ GROBID error: {response.status}")
                            return None
                            
        except Exception as e:
            print(f"⚠️ GROBID not available: {e}")
            return None
    
    def _parse_tei(self, tei_xml: str) -> ParsedPaper:
        """Parse GROBID TEI XML output."""
        import xml.etree.ElementTree as ET
        
        ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
        root = ET.fromstring(tei_xml)
        
        # Extract title
        title_elem = root.find('.//tei:titleStmt/tei:title', ns)
        title = title_elem.text if title_elem is not None else ""
        
        # Extract authors
        authors = []
        for author in root.findall('.//tei:sourceDesc//tei:persName', ns):
            forename = author.find('tei:forename', ns)
            surname = author.find('tei:surname', ns)
            name_parts = []
            if forename is not None and forename.text:
                name_parts.append(forename.text)
            if surname is not None and surname.text:
                name_parts.append(surname.text)
            if name_parts:
                authors.append(' '.join(name_parts))
        
        # Extract abstract
        abstract_elem = root.find('.//tei:profileDesc/tei:abstract', ns)
        abstract = ""
        if abstract_elem is not None:
            abstract = ' '.join(abstract_elem.itertext())
        
        # Extract full text from body
        body_elem = root.find('.//tei:body', ns)
        full_text = ' '.join(body_elem.itertext()) if body_elem is not None else ""
        
        # Extract sections
        sections = []
        for div in root.findall('.//tei:body/tei:div', ns):
            head = div.find('tei:head', ns)
            if head is not None:
                sec_title = head.text or ""
                sec_content = ' '.join(div.itertext())
                sections.append(PaperSection(title=sec_title, content=sec_content))
        
        # Extract references
        references = []
        for bibl in root.findall('.//tei:listBibl/tei:biblStruct', ns):
            ref_title = bibl.find('.//tei:title', ns)
            ref_authors = [' '.join(a.itertext()) for a in bibl.findall('.//tei:persName', ns)]
            ref_year = bibl.find('.//tei:date', ns)
            
            references.append(Reference(
                raw_text=' '.join(bibl.itertext()),
                title=ref_title.text if ref_title is not None else "",
                authors=ref_authors,
                year=ref_year.get('when', '') if ref_year is not None else "",
            ))
        
        return ParsedPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            full_text=full_text,
            sections=sections,
            references=references,
            locations=self._extract_locations(full_text),
            dates=self._extract_dates(full_text),
            methods=self._extract_methods(full_text),
        )
    
    def _extract_title(self, text: str) -> str:
        """Extract paper title (usually first lines, larger font)."""
        lines = text.split('\n')
        
        # Title is usually in first few non-empty lines
        title_lines = []
        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue
            if len(line) > 10:  # Skip short lines
                # Stop at abstract or author section
                if any(x in line.lower() for x in ['abstract', '@', 'university', 'department']):
                    break
                title_lines.append(line)
                if len(' '.join(title_lines)) > 150:  # Title usually < 200 chars
                    break
        
        return ' '.join(title_lines[:2])
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section."""
        # Look for "Abstract" keyword
        patterns = [
            r'(?:^|\n)(?:ABSTRACT|Abstract)\s*\n(.*?)(?:\n\s*(?:1\.?\s*)?(?:INTRODUCTION|Introduction|Keywords|KEY\s*WORDS))',
            r'(?:^|\n)(?:ABSTRACT|Abstract)\s*[:.]?\s*(.*?)(?:\n\s*\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                # Clean up
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 100:
                    return abstract[:2000]  # Cap at 2000 chars
        
        return ""
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names."""
        # This is tricky - usually between title and abstract
        # Look for name patterns
        lines = text.split('\n')
        authors = []
        
        # Look in first 20 lines
        for line in lines[1:20]:
            line = line.strip()
            # Skip empty or section headers
            if not line or any(x in line.lower() for x in ['abstract', 'introduction', 'university', 'department']):
                continue
            
            # Look for name-like patterns (First Last)
            name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)'
            names = re.findall(name_pattern, line)
            
            # If line has multiple names separated by commas
            if ',' in line and len(names) > 1:
                authors.extend(names[:10])  # Cap at 10 authors
                break
        
        return authors
    
    def _extract_sections(self, text: str) -> List[PaperSection]:
        """Extract paper sections."""
        sections = []
        
        # Combined pattern for all section headers
        all_patterns = '|'.join(self.SECTION_PATTERNS)
        pattern = re.compile(f'({all_patterns})', re.MULTILINE | re.IGNORECASE)
        
        matches = list(pattern.finditer(text))
        
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_title = match.group(1).strip()
            section_content = text[start:end].strip()
            
            # Clean up content
            section_content = re.sub(r'\s+', ' ', section_content)[:5000]  # Cap at 5000 chars
            
            sections.append(PaperSection(
                title=section_title,
                content=section_content,
            ))
        
        return sections
    
    def _extract_references(self, text: str) -> List[Reference]:
        """Extract references section."""
        references = []
        
        # Find references section
        ref_match = re.search(
            r'(?:^|\n)(?:REFERENCES?|References?|BIBLIOGRAPHY)\s*\n(.*?)(?:\n(?:APPENDIX|$))',
            text,
            re.DOTALL | re.IGNORECASE
        )
        
        if not ref_match:
            return []
        
        ref_text = ref_match.group(1)
        
        # Split by reference numbers or newlines
        # Common patterns: [1], (1), 1., 1)
        ref_pattern = r'(?:^|\n)\s*(?:\[?\d+\]?\.?|\(\d+\))\s*'
        ref_items = re.split(ref_pattern, ref_text)
        
        for item in ref_items:
            item = item.strip()
            if len(item) < 20:  # Too short
                continue
            
            # Try to extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', item)
            year = year_match.group(0) if year_match else ""
            
            # Try to extract DOI
            doi_match = re.search(r'10\.\d{4,}/[^\s]+', item)
            doi = doi_match.group(0) if doi_match else ""
            
            references.append(Reference(
                raw_text=item[:500],  # Cap at 500 chars
                year=year,
                doi=doi,
            ))
        
        return references[:100]  # Cap at 100 references
    
    def _extract_locations(self, text: str) -> List[str]:
        """Extract geographic locations."""
        locations = set()
        
        for pattern in self.LOCATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            locations.update(matches)
        
        # Also look for specific keywords
        specific_locations = [
            'Lago Maggiore', 'Lake Maggiore', 'Po Valley', 'Val Padana',
            'Alps', 'Mediterranean', 'Adriatic', 'Ticino',
        ]
        for loc in specific_locations:
            if loc.lower() in text.lower():
                locations.add(loc)
        
        return list(locations)[:20]
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates and time references."""
        dates = set()
        
        # Various date patterns
        patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}',
            r'\b(?:flood|event|storm)\s+(?:of\s+)?(?:19|20)\d{2}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.update(matches)
        
        return list(dates)[:20]
    
    def _extract_methods(self, text: str) -> List[str]:
        """Extract methods, models, and datasets mentioned."""
        methods = set()
        
        for pattern in self.METHOD_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            methods.update(matches)
        
        return list(methods)[:30]


# CLI test
if __name__ == "__main__":
    import sys
    
    async def test():
        if len(sys.argv) < 2:
            print("Usage: python pdf_parser.py <pdf_file>")
            return
        
        parser = PDFParser()
        paper = await parser.parse(sys.argv[1])
        
        if paper:
            print(f"📄 Title: {paper.title}")
            print(f"👥 Authors: {', '.join(paper.authors[:5])}")
            print(f"\n📝 Abstract:\n{paper.abstract[:500]}...")
            
            print(f"\n📑 Sections ({len(paper.sections)}):")
            for sec in paper.sections:
                print(f"  - {sec.title}")
            
            print(f"\n📚 References: {len(paper.references)}")
            print(f"📍 Locations: {paper.locations}")
            print(f"📅 Dates: {paper.dates}")
            print(f"🔧 Methods: {paper.methods}")
    
    asyncio.run(test())
