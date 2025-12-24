"""
Investigation System - Data Validation & Sanitization
Validates papers, cache entries, and data quality before storage.

Based on Great Expectations patterns:
- Schema validation
- Duplicate detection  
- Data quality checks
- Sanitization rules
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime


class ValidationLevel(str, Enum):
    """Validation severity levels"""
    CRITICAL = "critical"  # Must pass
    WARNING = "warning"   # Should pass
    INFO = "info"         # Nice to have


@dataclass
class ValidationResult:
    """Result of a validation check"""
    validator: str
    level: ValidationLevel
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validator": self.validator,
            "level": self.level.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details
        }


class PaperValidator:
    """Validates scientific papers before storage"""
    
    @staticmethod
    def validate_paper(paper: Dict[str, Any]) -> List[ValidationResult]:
        """Run all validations on a paper"""
        results = []
        
        # Critical: Required fields
        results.append(PaperValidator._check_required_fields(paper))
        
        # Critical: Valid DOI format (if present)
        if "doi" in paper and paper["doi"]:
            results.append(PaperValidator._check_doi_format(paper["doi"]))
        
        # Warning: Title length
        results.append(PaperValidator._check_title_length(paper.get("title", "")))
        
        # Warning: Abstract present
        results.append(PaperValidator._check_abstract_present(paper))
        
        # Info: Authors present
        results.append(PaperValidator._check_authors_present(paper))
        
        # Info: Publication year valid
        if "year" in paper and paper["year"]:
            results.append(PaperValidator._check_year_valid(paper["year"]))
        
        return results
    
    @staticmethod
    def _check_required_fields(paper: Dict[str, Any]) -> ValidationResult:
        """Check that required fields are present"""
        required = ["title"]
        missing = [f for f in required if not paper.get(f)]
        
        if missing:
            return ValidationResult(
                validator="required_fields",
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Missing required fields: {', '.join(missing)}",
                details={"missing_fields": missing}
            )
        
        return ValidationResult(
            validator="required_fields",
            level=ValidationLevel.CRITICAL,
            passed=True,
            message="All required fields present"
        )
    
    @staticmethod
    def _check_doi_format(doi: str) -> ValidationResult:
        """Validate DOI format"""
        # DOI pattern: 10.xxxx/xxxxx
        doi_pattern = r'^10\.\d{4,}(\.\d+)*\/[-._;()\/:A-Za-z0-9]+$'
        
        if re.match(doi_pattern, doi):
            return ValidationResult(
                validator="doi_format",
                level=ValidationLevel.CRITICAL,
                passed=True,
                message="Valid DOI format"
            )
        
        return ValidationResult(
            validator="doi_format",
            level=ValidationLevel.CRITICAL,
            passed=False,
            message=f"Invalid DOI format: {doi}",
            details={"doi": doi}
        )
    
    @staticmethod
    def _check_title_length(title: str) -> ValidationResult:
        """Check title has reasonable length"""
        if len(title) < 10:
            return ValidationResult(
                validator="title_length",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Title too short: {len(title)} characters",
                details={"length": len(title)}
            )
        
        if len(title) > 500:
            return ValidationResult(
                validator="title_length",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Title too long: {len(title)} characters",
                details={"length": len(title)}
            )
        
        return ValidationResult(
            validator="title_length",
            level=ValidationLevel.WARNING,
            passed=True,
            message="Title length appropriate"
        )
    
    @staticmethod
    def _check_abstract_present(paper: Dict[str, Any]) -> ValidationResult:
        """Check if abstract is present"""
        abstract = paper.get("abstract", "")
        
        if not abstract or len(abstract) < 50:
            return ValidationResult(
                validator="abstract_present",
                level=ValidationLevel.WARNING,
                passed=False,
                message="Abstract missing or too short",
                details={"length": len(abstract) if abstract else 0}
            )
        
        return ValidationResult(
            validator="abstract_present",
            level=ValidationLevel.WARNING,
            passed=True,
            message="Abstract present"
        )
    
    @staticmethod
    def _check_authors_present(paper: Dict[str, Any]) -> ValidationResult:
        """Check if authors are present"""
        authors = paper.get("authors", [])
        
        if not authors:
            return ValidationResult(
                validator="authors_present",
                level=ValidationLevel.INFO,
                passed=False,
                message="No authors listed"
            )
        
        return ValidationResult(
            validator="authors_present",
            level=ValidationLevel.INFO,
            passed=True,
            message=f"{len(authors)} authors listed"
        )
    
    @staticmethod
    def _check_year_valid(year: Any) -> ValidationResult:
        """Check if publication year is valid"""
        try:
            year_int = int(year)
            current_year = datetime.now().year
            
            if year_int < 1900 or year_int > current_year + 1:
                return ValidationResult(
                    validator="year_valid",
                    level=ValidationLevel.INFO,
                    passed=False,
                    message=f"Unusual publication year: {year_int}",
                    details={"year": year_int}
                )
            
            return ValidationResult(
                validator="year_valid",
                level=ValidationLevel.INFO,
                passed=True,
                message="Publication year valid"
            )
        except (ValueError, TypeError):
            return ValidationResult(
                validator="year_valid",
                level=ValidationLevel.INFO,
                passed=False,
                message=f"Invalid year format: {year}"
            )


class PaperSanitizer:
    """Sanitizes paper data before storage"""
    
    @staticmethod
    def sanitize_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize paper data"""
        sanitized = paper.copy()
        
        # Trim whitespace from strings
        for key in ["title", "abstract", "doi"]:
            if key in sanitized and isinstance(sanitized[key], str):
                sanitized[key] = sanitized[key].strip()
        
        # Normalize DOI
        if "doi" in sanitized and sanitized["doi"]:
            sanitized["doi"] = PaperSanitizer._normalize_doi(sanitized["doi"])
        
        # Ensure authors is a list
        if "authors" in sanitized:
            if isinstance(sanitized["authors"], str):
                sanitized["authors"] = [a.strip() for a in sanitized["authors"].split(",")]
            elif not isinstance(sanitized["authors"], list):
                sanitized["authors"] = []
        
        # Convert year to int
        if "year" in sanitized and sanitized["year"]:
            try:
                sanitized["year"] = int(sanitized["year"])
            except (ValueError, TypeError):
                sanitized["year"] = None
        
        # Remove None values
        sanitized = {k: v for k, v in sanitized.items() if v is not None}
        
        return sanitized
    
    @staticmethod
    def _normalize_doi(doi: str) -> str:
        """Normalize DOI format"""
        # Remove common prefixes
        doi = re.sub(r'^(https?:\/\/)?(dx\.)?doi\.org\/', '', doi)
        # Remove whitespace
        doi = doi.strip()
        return doi


class DuplicateDetector:
    """Detects duplicate papers"""
    
    @staticmethod
    def find_duplicates(papers: List[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
        """Find duplicate papers, returns list of (index1, index2, reason)"""
        duplicates = []
        
        for i, paper1 in enumerate(papers):
            for j, paper2 in enumerate(papers[i+1:], start=i+1):
                reason = DuplicateDetector._check_duplicate(paper1, paper2)
                if reason:
                    duplicates.append((i, j, reason))
        
        return duplicates
    
    @staticmethod
    def _check_duplicate(paper1: Dict[str, Any], paper2: Dict[str, Any]) -> Optional[str]:
        """Check if two papers are duplicates"""
        # Check DOI match
        if paper1.get("doi") and paper2.get("doi"):
            if paper1["doi"].lower() == paper2["doi"].lower():
                return "identical_doi"
        
        # Check title similarity (exact match)
        if paper1.get("title") and paper2.get("title"):
            title1 = paper1["title"].lower().strip()
            title2 = paper2["title"].lower().strip()
            if title1 == title2:
                return "identical_title"
        
        return None


def validate_papers_batch(papers: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[ValidationResult]]:
    """
    Validate and sanitize a batch of papers.
    Returns (sanitized_papers, validation_results)
    """
    all_results = []
    sanitized_papers = []
    
    # Sanitize all papers first
    for i, paper in enumerate(papers):
        # Sanitize
        sanitized = PaperSanitizer.sanitize_paper(paper)
        
        # Validate
        results = PaperValidator.validate_paper(sanitized)
        
        # Check for critical failures
        critical_failures = [r for r in results if r.level == ValidationLevel.CRITICAL and not r.passed]
        
        if not critical_failures:
            sanitized_papers.append(sanitized)
        else:
            # Log why this paper was rejected
            for result in critical_failures:
                all_results.append(result)
        
        # Add all non-critical results
        all_results.extend([r for r in results if r.level != ValidationLevel.CRITICAL or r.passed])
    
    # Check for duplicates
    duplicates = DuplicateDetector.find_duplicates(sanitized_papers)
    if duplicates:
        all_results.append(ValidationResult(
            validator="duplicate_detection",
            level=ValidationLevel.WARNING,
            passed=False,
            message=f"Found {len(duplicates)} duplicate papers",
            details={"duplicates": duplicates}
        ))
    
    return sanitized_papers, all_results
