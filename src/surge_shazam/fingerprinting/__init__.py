"""
🎵 Surge Shazam Fingerprinting Module
=====================================

Pattern fingerprinting for extreme event precursor detection.
Like Shazam for weather patterns.

Components:
    - engine: MiniRocket-based fingerprint extraction
    - database: SurrealDB storage for fingerprints
    - matcher: Similarity search
    - spectrogram: (optional) STFT-based features
    - hasher: (optional) LSH for fast lookup
    - peaks: (optional) Peak detection
"""

from .engine import (
    FingerprintEngine,
    Fingerprint,
    SimilarityResult,
    extract_fingerprint,
    compare_fingerprints,
    VARIABLE_WEIGHTS,
    VARIABLE_GROUPS,
)

from .database import (
    FingerprintDB,
    get_fingerprint_db,
    store_fingerprint,
    search_fingerprints,
)

from .matcher import (
    FingerprintMatcher,
    MatchResult,
    MatchType,
    Alert,
    AlertSeverity,
    MatcherConfig,
    get_matcher,
    quick_match,
)

__all__ = [
    # Engine
    "FingerprintEngine",
    "Fingerprint",
    "SimilarityResult",
    "extract_fingerprint",
    "compare_fingerprints",
    "VARIABLE_WEIGHTS",
    "VARIABLE_GROUPS",
    # Database
    "FingerprintDB",
    "get_fingerprint_db",
    "store_fingerprint",
    "search_fingerprints",
    # Matcher
    "FingerprintMatcher",
    "MatchResult",
    "MatchType",
    "Alert",
    "AlertSeverity",
    "MatcherConfig",
    "get_matcher",
    "quick_match",
]
