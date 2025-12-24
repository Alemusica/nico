"""
⚙️ Application Configuration
============================
Pydantic Settings for environment-based configuration.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # === Application ===
    app_name: str = "Causal Discovery API"
    app_version: str = "1.1.0"
    debug: bool = False
    
    # === Server ===
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # === CORS ===
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    # === LLM Service ===
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout: int = 120
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # === Databases ===
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "causalpass123"
    neo4j_enabled: bool = False
    
    surrealdb_url: str = "ws://localhost:8001"
    surrealdb_namespace: str = "causal"
    surrealdb_database: str = "knowledge"
    surrealdb_user: str = "root"
    surrealdb_password: str = "root"
    surrealdb_enabled: bool = True
    
    # === Data Management ===
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    max_cache_size_mb: int = 10000  # 10 GB
    cache_ttl_days: int = 30
    
    # === Investigation Agent ===
    investigation_timeout: int = 300  # 5 minutes
    max_papers_per_query: int = 20
    enable_web_scraping: bool = True
    
    # === Causal Discovery ===
    tigramite_enabled: bool = True
    default_alpha_level: float = 0.05
    default_max_lag: int = 7
    pcmci_max_conds_dim: int = 3
    
    # === API Rate Limiting ===
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    rate_limit_storage: str = "memory://"
    rate_limit_strategy: str = "fixed-window"
    
    # === Logging ===
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    log_file: Optional[Path] = None
    
    # === Security ===
    secret_key: str = "change-this-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # === External Services ===
    copernicus_user: Optional[str] = None
    copernicus_password: Optional[str] = None
    era5_api_key: Optional[str] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment (for testing)."""
    global _settings
    _settings = Settings()
    return _settings
