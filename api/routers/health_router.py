"""
🏥 Health Monitoring Router
============================
System health checks and status endpoints.
"""

from fastapi import APIRouter
from api.services.llm_service import get_llm_service

router = APIRouter()


@router.get("/")
async def root():
    """
    Simple API health check.
    
    Returns a basic status indicator to verify the API is running.
    
    **Response Example:**
    ```json
    {
      "status": "ok",
      "service": "Causal Discovery API",
      "version": "1.0.0"
    }
    ```
    """
    return {"status": "ok", "service": "Causal Discovery API", "version": "1.0.0"}


@router.get("/health")
async def health():
    """
    Detailed health check with component status.
    
    Checks all system components and returns their availability status.
    The system continues operating even when components are unavailable
    due to built-in fallback mechanisms.
    
    **Components Checked:**
    - **LLM Service**: Ollama availability (falls back to rule-based)
    - **Causal Discovery**: Tigramite/PCMCI (falls back to correlation)
    - **Databases**: Neo4j and SurrealDB (fall back to in-memory/JSON)
    
    **Response Example:**
    ```json
    {
      "status": "healthy",
      "version": "1.0.0",
      "components": {
        "llm": {
          "status": "available",
          "model": "llama3.1:8b-instruct-q8_0"
        },
        "causal_discovery": {
          "status": "available",
          "method": "PCMCI"
        },
        "databases": {
          "neo4j": "connected",
          "surrealdb": "available (not connected)"
        }
      },
      "robustness": "All components have fallbacks - system operational"
    }
    ```
    
    **Status Values:**
    - `available` / `connected`: Component fully operational
    - `fallback (...)`: Using fallback mechanism
    - `unavailable`: Component not accessible (system continues)
    """
    llm = get_llm_service()
    llm_available = await llm.check_availability()
    
    # Check tigramite availability
    try:
        from tigramite import data_processing as pp
        tigramite_status = "available"
    except ImportError:
        tigramite_status = "fallback (correlation)"
    
    # Check database connections
    neo4j_status = "fallback (in-memory)"
    surrealdb_status = "fallback (json)"
    
    try:
        from neo4j import GraphDatabase
        # Try to connect if env vars set
        import os
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        try:
            driver = GraphDatabase.driver(uri, auth=("neo4j", os.getenv("NEO4J_PASSWORD", "causalpass123")))
            driver.verify_connectivity()
            neo4j_status = "connected"
            driver.close()
        except Exception:
            pass
    except ImportError:
        pass
    
    try:
        import surrealdb
        surrealdb_status = "available (not connected)"
    except ImportError:
        pass
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "api_version": "v1",
        "components": {
            "llm": {
                "status": "available" if llm_available else "fallback (rules)",
                "model": llm.config.model if llm_available else None
            },
            "causal_discovery": {
                "status": tigramite_status,
                "method": "PCMCI" if tigramite_status == "available" else "cross-correlation"
            },
            "databases": {
                "neo4j": neo4j_status,
                "surrealdb": surrealdb_status
            }
        },
        "robustness": "All components have fallbacks - system operational"
    }
