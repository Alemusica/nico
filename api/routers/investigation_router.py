"""
Investigation Router - Endpoints for investigation workflow

Handles:
- Investigation WebSocket streaming
- Investigation briefing creation and confirmation
- Investigation status checks
"""

from fastapi import APIRouter, WebSocket, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import hashlib

# Import Investigation Agent
try:
    from src.agent.investigation_agent import InvestigationAgent
    INVESTIGATION_AGENT_AVAILABLE = True
except ImportError:
    INVESTIGATION_AGENT_AVAILABLE = False
    InvestigationAgent = None

# Import Data Manager
try:
    from src.data_manager.manager import DataManager, InvestigationBriefing
    from src.data_manager.config import ResolutionConfig, TemporalResolution, SpatialResolution
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False
    DataManager = None

router = APIRouter(prefix="/investigate", tags=["investigation"])

# Global data manager instance
_data_manager = None


def get_data_manager() -> DataManager:
    """Get or create data manager instance."""
    global _data_manager
    if _data_manager is None and DATA_MANAGER_AVAILABLE:
        _data_manager = DataManager()
    return _data_manager


async def get_knowledge_service(backend: str = "surrealdb"):
    """Get knowledge service instance."""
    from api.services.knowledge_service import create_knowledge_service, KnowledgeBackend
    
    # Convert string to enum
    backend_enum = KnowledgeBackend.SURREALDB if backend == "surrealdb" else KnowledgeBackend.NEO4J
    return create_knowledge_service(backend_enum)



# ============== MODELS ==============

class ResolutionRequest(BaseModel):
    """Request model for resolution configuration."""
    temporal: str = "daily"  # hourly, 3-hourly, 6-hourly, daily, monthly
    spatial: str = "0.25"    # 0.1, 0.25, 0.5, 1.0


class BriefingRequest(BaseModel):
    """Request model for investigation briefing."""
    query: str
    location_name: str
    location_bbox: List[float]  # [lat_min, lat_max, lon_min, lon_max]
    event_type: str
    event_start: str  # YYYY-MM-DD
    event_end: str    # YYYY-MM-DD
    precursor_days: int = 30
    resolution: Optional[ResolutionRequest] = None


class BriefingConfirmation(BaseModel):
    """Request model for confirming a briefing."""
    briefing_id: str
    confirmed: bool = True
    modified_resolution: Optional[ResolutionRequest] = None


# ============== ENDPOINTS ==============

@router.websocket("/ws")
async def websocket_investigate(websocket: WebSocket):
    """WebSocket for streaming investigation progress."""
    await websocket.accept()
    
    try:
        # Receive investigation request
        data = await websocket.receive_text()
        request = json.loads(data)
        
        if not INVESTIGATION_AGENT_AVAILABLE:
            await websocket.send_text(json.dumps({
                "step": "error",
                "status": "error",
                "message": "Investigation Agent not available"
            }))
            await websocket.close()
            return
        
        # Get knowledge service for storing papers (use SurrealDB)
        knowledge_service = None
        try:
            backend = request.get("backend", "surrealdb")
            knowledge_service = await get_knowledge_service(backend)
            
            # Connect to the service (important!)
            await knowledge_service.connect()
            print(f"✅ Knowledge service connected ({backend})")
        except Exception as e:
            print(f"Warning: Could not initialize knowledge service: {e}")
            # Continue without knowledge service - papers won't be saved but investigation can proceed
        
        # Create agent and run streaming investigation
        agent = InvestigationAgent(knowledge_service=knowledge_service)
        
        # Extract resolution config if provided
        temporal_resolution = request.get("temporal_resolution", "daily")
        spatial_resolution = request.get("spatial_resolution", "0.25")
        
        async for progress in agent.investigate_streaming(
            query=request.get("query", ""),
            collect_satellite=request.get("collect_satellite", True),
            collect_reanalysis=request.get("collect_reanalysis", True),
            collect_climate_indices=request.get("collect_climate_indices", True),
            collect_papers=request.get("collect_papers", True),
            collect_news=request.get("collect_news", False),
            run_correlation=request.get("run_correlation", True),
            expand_search=request.get("expand_search", True),
            temporal_resolution=temporal_resolution,
            spatial_resolution=spatial_resolution,
        ):
            await websocket.send_text(json.dumps(progress))
        
        await websocket.close()
        
    except Exception as e:
        import traceback
        await websocket.send_text(json.dumps({
            "step": "error",
            "status": "error",
            "message": f"Investigation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }))
        await websocket.close()


@router.get("/status")
async def get_investigation_status():
    """Get investigation components status."""
    status = {
        "agent_available": INVESTIGATION_AGENT_AVAILABLE,
        "components": {}
    }
    
    if INVESTIGATION_AGENT_AVAILABLE:
        try:
            from src.agent.tools.geo_resolver import GeoResolver
            status["components"]["geo_resolver"] = True
        except ImportError:
            status["components"]["geo_resolver"] = False
            
        try:
            from src.surge_shazam.data.cmems_client import CMEMSClient
            status["components"]["cmems_client"] = True
        except ImportError:
            status["components"]["cmems_client"] = False
            
        try:
            from src.surge_shazam.data.era5_client import ERA5Client
            status["components"]["era5_client"] = True
        except ImportError:
            status["components"]["era5_client"] = False
            
        try:
            from src.agent.tools.literature_scraper import LiteratureScraper
            status["components"]["literature_scraper"] = True
        except ImportError:
            status["components"]["literature_scraper"] = False
    
    return status


@router.post("/briefing")
async def create_investigation_briefing(request: BriefingRequest):
    """
    Create an investigation briefing for user confirmation.
    
    This is the Ishikawa-style planning step where LLM proposes
    what data to collect and user confirms before download.
    """
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    manager = get_data_manager()
    
    # Parse resolution if provided
    resolution = None
    if request.resolution:
        resolution = ResolutionConfig(
            temporal=TemporalResolution(request.resolution.temporal),
            spatial=SpatialResolution(request.resolution.spatial),
        )
    
    # Create briefing
    briefing = manager.create_briefing(
        query=request.query,
        location_name=request.location_name,
        location_bbox=tuple(request.location_bbox),
        event_type=request.event_type,
        event_time_range=(request.event_start, request.event_end),
        precursor_days=request.precursor_days,
        resolution=resolution,
    )
    
    # Store briefing for later confirmation
    briefing_id = hashlib.md5(f"{request.query}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    # Store in simple in-memory cache (for now)
    # TODO: Move to proper state management or database
    if not hasattr(router, "_briefings"):
        router._briefings = {}
    router._briefings[briefing_id] = briefing
    
    return {
        "briefing_id": briefing_id,
        "briefing": briefing.to_dict(),
        "summary": briefing.summary(),
    }


@router.post("/briefing/{briefing_id}/confirm")
async def confirm_briefing(briefing_id: str, confirmation: BriefingConfirmation):
    """
    Confirm a briefing and start download.
    
    User has reviewed the briefing and confirmed (or adjusted) the parameters.
    Now we actually download the data.
    """
    if not DATA_MANAGER_AVAILABLE:
        raise HTTPException(503, "Data Manager not available")
    
    # Retrieve briefing
    if not hasattr(router, "_briefings") or briefing_id not in router._briefings:
        raise HTTPException(404, f"Briefing {briefing_id} not found")
    
    briefing = router._briefings[briefing_id]
    
    if not confirmation.confirmed:
        # User canceled
        del router._briefings[briefing_id]
        return {"status": "canceled", "message": "Briefing canceled by user"}
    
    # Apply modified resolution if provided
    if confirmation.modified_resolution:
        briefing.resolution = ResolutionConfig(
            temporal=TemporalResolution(confirmation.modified_resolution.temporal),
            spatial=SpatialResolution(confirmation.modified_resolution.spatial),
        )
    
    # Start data collection (async job)
    manager = get_data_manager()
    
    # TODO: Make this truly async with background tasks
    try:
        datasets = await manager.collect_briefing_data(briefing)
        
        return {
            "status": "completed",
            "briefing_id": briefing_id,
            "datasets_collected": len(datasets),
            "cache_paths": [str(path) for path in datasets.values()],
        }
    except Exception as e:
        raise HTTPException(500, f"Data collection failed: {str(e)}")
    finally:
        # Clean up briefing
        if briefing_id in router._briefings:
            del router._briefings[briefing_id]
