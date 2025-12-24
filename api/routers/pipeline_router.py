from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)

# ========================
# Models
# ========================
class FullPipelineRequest(BaseModel):
    topics: Optional[List[str]] = None
    paper_queries: Optional[List[str]] = None
    max_per_topic: int = 5

# ========================
# PIPELINE
# ========================
@router.post("/run")
async def run_full_pipeline(request: FullPipelineRequest):
    """
    Run the full research pipeline:
    1. SCRAPE: papers from Copernicus
    2. REFINE: extract causal patterns with LLM
    3. CORRELATE: find region-gate correlations
    4. SCORE: generate hypotheses for investigation

    Request:
        topics: List of research topics (default: ["Arctic amplification", "Sea ice loss", ...])
        paper_queries: Custom search queries (overrides topics)
        max_per_topic: Max papers per topic (default: 5)

    Returns:
        summary: Pipeline execution summary with all results
    """
    try:
        from src.pipeline.research_pipeline import ResearchPipeline
        pipeline = ResearchPipeline()

        topics_to_use = request.topics or [
            "Arctic amplification",
            "Sea ice loss",
            "Ocean heat transport",
            "Atlantic Water inflow",
            "Barents Sea Opening dynamics"
        ]

        if request.paper_queries:
            result = await pipeline.run(
                paper_queries=request.paper_queries,
                max_per_topic=request.max_per_topic
            )
        else:
            result = await pipeline.run(
                topics=topics_to_use,
                max_per_topic=request.max_per_topic
            )

        return {"summary": result}
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
