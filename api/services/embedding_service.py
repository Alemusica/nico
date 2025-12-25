"""
Embedding Service for semantic search.

Uses sentence-transformers for local embedding generation.
Supports batch processing for efficiency.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy load model to avoid startup delay
_model = None
_model_name = "all-MiniLM-L6-v2"  # Fast, 384 dimensions


def get_model():
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {_model_name}")
            _model = SentenceTransformer(_model_name)
            logger.info(f"✅ Embedding model loaded (dim={_model.get_sentence_embedding_dimension()})")
        except ImportError:
            logger.warning("sentence-transformers not installed. Embeddings disabled.")
            return None
    return _model


async def generate_embedding(text: str) -> Optional[list[float]]:
    """Generate embedding for a single text."""
    model = get_model()
    if model is None:
        return None
    
    # Run in thread to not block event loop
    embedding = await asyncio.to_thread(model.encode, text)
    return embedding.tolist()


async def generate_embeddings_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """Generate embeddings for multiple texts efficiently."""
    model = get_model()
    if model is None:
        return [None] * len(texts)
    
    # Batch encode in thread
    embeddings = await asyncio.to_thread(model.encode, texts, show_progress_bar=True)
    return [emb.tolist() for emb in embeddings]


async def generate_paper_embedding(title: str, abstract: str, keywords: list[str] = None) -> Optional[list[float]]:
    """Generate embedding for a paper from its metadata."""
    # Combine title, abstract, and keywords
    parts = [title]
    if abstract:
        parts.append(abstract[:1000])  # Limit abstract length
    if keywords:
        parts.append(" ".join(keywords))
    
    text = " | ".join(parts)
    return await generate_embedding(text)


async def add_embeddings_to_papers(db, limit: int = None) -> int:
    """Add embeddings to papers that don't have them yet."""
    from surrealdb import Surreal
    
    # Get papers without embeddings
    query = "SELECT id, title, abstract, keywords FROM paper WHERE embedding = NONE"
    if limit:
        query += f" LIMIT {limit}"
    
    result = await asyncio.to_thread(db.query, query)
    papers = result if result else []
    
    if not papers:
        logger.info("All papers already have embeddings")
        return 0
    
    logger.info(f"📊 Generating embeddings for {len(papers)} papers...")
    
    # Prepare texts for batch embedding
    texts = []
    for paper in papers:
        title = paper.get("title", "") or ""
        abstract = (paper.get("abstract", "") or "")[:1000]
        kw = paper.get("keywords") or []
        keywords = " ".join(kw) if isinstance(kw, list) else ""
        texts.append(f"{title} | {abstract} | {keywords}")
    
    # Generate embeddings in batch
    embeddings = await generate_embeddings_batch(texts)
    
    # Update papers with embeddings
    updated = 0
    for paper, embedding in zip(papers, embeddings):
        if embedding:
            paper_id = paper.get("id")
            # Extract record ID string
            if hasattr(paper_id, 'key'):
                paper_id = f"paper:{paper_id.key}"
            elif hasattr(paper_id, 'record_id'):
                paper_id = f"paper:{paper_id.record_id}"
            
            await asyncio.to_thread(
                db.query,
                f"UPDATE {paper_id} SET embedding = $embedding;",
                {"embedding": embedding}
            )
            updated += 1
    
    logger.info(f"✅ Added embeddings to {updated} papers")
    return updated


async def semantic_search(db, query: str, limit: int = 10) -> list[dict]:
    """Search papers by semantic similarity."""
    # Generate query embedding
    query_embedding = await generate_embedding(query)
    if not query_embedding:
        return []
    
    # Vector similarity search using SurrealDB
    result = await asyncio.to_thread(
        db.query,
        """
        SELECT 
            id, title, abstract, year, authors, keywords,
            vector::similarity::cosine(embedding, $query_embedding) AS score
        FROM paper
        WHERE embedding != NONE
        ORDER BY score DESC
        LIMIT $limit
        """,
        {"query_embedding": query_embedding, "limit": limit}
    )
    
    return result if result else []
