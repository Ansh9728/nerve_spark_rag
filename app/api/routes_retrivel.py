from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from app.services.rag_service import rag_service, RAGResponse
from app.services.embeddings import initialize_vector_store
from fastapi.encoders import jsonable_encoder

retrivel_router = APIRouter()


@retrivel_router.get("/similar_docs")
async def similar_docs(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(
        10, ge=1, le=20, description="Number of similar documents to return"
    ),
    rerank_top_n: int = Query(
        5, ge=1, le=10, description="Number of rerank document to return"
    ),
):
    """Get similar documents based on vector similarity"""
    try:
        vector_store = initialize_vector_store()
        results = vector_store.search_similar(
            query=query, top_k=top_k, rerank_top_n=rerank_top_n
        )

        return JSONResponse(
            content={"query": query, "results_count": len(results), "results": results},
            status_code=200,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@retrivel_router.get("/query", response_model=dict)
async def query_with_context(
    query: str = Query(..., description="Your question about smart buildings"),
    top_k: int = Query(
        10, ge=1, le=15, description="Number of context documents to use"
    ),
    rerank_top_n: int = Query(5, ge=1, le=8, description="Number of rerank documents"),
    include_sources: bool = Query(
        True, description="Include source documents in response"
    ),
):
    """
    Get context-aware responses with retrieval-augmented generation

    Features:
    - Vector similarity search
    - Context-aware generation
    - Performance metrics
    - Confidence scoring
    """
    try:
        # Initialize RAG service if not already done
        await rag_service.initialize()

        # Get context-aware response
        response = await rag_service.query_with_context(query=query, top_k=top_k)

        result = {
            "query": query,
            "answer": response.answer,
            "confidence_score": response.confidence_score,
            "metrics": response.metrics,
            "performance": {
                "retrieval_time_ms": round(response.retrieval_time * 1000, 2),
                "generation_time_ms": round(response.generation_time * 1000, 2),
                "total_time_ms": round(
                    (response.retrieval_time + response.generation_time) * 1000, 2
                ),
            },
        }

        if include_sources:
            # result["sources"] = response.sources
            result["sources"] = jsonable_encoder(response.sources)

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
