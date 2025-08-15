from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from app.ingestions.folder_ingest import process_documents
from app.core.scheduler import scheduler
from app.services.rag_service import rag_service, RAGResponse
from app.services.embeddings import initialize_vector_store

ingest_router = APIRouter()


@ingest_router.post('/process-documents')
async def trigger_manual_processing():
    """Manually trigger document processing."""
    await process_documents()
    return JSONResponse(content={"message": "Document processing triggered successfully"}, status_code=200)


@ingest_router.get('/scheduler/status')
async def get_scheduler_status():
    """Get the current scheduler status."""
    return JSONResponse(content={
        "running": scheduler.is_running(),
        "interval_seconds": scheduler.interval_seconds
    }, status_code=200)

