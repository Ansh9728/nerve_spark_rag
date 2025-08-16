from fastapi import APIRouter, Query, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
from app.core.logging import logger
from app.ingestions.folder_ingest import process_documents
from app.core.scheduler import scheduler
from app.services.rag_service import rag_service, RAGResponse
from app.services.document_services import document_service

ingest_router = APIRouter()


@ingest_router.post("/process-documents")
async def trigger_manual_processing():
    """Manually trigger document processing."""
    await process_documents()
    return JSONResponse(
        content={"message": "Document processing triggered successfully"},
        status_code=200,
    )


@ingest_router.get("/scheduler/status")
async def get_scheduler_status():
    """Get the current scheduler status."""
    return JSONResponse(
        content={
            "running": scheduler.is_running(),
            "interval_seconds": scheduler.interval_seconds,
        },
        status_code=200,
    )


@ingest_router.post("/upload-multiple-files")
async def upload_multiple_files(files: list[UploadFile] = File(...)):
    """
    Upload multiple files and create embeddings for each.

    This endpoint accepts multiple files and processes them sequentially.
    Supported file types: PDF, DOCX, TXT, MD, and other text-based formats
    """
    try:
        result = await document_service.upload_multiple_files(files)
        return JSONResponse(content=result, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
