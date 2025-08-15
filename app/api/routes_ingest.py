from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.ingestions.folder_ingest import process_documents
from app.core.scheduler import scheduler

ingest_router = APIRouter()

@ingest_router.get('/pdf')
def test():
    return JSONResponse(content="testing this endpoint", status_code=200)

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
