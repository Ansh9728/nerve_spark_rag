import asyncio
from datetime import datetime, timezone
from typing import Optional
import logging
from app.ingestions.folder_ingest import process_documents
from app.core.config import settings
from app.core.logging import logger

class DocumentScheduler:
    """Scheduler for automatically processing documents at regular intervals."""
    
    def __init__(self, interval_seconds: int = None):
        """
        Initialize the scheduler.
        
        Args:
            interval_seconds: Time between processing runs (uses settings if None)
        """
        self.interval_seconds = interval_seconds or settings.SCHEDULER_INTERVAL_SECONDS
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler is already running")
            return
            
        self._running = True
        self._task = asyncio.create_task(self._run_scheduler())
        logger.info(f"Document scheduler started with {self.interval_seconds}s interval")
        
    async def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return
            
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Document scheduler stopped")
        
    async def _run_scheduler(self):
        """Main scheduler loop."""
        while self._running:
            try:
                logger.info(f"[{datetime.now(timezone.utc)}] Running scheduled document processing...")
                await process_documents()
            except Exception as e:
                logger.error(f"Error during scheduled processing: {e}")
                
            # Wait for the next interval
            await asyncio.sleep(self.interval_seconds)
            
    def is_running(self) -> bool:
        """Check if the scheduler is currently running."""
        return self._running

# Global scheduler instance
scheduler = DocumentScheduler()
