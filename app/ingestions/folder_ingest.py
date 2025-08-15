# this will file read document from directry on timely basis and create embedding and store into database
import os
import asyncio
import aiofiles
import shutil
from datetime import datetime, timezone
from app.core.config import settings
from app.core.logging import logger
from app.services.embeddings import initialize_vector_store

INCOMING_DIR = settings.INCOMING_DIR
PROCESSED_DIR = settings.PROCESSED_DIR

# Initialize vector store
vector_store = None

async def get_vector_store():
    """Get or initialize the vector store"""
    global vector_store
    if vector_store is None:
        vector_store = initialize_vector_store()
    return vector_store

async def process_file(filename: str):
    """Process each file: read, chunk, embed, and store in vector database"""
    
    file_path = os.path.join(INCOMING_DIR, filename)
    
    try:
        async with aiofiles.open(file_path, mode='r', encoding='utf-8', errors='replace') as f:
            content = await f.read()
        
        if not content.strip():
            logger.warning(f"Empty file: {filename}")
            return
            
        # Get vector store instance
        vs = await get_vector_store()
        
        # Prepare metadata
        metadata = {
            "filename": filename,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "file_size": len(content)
        }
        
        # Process content: chunk, embed, and store
        success = await vs.process_and_store_content(
            content=content,
            source_file=filename,
            metadata=metadata
        )
        
        if success:
            # Move file to processed directory only if successful
            shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))
            logger.info(f"✅ Processed & moved: {filename}")
        else:
            logger.error(f"❌ Failed to process: {filename}")
            
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")

async def process_documents():
    logger.info(f"[{datetime.now(timezone.utc)}] checking for new documents")
    
    os.makedirs(INCOMING_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(INCOMING_DIR) if os.path.isfile(os.path.join(INCOMING_DIR, f))]
    
    if not files:
        logger.info("NO document find for processing")
        return 
    
    # Process all files
    await asyncio.gather(*(process_file(f) for f in files))
    
