# this will file read document from directry on timely basis and create embedding and store into database
import os
import asyncio
import aiofiles
import shutil
from datetime import datetime, timezone
from app.core.config import settings
from app.core.logging import logger

INCOMING_DIR = settings.INCOMING_DIR
PROCESSED_DIR = settings.PROCESSED_DIR

async def process_file(filename: str):
    """ process each file """
    
    file_path = os.path.join(INCOMING_DIR, filename)
    
    async with aiofiles.open(file_path, mode='r', encoding='utf-8', errors='replace') as f:
        content = await f.read()
        
    # handle creation emedding and storing in vectordatbase
    
    
    #  end
    shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))
    logger.info(f"✅ Processed & moved: {filename}")
    

async def process_documents():
    logger.info(f"[{datetime.now(timezone.utc)}] checking for new documents")
    
    os.makedirs(INCOMING_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(INCOMING_DIR) if os.path.isfile(os.path.join(INCOMING_DIR, f))]
    
    if not files:
        logger.info("NO document find for processing")
        return 
    
    # process all files
    await asyncio.gather(*(process_file(f) for f in files))
    
