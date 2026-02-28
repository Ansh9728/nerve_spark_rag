# this will file read document from directry on timely basis and create embedding and store into database
import os
import asyncio
import shutil
from datetime import datetime, timezone
from typing import List
from app.core.config import settings
from app.core.logging import logger
from app.services.embeddings import initialize_vector_store
from app.utils.document_loaders import DocumentProcessor
from langchain_core.documents import Document

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


async def process_document_with_langchain(file_path: str, filename: str) -> bool:
    """Process document using LangChain loaders based on file type.

    Returns True if at least one page produced embeddings, False otherwise.
    """
    try:
        # Load document using appropriate LangChain loader
        documents = DocumentProcessor.load_document(file_path)
        
        if not documents:
            logger.warning(f"No content extracted from: {filename}")
            return False
        
        # Get vector store instance
        vs = await get_vector_store()
        
        # Process each document (page/chunk) separately
        success_count = 0
        for idx, doc in enumerate(documents):
            # skip pages with no text
            if not doc.page_content or not doc.page_content.strip():
                logger.warning(f"Empty page content for {filename} page {idx + 1}, skipping")
                continue

            # Prepare metadata combining file info and document metadata
            metadata = {
                "filename": filename,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "file_type": doc.metadata.get("file_type", "unknown"),
                "page_number": doc.metadata.get("page", idx + 1),
                "source_file": filename,
                "chunk_index": idx,
                **doc.metadata
            }
            
            # Process content: chunk, embed, and store
            success = await vs.process_and_store_content(
                content=doc.page_content,
                source_file=f"{filename}_chunk_{idx}",
                metadata=metadata
            )
            
            if success:
                success_count += 1
        
        logger.info(f"✅ Processed {success_count}/{len(documents)} chunks from {filename}")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error processing {filename} with LangChain: {str(e)}")
        return False


async def process_file_legacy(filename: str) -> bool:
    """Legacy processing for plain text files (backward compatibility).

    Returns True if processing succeeded, False otherwise.
    """
    file_path = os.path.join(INCOMING_DIR, filename)
    
    try:
        import aiofiles
        async with aiofiles.open(file_path, mode='r', encoding='utf-8', errors='replace') as f:
            content = await f.read()
        
        if not content.strip():
            logger.warning(f"Empty file: {filename}")
            return False
            
        # Get vector store instance
        vs = await get_vector_store()
        
        # Prepare metadata
        metadata = {
            "filename": filename,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "file_size": len(content),
            "file_type": "text"
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
            return True
        else:
            logger.error(f"❌ Failed to process: {filename}")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return False


async def process_file(filename: str):
    """Process each file using appropriate method based on file type."""
    file_path = os.path.join(INCOMING_DIR, filename)
    
    # Check file extension
    file_extension = os.path.splitext(filename)[1].lower()
    supported_extensions = DocumentProcessor.get_supported_extensions()
    
    if file_extension in supported_extensions:
        # Use LangChain loaders for supported file types
        success = await process_document_with_langchain(file_path, filename)
    else:
        # Fall back to legacy text processing
        logger.info(f"Using legacy processing for: {filename}")
        success = await process_file_legacy(filename)
    
    if success:
        # Move file to processed directory only if successful
        try:
            shutil.move(file_path, os.path.join(PROCESSED_DIR, filename))
            logger.info(f"✅ Processed & moved: {filename}")
        except Exception as e:
            logger.error(f"Failed to move file {filename}: {str(e)}")
    else:
        logger.error(f"❌ Failed to process: {filename}")
    
    return success


async def process_documents():
    logger.info(f"[{datetime.now(timezone.utc)}] checking for new documents")
    
    os.makedirs(INCOMING_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(INCOMING_DIR) if os.path.isfile(os.path.join(INCOMING_DIR, f))]
    
    if not files:
        logger.info("NO document find for processing")
        return 
    
    # Log supported file types
    supported_extensions = DocumentProcessor.get_supported_extensions()
    logger.info(f"Supported file types: {supported_extensions}")
    
    # Process all files and record statuses
    results = await asyncio.gather(*(process_file(f) for f in files))
    for fname, status in zip(files, results):
        if status:
            logger.info(f"{fname}: processed successfully")
        else:
            logger.warning(f"{fname}: processing failed or no embeddings generated")
