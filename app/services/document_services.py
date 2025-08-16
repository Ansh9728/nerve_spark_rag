import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from fastapi import UploadFile, HTTPException
from app.core.config import settings
from app.core.logging import logger
from app.ingestions.folder_ingest import process_file


class DocumentService:
    """Service for handling document uploads and processing."""
    
    def __init__(self):
        self.incoming_dir = Path(settings.INCOMING_DIR)
        self.incoming_dir.mkdir(parents=True, exist_ok=True)
    
    def create_safe_filename(self, filename: str) -> str:
        """Create a safe filename by replacing spaces and special characters."""
        return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    async def save_uploaded_file(self, file: UploadFile) -> Path:
        """Save uploaded file to incoming directory."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        safe_filename = self.create_safe_filename(file.filename)
        file_path = self.incoming_dir / safe_filename
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"File saved successfully: {safe_filename}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file {safe_filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    async def process_uploaded_file(self, filename: str) -> Dict[str, Any]:
        """Process a single uploaded file and create embeddings."""
        try:
            await process_file(filename)
            logger.info(f"Embeddings created successfully for: {filename}")
            
            return {
                "filename": filename,
                "status": "success",
                "message": "File processed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            
            # Clean up failed file
            file_path = self.incoming_dir / filename
            if file_path.exists():
                file_path.unlink()
                
            raise HTTPException(
                status_code=500,
                detail=f"File uploaded but processing failed: {str(e)}"
            )
    
    async def upload_single_file(self, file: UploadFile) -> Dict[str, Any]:
        """Upload and process a single file."""
        file_path = await self.save_uploaded_file(file)
        filename = file_path.name
        
        result = await self.process_uploaded_file(filename)
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": filename,
            "file_path": str(file_path),
            "status": "processed"
        }
    
    async def upload_multiple_files(self, files: List[UploadFile]) -> Dict[str, Any]:
        """Upload and process multiple files."""
        results = []
        
        for file in files:
            try:
                if not file.filename:
                    continue
                
                file_path = await self.save_uploaded_file(file)
                filename = file_path.name
                
                result = await self.process_uploaded_file(filename)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": str(e)
                })
        
        return {
            "message": f"Processed {len(results)} files",
            "results": results
        }


# Create singleton instance
document_service = DocumentService()
