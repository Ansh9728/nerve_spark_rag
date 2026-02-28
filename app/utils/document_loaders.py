"""
Document loading utilities for different file types using LangChain loaders.
"""
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)
from langchain_core.documents import Document
from app.core.logging import logger

class DocumentProcessor:
    """Handles document loading and processing for different file types."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.csv': 'csv',
        '.txt': 'text',
        '.md': 'text',
        '.docx': 'word',
        '.doc': 'word',
        '.xlsx': 'excel',
        '.xls': 'excel'
    }
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Determine file type based on extension."""
        file_extension = Path(file_path).suffix.lower()
        return DocumentProcessor.SUPPORTED_EXTENSIONS.get(file_extension, 'unknown')
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load PDF documents using PyPDFLoader.

        Pages that contain no extracted text are dropped.  This prevents later
        processing from trying to split/embed empty strings (which results in
        "No chunks created" warnings).
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add file metadata and filter blank pages
            cleaned = []
            blank_count = 0
            for doc in documents:
                # note: page_content might be None or empty string
                content = doc.page_content or ""
                if not content.strip():
                    blank_count += 1
                    continue
                doc.metadata.update({
                    "file_type": "pdf",
                    "source_file": os.path.basename(file_path),
                    "file_path": file_path
                })
                cleaned.append(doc)
            
            if blank_count > 0:
                logger.warning(f"{blank_count} blank page(s) removed from PDF: {file_path}")
            logger.info(f"Loaded {len(cleaned)} nonempty page(s) from PDF: {file_path}")
            return cleaned
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_csv(file_path: str) -> List[Document]:
        """Load CSV documents using CSVLoader."""
        try:
            loader = CSVLoader(
                file_path=file_path,
                encoding='utf-8',
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"'
                }
            )
            documents = loader.load()
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    "file_type": "csv",
                    "source_file": os.path.basename(file_path),
                    "file_path": file_path
                })
            
            logger.info(f"Loaded {len(documents)} rows from CSV: {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """Load text documents using TextLoader."""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    "file_type": "text",
                    "source_file": os.path.basename(file_path),
                    "file_path": file_path
                })
            
            logger.info(f"Loaded text document: {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_word(file_path: str) -> List[Document]:
        """Load Word documents using UnstructuredWordDocumentLoader."""
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    "file_type": "word",
                    "source_file": os.path.basename(file_path),
                    "file_path": file_path
                })
            
            logger.info(f"Loaded Word document: {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading Word document {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_excel(file_path: str) -> List[Document]:
        """Load Excel documents using UnstructuredExcelLoader."""
        try:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            documents = loader.load()
            
            # Add file metadata
            for doc in documents:
                doc.metadata.update({
                    "file_type": "excel",
                    "source_file": os.path.basename(file_path),
                    "file_path": file_path
                })
            
            logger.info(f"Loaded Excel document: {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading Excel document {file_path}: {str(e)}")
            return []
    
    @classmethod
    def load_document(cls, file_path: str) -> List[Document]:
        """
        Load document based on file type using appropriate loader.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List[Document]: List of LangChain Document objects
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        file_type = cls.get_file_type(file_path)
        
        if file_type == 'pdf':
            return cls.load_pdf(file_path)
        elif file_type == 'csv':
            return cls.load_csv(file_path)
        elif file_type == 'text':
            return cls.load_text(file_path)
        elif file_type == 'word':
            return cls.load_word(file_path)
        elif file_type == 'excel':
            return cls.load_excel(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Return list of supported file extensions."""
        return list(cls.SUPPORTED_EXTENSIONS.keys())
