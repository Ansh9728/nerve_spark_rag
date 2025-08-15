import os
import uuid
from typing import List, Dict, Any, Optional
from app.core.logging import logger
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings
from pinecone import Pinecone, ServerlessSpec
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import hashlib

def get_embedding_model(model_name="Snowflake/snowflake-arctic-embed-s"):
    """
    Returns an embedding model for encoding text into vector embeddings using HuggingFace models.
    Args:
        model_name (str, optional): The name of the HuggingFace model to be used for generating embeddings. 
        Defaults to "Snowflake/snowflake-arctic-embed-s".
    Returns:
        HuggingFaceEmbeddings: An embedding model with normalized embeddings enabled for cosine similarity computation.
    """
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    base_embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

    return base_embedding_model


# setup pinecone
def setup_pinecone():
    """Initialize Pinecone client and ensure the index exists."""
    try:
        # Initialize Pinecone with modern API
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Check if index exists
        index_list = pc.list_indexes()
        
        if settings.PINECONE_INDEX_NAME not in index_list.names():
            logger.info(f"Creating index: {settings.PINECONE_INDEX_NAME}")
            # Create serverless index
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            while not pc.describe_index(settings.PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            
            logger.info(f"Index {settings.PINECONE_INDEX_NAME} created successfully")
        
        # Connect to the index
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        logger.info(f"Connected to index: {settings.PINECONE_INDEX_NAME}")
        
        return index
    except Exception as e:
        logger.error(f"Failed to setup Pinecone: {str(e)}")
        raise


class StoreIntoVectorDatabase:
    def __init__(self, pinecone_index, embedding_model):
        self.pinecone_index = pinecone_index
        self.embedding_model = embedding_model
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=220,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def create_chunks(self, content: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Create chunks from content with metadata
        Args:
            content (str): The text content to chunk
            metadata (Dict[str, Any]): Additional metadata for the chunks
        Returns:
            List[Document]: List of LangChain Document objects
        """
        if metadata is None:
            metadata = {}
            
        # Create documents with metadata
        documents = [Document(page_content=content, metadata=metadata)]
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks from content")
        return chunks
    
    def generate_chunk_id(self, content: str, source: str, chunk_index: int) -> str:
        """Generate a unique ID for each chunk"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{source}_{chunk_index}_{content_hash}"
    
    async def store_embeddings(self, chunks: List[Document], source_file: str = None) -> bool:
        """
        Generate embeddings and store in Pinecone
        Args:
            chunks (List[Document]): List of document chunks
            source_file (str): Source file name for tracking
        Returns:
            bool: Success status
        """
        try:
            if not chunks:
                logger.warning("No chunks provided for embedding storage")
                return False
            
            # Prepare texts and metadata
            texts = [chunk.page_content for chunk in chunks]
            metadatas = []
            
            for idx, chunk in enumerate(chunks):
                metadata = chunk.metadata.copy()
                metadata.update({
                    "chunk_index": idx,
                    "source_file": source_file or "unknown",
                    "chunk_id": self.generate_chunk_id(chunk.page_content, source_file or "unknown", idx),
                    "page_content": chunk.page_content
                })
                metadatas.append(metadata)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Prepare vectors for Pinecone
            vectors = []
            for idx, (embedding, text, metadata) in enumerate(zip(embeddings, texts, metadatas)):
                vector_id = str(uuid.uuid4())
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Store in Pinecone
            logger.info(f"Storing {len(vectors)} vectors in Pinecone...")
            self.pinecone_index.upsert(vectors=vectors)
            
            logger.info(f"Successfully stored {len(vectors)} embeddings in Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            return False
    
    async def process_and_store_content(self, content: str, source_file: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Complete pipeline: chunk content, generate embeddings, and store in Pinecone
        Args:
            content (str): The text content to process
            source_file (str): Source file name
            metadata (Dict[str, Any]): Additional metadata
        Returns:
            bool: Success status
        """
        try:
            # Create chunks
            chunks = self.create_chunks(content, metadata)
            
            if not chunks:
                logger.warning(f"No chunks created from {source_file}")
                return False
            
            # Store embeddings
            success = await self.store_embeddings(chunks, source_file)
            
            if success:
                logger.info(f"Successfully processed and stored content from {source_file}")
            else:
                logger.error(f"Failed to process content from {source_file}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error in process_and_store_content: {str(e)}")
            return False
    
    async def delete_by_source(self, source_file: str) -> bool:
        """
        Delete all vectors associated with a source file
        Args:
            source_file (str): Source file name to delete
        Returns:
            bool: Success status
        """
        try:
            # Delete vectors with matching source_file in metadata
            self.pinecone_index.delete(
                filter={"source_file": {"$eq": source_file}}
            )
            logger.info(f"Deleted all vectors for source: {source_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors for {source_file}: {str(e)}")
            return False
    
    def search_similar(self, query: str, top_k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            filter_dict (Dict[str, Any]): Optional filters
        Returns:
            List[Dict[str, Any]]: Search results with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search in Pinecone
            results = self.pinecone_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "id": match.id,
                    "score": match.score,
                    "content": match.metadata.get("page_content", ""),
                    "metadata": match.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

# Initialize components
def initialize_vector_store():
    """Initialize the vector store components"""
    try:
        # Setup embedding model
        embedding_model = get_embedding_model(settings.EMBEDDING_MODEL)
        
        # Setup Pinecone
        pinecone_index = setup_pinecone()
        
        # Create vector store instance
        vector_store = StoreIntoVectorDatabase(pinecone_index, embedding_model)
        
        logger.info("Vector store initialized successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise
