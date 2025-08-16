"""
RAG (Retrieval-Augmented Generation) service for context-aware responses
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.core.logging import logger
from app.services.embeddings import initialize_vector_store
import openai
from openai import OpenAI
from app.core.config import settings

@dataclass
class RAGResponse:
    """Response structure for RAG queries"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    retrieval_time: float
    generation_time: float
    total_tokens: int
    metrics: Dict[str, Any]


class RAGService:
    def __init__(self):
        self.vector_store = None
        self.openai_client = None
        
    async def initialize(self):
        """Initialize the RAG service"""
        self.vector_store = initialize_vector_store()
        
        # Initialize OpenAI client if API key is available
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_client = OpenAI(
                api_key=settings.OPENAI_API_KEY
            )
    
    def calculate_confidence_score(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on retrieval scores"""
        if not sources:
            return 0.0
        
        # Weighted average of similarity scores
        scores = []
        for source in sources:
            if source and isinstance(source, dict):
                scores.append(source.get('score', 0))
        return sum(scores) / len(scores) if scores else 0.0
    
    def format_context(self, sources: List[Dict[str, Any]]) -> str:
        """Format retrieved sources into context string"""
        if not sources:
            return "No relevant context found."
        
        context_parts = []
        for i, source in enumerate(sources, 1):
            content = source.get('content', '')
            metadata = source.get('metadata', {})
            source_file = metadata.get('source_file', 'Unknown')
            
            context_parts.append(
                f"[Source {i} - {source_file}]: {content[:500]}..."
            )
        
        return "\n\n".join(context_parts)
    
    async def generate_response(self, query: str, context: str) -> Dict[str, Any]:
        """Generate context-aware response using OpenAI"""
        if not self.openai_client:
            return {
                "answer": "OpenAI API not configured. Please provide context manually.",
                "tokens": 0
            }
            
        prompt = f"""You are a question-answering system. 
        Use ONLY the provided context to answer the question. 

        If the answer is not present in the context, respond with:
        "⚠️ The answer is not present in the knowledge base."

        Context:
        {context}

        Question:
        {query}

        Instructions:
        - Base your answer strictly on the given context.
        - Do not add external knowledge or assumptions.
        - Be concise and accurate.
        - If multiple relevant parts exist, summarize them clearly.
        - If no relevant details exist, state the answer is not present in the knowledge base.

        Answer:
        """


        # prompt = f"""You are a specialized assistant for smart building management. 
        # You answer questions only using the provided context. If the context does not 
        # contain the answer, explicitly respond with:

        # "⚠️ The answer is not present in the knowledge base."

        # Context (from building documents, IoT logs, and manuals):
        # {context}

        # User Question:
        # {query}

        # Guidelines:
        # - ONLY use the given context to answer.
        # - If no relevant details exist in the context, say the answer is not present in the knowledge base.
        # - Be precise, concise, and technical when needed.
        # - If multiple sources give overlapping information, summarize consistently.
        # - If relevant, point to the source (e.g., 'from maintenance manual', 'from sensor logs').
        # - Avoid speculation or adding extra knowledge outside the given context.

        # Answer:
        # """


        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for smart building management."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=500,
                # temperature=0.7
            )
            
            logger.info(f"OpenAI Response : {response}")
            
            return {
                "answer": response.choices[0].message.content,
                "tokens": response.usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "tokens": 0
            }
    
    async def query_with_context(self, query: str, top_k: int = 10, rerank_top_n:int = 5) -> RAGResponse:
        """
        Main RAG query method with context-aware generation and metrics
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            sources = self.vector_store.search_similar(query, top_k=top_k, rerank_top_n=rerank_top_n)
            
            # logger.info(f"sources : {sources}")
            
            retrieval_time = time.time() - retrieval_start
            
            # Step 2: Calculate confidence score
            confidence_score = self.calculate_confidence_score(sources)
            
            # Step 3: Format context
            context = self.format_context(sources)
            
            # Step 4: Generate response
            generation_start = time.time()
            generation_result = await self.generate_response(query, context)
            generation_time = time.time() - generation_start
            
            # Step 5: Calculate metrics
            total_time = time.time() - start_time
            
            metrics = {
                "retrieval_count": len(sources),
                "avg_similarity_score": confidence_score,
                "context_length": len(context),
                "query_length": len(query),
                "total_processing_time": total_time
            }
            
            return RAGResponse(
                answer=generation_result["answer"],
                sources=sources,
                confidence_score=confidence_score,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_tokens=generation_result["tokens"],
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                confidence_score=0.0,
                retrieval_time=0.0,
                generation_time=0.0,
                total_tokens=0,
                metrics={"error": str(e)}
            )

# Global RAG service instance
rag_service = RAGService()
