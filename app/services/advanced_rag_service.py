# """
# Advanced RAG (Retrieval-Augmented Generation) service using LangGraph
# with document grading, query rewriting, and hallucination detection
# """
# import time
# from typing import List, Dict, Any, Optional, Literal, Annotated, TypedDict
# from dataclasses import dataclass
# from app.core.logging import logger
# from app.services.embeddings import initialize_vector_store
# from app.core.config import settings

# # LangChain and LangGraph imports
# from langchain_ollama import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.documents import Document
# from langchain_core.retrievers import BaseRetriever
# from langchain_core.callbacks import CallbackManagerForRetrieverRun
# from langgraph.graph import END, StateGraph, START
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_core.messages import BaseMessage
# from langgraph.graph.message import add_messages
# from langchain import hub

# # Memory for LangGraph
# memory = MemorySaver()

# @dataclass
# class RAGResponse:
#     """Response structure for RAG queries"""
#     answer: str
#     sources: List[Dict[str, Any]]
#     confidence_score: float
#     retrieval_time: float
#     generation_time: float
#     total_tokens: int
#     metrics: Dict[str, Any]


# class PineconeRetriever(BaseRetriever):
#     """LangChain-compatible retriever wrapper for our Pinecone vector store"""

#     vector_store: Any
#     top_k: int = 10
#     rerank_top_n: int = 5

#     def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
#         """Retrieve relevant documents from Pinecone"""
#         results = self.vector_store.search_similar(query, top_k=self.top_k, rerank_top_n=self.rerank_top_n)

#         documents = []
#         for result in results:
#             doc = Document(
#                 page_content=result.get('content', ''),
#                 metadata=result.get('metadata', {})
#             )
#             documents.append(doc)

#         return documents


# def get_llm_model(model_name: str = None):
#     """Get Ollama LLM model"""
#     if model_name is None:
#         model_name = getattr(settings, 'OLLAMA_DEFAULT_MODEL', 'deepseek-r1:latest')

#     model = ChatOllama(
#         temperature=0,
#         # model=model_name,
#         model=settings.OLLAMA_DEFAULT_MODEL,
#         base_url=getattr(settings, 'OLLAMA_HOST', 'http://localhost:11434'),
#         timeout=getattr(settings, 'OLLAMA_TIMEOUT', 120)
#     )
#     return model


# # Global retriever instance
# global_retriever = None

# # Graph State
# class GraphState(TypedDict):
#     """
#     Represent the state of Our Graph

#     Attributes:

#         question: question
#         generation: LLM Generation
#         documents: list of documents
#         retriever: object to retrive data from vectordatabase

#     """

#     question: str
#     generation: str
#     documents: List[Document]
#     intent: str
#     retry_count: int
#     messages: Annotated[List[BaseMessage], add_messages]


# # Define Graph Nodes

# def retrieve(state):
#     """
#     Retrieve documents

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, documents, that contains retrieved documents
#     """
#     logger.info("---RETRIEVE---")

#     question = state['question']
#     global global_retriever

#     # Retrieval
#     documents = global_retriever.invoke(question)
#     return {"documents": documents, "question": question}


# def grade_documents(state):
#     """
#     Determines whether the retrieved documents are relevant to the question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with only filtered relevant documents
#     """

#     logger.info("---CHECK DOCUMENTS RELEVENT TO THE QUESTION")

#     question  = state['question']
#     documents = state['documents']
#     retry_count = state.get('retry_count', 0)


#     prompt = PromptTemplate(

#         template="""You are a grader assessing relevance of a retrieved document to a user question. \n
#         Here is the retrieved document: \n\n {context} \n\n
#         Here is the user question: {question} \n
#         If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
#         Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

#         Respond with only 'yes' or 'no'.""",
#         input_variables=["context", "question"],
#     )

#     llm = get_llm_model()

#     # Store Filtered doc
#     filterd_docs = []

#     for d in documents:
#         chain = prompt | llm | StrOutputParser()
#         response = chain.invoke({"question": question, "context": d.page_content})
#         grade = response.strip().lower()

#         if grade=='yes':
#             logger.info("---GRADE: DOCUMENT RELEVENT")
#             filterd_docs.append(d)

#         else:
#             logger.info("---GRADE: DOCUMENT NOT RELEVENT")
#             continue

#     return {"documents": filterd_docs, 'question': question, 'retry_count': retry_count}


# def generate(state):
#     """
#     Generate Answer

#     Args:
#         state(dict): The current graph state

#     Return:
#         state(dict): New  key added to state, generation, that contains LLM Answers

#     """

#     logger.info('---GENERATE---')

#     question = state['question']

#     documents = state['documents']

#     # Prompt
#     prompt = hub.pull("rlm/rag-prompt")

#     llm = get_llm_model()

#     # Post-processing
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     rag_chain = prompt | llm | StrOutputParser()

#     generation = rag_chain.invoke({"context": documents, "question": question})
#     retry_count = state.get('retry_count', 0)

#     return {"documents": documents, "question": question, "generation": generation, "retry_count": retry_count}


# def transform_query(state):
#     """
#     Transform the query to produce a better question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates question key with a re-phrased question
#     """

#     logger.info("---TRANSFORM QUERY---")
#     question = state["question"]
#     documents = state["documents"]


#     # Prompt
#     system = """You a question re-writer that converts an input question to a better version that is optimized \n
#         for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
#     re_write_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             (
#                 "human",
#                 "Here is the initial question: \n\n {question} \n Formulate an improved question.",
#             ),
#         ]
#     )

#     llm = get_llm_model()

#     question_rewriter = re_write_prompt | llm | StrOutputParser()

#     # Re-write question
#     better_question = question_rewriter.invoke({"question": question})
#     retry_count = state.get('retry_count', 0) + 1
#     logger.info(f"---QUERY REWRITE ATTEMPT {retry_count}---")
#     return {"documents": documents, "question": better_question, "retry_count": retry_count}


# def out_of_context(state):
#     """
#     Determine the user intent. If it is a greeting, reply; otherwise, return "I don't know" and end the graph.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Decision for the next node to call
#     """
#     logger.info("---OUT OF THE CONTEXT")
#     question = state['question']
#     documents = state['documents']

#     # Prompt for intent classification
#     system_message = """You are a question intent classifier. If the question is related to a greeting, return 'yes' and provide a greeting message; otherwise, return 'no'."""

#     intent_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system_message),
#             (
#                 "human",
#                 "Here is the initial question: \n\n {question} \n and Documents {documents}",
#             ),
#         ]
#     )

#     llm = get_llm_model()
#     intent_chain = intent_prompt | llm | StrOutputParser()

#     # Invoke the intent model
#     docs_content = "\n\n".join([doc.page_content for doc in documents[:4]])  # Limit to avoid too long input
#     intent_response = intent_chain.invoke({"documents": docs_content, "question": question})

#     intent_result = intent_response.strip().lower()
#     logger.info(f'intent_result: {intent_result}')
#     retry_count = state.get('retry_count', 0)
#     if 'yes' in intent_result:
#         # Generate and return a greeting message
#         return {"intent": generate_greeting_message(), "question": question, 'documents':documents, 'retry_count': retry_count}
#     else:
#         return {"intent": "Out OF the Context", "question": question, 'documents':documents, 'retry_count': retry_count}


# def generate_greeting_message():
#     """Generates a greeting message."""
#     return "Hello! How can I assist you today?"


# # Edges

# def grade_generation_v_documents_and_question(state):
#     """
#     Determines whether the generation is grounded in the document and answers question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Decision for next node to call
#     """

#     logger.info("---CHECK HALLUCINATIONS---")
#     question = state["question"]
#     documents = state["documents"]
#     generation = state["generation"]



#     # Prompt
#     system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
#         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
#     hallucination_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
#         ]
#     )

#     llm = get_llm_model()
#     hallucination_chain = hallucination_prompt | llm | StrOutputParser()

#     # Format documents
#     docs_content = "\n\n".join([doc.page_content for doc in documents])

#     score = hallucination_chain.invoke(
#        {"documents": docs_content, "generation": generation}
#     )

#     grade = score.strip().lower()


#     # Answer Grader
#     system = """You are a grader assessing whether an answer addresses / resolves a question \n
#         Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
#     answer_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", system),
#             ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
#         ]
#     )

#     answer_chain = answer_prompt | llm | StrOutputParser()


#      # Check hallucination
#     if 'yes' in grade:
#         logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
#         # Check question-answering
#         logger.info("---GRADE GENERATION vs QUESTION---")
#         answer_response = answer_chain.invoke({"question": question, "generation": generation})
#         grade = answer_response.strip().lower()
#         if 'yes' in grade:
#             logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
#             return "useful"
#         else:
#             logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
#             return "not useful"
#     else:
#         retry_count = state.get('retry_count', 0)
#         if retry_count >= 2:
#             logger.info("---MAX RETRIES (2) REACHED, ACCEPTING GENERATION---")
#             return "useful"
#         logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RETRY---")
#         return "not supported"


# def decide_to_generate(state) -> Literal['generate', 'transform_query']:
#     """
#     Determine wheather to generate an answer, or regenerate a question

#     Args:
#         state(dict): The current graph state

#     Returns:
#         str: Binary Decision for next node to call

#     """

#     logger.info("---ASSESS GRADED DOCUMENTS---")
#     state["question"]
#     filtered_documents = state["documents"]
#     retry_count = state.get('retry_count', 0)

#     if not filtered_documents:
#         # All documents have been filtered check_relevance
#         if retry_count >= 2:
#             logger.info("---MAX RETRIES (2) REACHED, GENERATING ANSWER WITH EMPTY DOCS---")
#             return "generate"
#         logger.info(
#             "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
#         )
#         return "transform_query"
#     else:
#         # We have relevant documents, so generate answer
#         logger.info("---DECISION: GENERATE---")
#         return "generate"


# class AdvancedRAGService:
#     def __init__(self):
#         self.vector_store = None
#         self.retriever = None
#         self.app = None

#     async def initialize(self):
#         """Initialize the advanced RAG service"""
#         self.vector_store = initialize_vector_store()

#         # Create LangChain retriever
#         self.retriever = PineconeRetriever(
#             vector_store=self.vector_store,
#             top_k=10,
#             rerank_top_n=5
#         )

#         # Set global retriever
#         global global_retriever
#         global_retriever = self.retriever

#         # Build the graph
#         self._build_graph()

#         logger.info("Advanced RAG service initialized successfully")

#     def _build_graph(self):
#         """Build the LangGraph workflow"""
#         workflow = StateGraph(GraphState)

#         # Define the nodes
#         workflow.add_node("retrieve", retrieve)
#         workflow.add_node("grade_documents", grade_documents)
#         workflow.add_node("generate", generate)
#         workflow.add_node("transform_query", transform_query)
#         workflow.add_node("out_of_context", out_of_context)

#         workflow.add_edge(START, 'retrieve')
#         workflow.add_edge('retrieve', 'grade_documents')
#         workflow.add_conditional_edges(
#             'grade_documents',
#             decide_to_generate,
#             {
#                 "transform_query": "transform_query",
#                 "generate": "generate",
#             },
#         )
#         workflow.add_edge("transform_query", "retrieve")
#         workflow.add_conditional_edges(
#             "generate",
#             grade_generation_v_documents_and_question,
#             {
#                 "not supported": "out_of_context",
#                 "useful": END,
#                 "not useful": "transform_query",
#             },
#         )
#         workflow.add_edge('out_of_context', END)

#         # Compile with memory
#         self.app = workflow.compile(checkpointer=memory)

#     def calculate_confidence_score(self, documents: List[Document]) -> float:
#         """Calculate confidence score based on document relevance"""
#         if not documents:
#             return 0.0
#         # Simple scoring based on number of documents
#         return min(len(documents) * 0.1, 1.0)

#     def format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
#         """Format documents for response"""
#         sources = []
#         for i, doc in enumerate(documents):
#             sources.append({
#                 "id": f"source_{i}",
#                 "content": doc.page_content,
#                 "metadata": doc.metadata,
#                 "score": 1.0  # Simplified scoring
#             })
#         return sources

#     async def query_with_context(self, query: str, top_k: int = 10, rerank_top_n: int = 5) -> RAGResponse:
#         """
#         Main RAG query method using LangGraph pipeline
#         """
#         start_time = time.time()

#         try:
#             # Update retriever parameters
#             self.retriever.top_k = top_k
#             self.retriever.rerank_top_n = rerank_top_n

#             # Run the graph
#             inputs = {"question": query}
#             config = {"configurable": {"thread_id": "1"}}

#             generation = None
#             documents = []

#             for output in self.app.stream(inputs, config=config):
#                 for key, value in output.items():
#                     logger.info(f"Node '{key}' completed")
#                     if key == "generate" and "generation" in value:
#                         generation = value["generation"]
#                         documents = value.get("documents", [])
#                     elif key == "out_of_context" and "intent" in value:
#                         generation = value["intent"]
#                         documents = value.get("documents", [])

#             if generation is None:
#                 generation = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."

#             # Calculate metrics
#             retrieval_time = time.time() - start_time
#             generation_time = 0.0  # Simplified
#             total_time = time.time() - start_time

#             confidence_score = self.calculate_confidence_score(documents)
#             sources = self.format_sources(documents)

#             metrics = {
#                 "retrieval_count": len(documents),
#                 "avg_similarity_score": confidence_score,
#                 "total_processing_time": total_time,
#                 "pipeline_steps": ["retrieve", "grade", "generate", "validate"]
#             }

#             return RAGResponse(
#                 answer=generation,
#                 sources=sources,
#                 confidence_score=confidence_score,
#                 retrieval_time=retrieval_time,
#                 generation_time=generation_time,
#                 total_tokens=0,  # Ollama doesn't provide token counts easily
#                 metrics=metrics
#             )

#         except Exception as e:
#             logger.error(f"Error in advanced RAG query: {str(e)}")
#             return RAGResponse(
#                 answer=f"Error processing query: {str(e)}",
#                 sources=[],
#                 confidence_score=0.0,
#                 retrieval_time=0.0,
#                 generation_time=0.0,
#                 total_tokens=0,
#                 metrics={"error": str(e)}
#             )


# # Global instance
# advanced_rag_service = AdvancedRAGService()



"""
Optimized Advanced RAG (Retrieval-Augmented Generation) service
Low latency + improved grounding + batching + caching
"""

import time
import json
from typing import List, Dict, Any, Literal, Annotated, TypedDict
from dataclasses import dataclass
from cachetools import TTLCache

from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
import os


from app.core.logging import logger
from app.services.embeddings import initialize_vector_store
from app.core.config import settings

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# -------------------------
# GLOBALS
# -------------------------

memory = MemorySaver()
_llm_singleton = None
retrieval_cache = TTLCache(maxsize=1024, ttl=600)  # 10 min cache
global_retriever = None  # Will be set during initialization


# -------------------------
# DATA STRUCTURE
# -------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    retrieval_time: float
    generation_time: float
    total_tokens: int
    metrics: Dict[str, Any]


# -------------------------
# RETRIEVER
# -------------------------

class PineconeRetriever(BaseRetriever):
    vector_store: Any
    top_k: int = 5
    rerank_top_n: int = 3
    similarity_threshold: float = settings.SIMILARITY_THRESHOLD

    async def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun=None
    ) -> List[Document]:

        cache_key = f"{query}|{self.top_k}|{self.rerank_top_n}"
        if cache_key in retrieval_cache:
            return retrieval_cache[cache_key]

        results = await self.vector_store.search_similar(
            query,
            top_k=self.top_k,
            rerank_top_n=self.rerank_top_n
        )

        documents = []
        for r in results:
            score = r.get("score", 1.0)

            # Apply threshold only if score is provided
            if isinstance(score, (int, float)) and score < self.similarity_threshold:
                continue

            documents.append(
                Document(
                    page_content=r.get("content", "")[:4000],
                    metadata=r.get("metadata", {})
                )
            )

        retrieval_cache[cache_key] = documents
        return documents


# -------------------------
# LLM SINGLETON
# -------------------------

def get_llm_model():
    global _llm_singleton

    if _llm_singleton:
        return _llm_singleton

    _llm_singleton = ChatOllama(
        temperature=0,
        model=settings.OLLAMA_DEFAULT_MODEL,
        base_url=getattr(settings, 'OLLAMA_HOST', 'http://localhost:11434'),
        timeout=60,
        # num_predict=300
    )

    # _llm_singleton = ChatGroq(
    #     groq_api_key=settings.GROQ_API_KEY,  # or os.getenv("GROQ_API_KEY")
    #     model_name=settings.GROQ_MODEL_NAME,
    #     temperature=0,
    #     max_tokens=2048,
    # )

    # _llm_singleton = ChatOpenAI(
    #     api_key=settings.OPENAI_API_KEY,
    #     model="gpt-4o-mini",  # fastest + cheapest
    #     temperature=0,
    #     max_tokens=2048,
    # )


    return _llm_singleton


# -------------------------
# GRAPH STATE
# -------------------------

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    retry_count: int
    messages: Annotated[List[BaseMessage], add_messages]


# -------------------------
# GRAPH NODES
# -------------------------

async def retrieve(state):
    logger.info("RETRIEVE")
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    
    # docs = global_retriever.invoke(question)
    docs = await global_retriever._get_relevant_documents(question)
    return {"documents": docs, "question": question, "retry_count": retry_count}


def grade_documents(state):
    """
    Batch document grading (single LLM call)
    """
    logger.info("BATCH GRADE DOCUMENTS")

    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    if not documents:
        return {"documents": [], "question": question, "retry_count": retry_count}

    combined = "\n\n".join(
        f"[DOC {i}]\n{doc.page_content[:500]}"
        for i, doc in enumerate(documents)
    )

    prompt = PromptTemplate(
        template="""You are a relevance grader.

Question:
{question}

Documents:
{documents}

For each document, determine if it's relevant to the question.
Respond with ONLY this format (no markdown, no extra text):
yes
no
yes

One word per line, in the same order as documents.""",
        input_variables=["question", "documents"],
    )

    try:
        llm = get_llm_model()
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "question": question,
            "documents": combined
        })

        # Parse line-by-line responses
        lines = response.strip().split('\n')
        filtered = []
        
        for i, doc in enumerate(documents):
            if i < len(lines):
                grade = lines[i].strip().lower()
                if grade.startswith('y'):
                    filtered.append(doc)
            
        # If no docs filtered, keep at least top 2
        if not filtered and documents:
            filtered = documents[:min(2, len(documents))]
            
    except Exception as e:
        logger.warning(f"Error in grade_documents: {str(e)}, using fallback")
        # Fallback: keep top 2 documents
        filtered = documents[:min(2, len(documents))]

    return {"documents": filtered, "question": question, "retry_count": retry_count}


def generate(state):
    logger.info("GENERATE")

    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    sources = "\n\n".join(
        f"[SOURCE {i}]\n{doc.page_content[:500]}"
        for i, doc in enumerate(documents[:3])
    ) if documents else "No documents available"

    prompt = PromptTemplate(
        template="""You are a factual assistant.

Answer the question using ONLY the provided sources.
If the answer is not present in the sources, say 'I do not have enough information to answer this question.'

Question:
{question}

Sources:
{sources}

Answer:""",
        input_variables=["question", "sources"],
    )

    try:
        llm = get_llm_model()
        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke({
            "question": question,
            "sources": sources
        })
    except Exception as e:
        logger.error(f"Error in generate: {str(e)}")
        answer = f"I encountered an error processing your question: {str(e)}"

    return {
        "generation": answer,
        "documents": documents,
        "question": question,
        "retry_count": retry_count
    }


# -------------------------
# DECISION
# -------------------------

def decide_to_generate(state) -> Literal["generate", "transform"]:
    if not state["documents"]:
        return "generate"
    return "generate"


# -------------------------
# MAIN SERVICE
# -------------------------

class AdvancedRAGService:

    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.app = None

    async def initialize(self):
        self.vector_store = initialize_vector_store()

        self.retriever = PineconeRetriever(
            vector_store=self.vector_store
        )

        global global_retriever
        global_retriever = self.retriever

        self._build_graph()
        logger.info("Optimized Advanced RAG initialized")

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        # workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        self.app = workflow.compile(checkpointer=memory)

    def calculate_confidence_score(self, documents):
        if not documents:
            return 0.2
        return min(0.5 + len(documents) * 0.1, 0.95)

    def format_sources(self, documents):
        return [
            {
                "id": f"source_{i}",
                "content": doc.page_content[:500],
                "metadata": doc.metadata
            }
            for i, doc in enumerate(documents[:3])
        ]

    async def query_with_context(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_n: int = 3
    ) -> RAGResponse:

        start_time = time.time()

        try:
            self.retriever.top_k = top_k
            self.retriever.rerank_top_n = rerank_top_n

            inputs = {"question": query, "retry_count": 0}
            config = {"configurable": {"thread_id": "1"}}

            generation = "Unable to generate response"
            documents = []

            # for output in self.app.stream(inputs, config=config):
            async for output in self.app.astream(inputs, config=config):
                for key, value in output.items():
                    logger.info(f"Node '{key}' completed")
                    if key == "generate":
                        generation = value.get("generation", "No response")
                        documents = value.get("documents", [])

            total_time = time.time() - start_time

            return RAGResponse(
                answer=generation,
                sources=self.format_sources(documents),
                confidence_score=self.calculate_confidence_score(documents),
                retrieval_time=total_time,
                generation_time=0.0,
                total_tokens=0,
                metrics={
                    "retrieved_docs": len(documents),
                    "processing_time": round(total_time, 2)
                }
            )
        except Exception as e:
            logger.error(f"Error in query_with_context: {str(e)}")
            total_time = time.time() - start_time
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                confidence_score=0.0,
                retrieval_time=total_time,
                generation_time=0.0,
                total_tokens=0,
                metrics={"error": str(e)}
            )


advanced_rag_service = AdvancedRAGService()