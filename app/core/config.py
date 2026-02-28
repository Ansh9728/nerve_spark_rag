import os
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # app configuration
    APP_NAME: str = "Smart Building Rag Application"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    API_V1_STR: str = "/v1"

    # Server settings
    HOST: str = os.getenv("HOST", "localhost")
    PORT: int = int(os.getenv("PORT", "7777"))

    # Database settings
    # DATABASE_URI: str = os.getenv("DATABASE_URI", "")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # processing folders
    INCOMING_DIR: str = os.getenv("INCOMING_DIR", "incoming_docs")
    PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", "processed_docs")

    # scheduler settings
    SCHEDULER_INTERVAL_SECONDS: int = int(
        os.getenv("SCHEDULER_INTERVAL_SECONDS", "300")
    )
    SCHEDULER_ENABLED: bool = os.getenv("SCHEDULER_ENABLED", "True").lower() in (
        "true",
        "1",
        "t",
    )

    # pinecone settings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-s"
    )
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "nerve-rag")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    # openai
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", None)

    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL")
    OLLAMA_TIMEOUT: int = 120

    GROQ_API_KEY:str = os.getenv("GROQ_API_KEY")  # or os.getenv("GROQ_API_KEY")
    GROQ_MODEL_NAME: str= os.getenv("GROQ_MODEL_NAME")
    
    
    #streammlit app
    API_BASE_URL: str = os.getenv("API_BASE_URL", f"http://{os.getenv('HOST')}:{int(os.getenv('PORT', '7777'))}")

    class Config:
        "pydantic config"

        env_file = ".env"
        extra = "ignore"


settings = Settings()
