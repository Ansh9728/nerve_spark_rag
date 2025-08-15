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
    DATABASE_URI: str = os.getenv('DATABASE_URI')
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    
    # processing folders
    INCOMING_DIR: str = os.getenv('INCOMING_DIR', 'incoming_docs')
    PROCESSED_DIR: str = os.getenv('PROCESSED_DIR', 'processed_docs')
    
    # scheduler settings
    SCHEDULER_INTERVAL_SECONDS: int = int(os.getenv('SCHEDULER_INTERVAL_SECONDS', '300'))
    SCHEDULER_ENABLED: bool = os.getenv('SCHEDULER_ENABLED', 'True').lower() in ('true', '1', 't')
    
    
    class Config:
        "pydantic config"
        env_file=".env"
        extra='ignore'
    
        
settings = Settings()
    