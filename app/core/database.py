from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.logging import logger
from app.core.config import settings


class PostgresDB:
    
    def __init__(self):
        self.DATABASE_URI = settings.DATABASE_URI
        self.engine = create_async_engine(self.DATABASE_URI, echo=settings.DEBUG, future=True)
        self.SessionLocal = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)
        self.Base = declarative_base()
        
    async def get_db(self):
        """Get a database session."""
        async with self.SessionLocal() as session:
            yield session
            
    async def connect(self):
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(self.Base.metadata.create_all)
                logger.info("Connected to PostgresSQL database.")
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            
    async def close(self):
        await self.engine.dispose()
        logger.info("Closed PostgreSQL connection. ")
            

postgres_db = PostgresDB()