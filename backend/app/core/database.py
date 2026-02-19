"""
Database connection and session management.
"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


# Create async engine with connection pool settings and statement timeout
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_timeout=30,  # Wait max 30s for connection from pool
    pool_recycle=1800,  # Recycle connections after 30 minutes
    connect_args={
        "command_timeout": 60,  # PostgreSQL statement timeout in seconds
        "server_settings": {
            "statement_timeout": "60000",  # 60 seconds in milliseconds
        },
    },
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# Base class for models
class Base(DeclarativeBase):
    """Base class for all database models."""
    metadata = MetaData(schema=settings.DB_SCHEMA)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
