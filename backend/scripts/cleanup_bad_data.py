#!/usr/bin/env python3
"""
Script to clean up bad data (models/releases named .git)
"""
import asyncio
import sys
from pathlib import Path

from sqlalchemy import delete

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import async_session_maker
from app.models import Model, Release


async def cleanup():
    async with async_session_maker() as session:
        print("🧹 Cleaning up bad data...")

        # Delete releases with version '.git'
        stmt = delete(Release).where(Release.version == '.git')
        result = await session.execute(stmt)
        print(f"  Deleted {result.rowcount} releases with version '.git'")

        # Delete releases with version '__pycache__'
        stmt = delete(Release).where(Release.version == '__pycache__')
        result = await session.execute(stmt)
        print(f"  Deleted {result.rowcount} releases with version '__pycache__'")

        # Delete releases with version '.cache'
        stmt = delete(Release).where(Release.version == '.cache')
        result = await session.execute(stmt)
        print(f"  Deleted {result.rowcount} releases with version '.cache'")

        # Delete models with name '.git'
        stmt = delete(Model).where(Model.name == '.git')
        result = await session.execute(stmt)
        print(f"  Deleted {result.rowcount} models with name '.git'")
        
        await session.commit()
        print("✨ Cleanup complete!")

if __name__ == "__main__":
    asyncio.run(cleanup())
