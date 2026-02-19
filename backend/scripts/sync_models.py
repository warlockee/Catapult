#!/usr/bin/env python3
"""
Script to sync models from storage to database.
Usage: python scripts/sync_models.py
"""
import asyncio
import os
import sys
from pathlib import Path

from sqlalchemy import select

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import configure_mappers

from app.core.config import settings
from app.core.database import async_session_maker
from app.models import Model, Release

# Force mapper configuration
configure_mappers()


async def sync_models():
    """Sync models from storage to database."""
    storage_root = Path(settings.CEPH_MOUNT_PATH) / settings.MODEL_STORAGE_DIR
    
    if not storage_root.exists():
        print(f"❌ Storage directory not found: {storage_root}")
        return

    print(f"📂 Scanning storage directory: {storage_root}")
    
    async with async_session_maker() as session:
        # Iterate over model directories
        for model_dir in storage_root.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
                
            model_name = model_dir.name
            print(f"  Found model directory: {model_name}")
            
            # Check/Create Model
            stmt = select(Model).where(Model.name == model_name)
            result = await session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if not model:
                print(f"    ➕ Creating new model: {model_name}")
                model = Model(
                    name=model_name,
                    storage_path=str(model_dir.relative_to(Path(settings.CEPH_MOUNT_PATH))),
                    description=f"Automatically imported from {model_dir.name}"
                )
                session.add(model)
                await session.commit()
                await session.refresh(model)
            else:
                print(f"    ✅ Model exists: {model_name}")

            # Determine versions
            versions = []
            
            # Heuristic: Check for files directly in model dir
            has_files = any(f.is_file() for f in model_dir.iterdir())
            
            if has_files:
                # Treat as "latest" version
                versions.append(("latest", model_dir))
            
            # Check for subdirectories (explicit versions)
            for version_dir in model_dir.iterdir():
                if version_dir.is_dir() and not version_dir.name.startswith(".") and version_dir.name not in ["__pycache__", ".cache"]:
                    versions.append((version_dir.name, version_dir))
            
            if not versions:
                print(f"    ⚠️  No versions found for {model_name}")
                continue
                
            for version, path in versions:
                # Check/Create Release
                stmt = select(Release).where(
                    Release.image_id == model.id,
                    Release.version == version
                )
                result = await session.execute(stmt)
                release = result.scalar_one_or_none()
                
                if not release:
                    print(f"      ➕ Creating release: {version}")
                    
                    # Calculate size
                    total_size = 0
                    for root, _, files in os.walk(path):
                        for f in files:
                            try:
                                total_size += (Path(root) / f).stat().st_size
                            except (OSError, FileNotFoundError):
                                pass
                            
                    release = Release(
                        image_id=model.id,
                        version=version,
                        tag=version,
                        digest=f"sha256:imported-{model_name}-{version}", # Placeholder
                        size_bytes=total_size,
                        status="active",
                        ceph_path=str(path.relative_to(Path(settings.CEPH_MOUNT_PATH))),
                        release_notes="Automatically imported",
                        is_release=False
                    )
                    session.add(release)
                    await session.commit()
                else:
                    print(f"      ✅ Release exists: {version}")

    print("\n✨ Sync complete!")


if __name__ == "__main__":
    asyncio.run(sync_models())
