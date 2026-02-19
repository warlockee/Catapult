
import asyncio
import os
import uuid
import yaml
import shutil
from sqlalchemy import text
from app.core.database import async_session_maker
from app.models.version import Version
from app.models.docker_build import DockerBuild

async def setup_test():
    async with async_session_maker() as db:
        # Set Schema
        await db.execute(text("SET search_path TO model_registry"))

        # 1. Find a valid version (release)
        result = await db.execute(text("SELECT id FROM versions LIMIT 1"))
        row = result.first()
        if not row:
            print("No versions found in DB. Cannot test sync (requires FK).")
            return

        version_id = row[0]
        print(f"Using Version ID: {version_id}")

        # 2. Generate a fake Job ID
        job_id = uuid.uuid4()
        print(f"Generated Test Job ID: {job_id}")

        # 3. Create Archive on Disk
        job_dir = f"./storage/dockerbuild_jobs/{job_id}"
        os.makedirs(job_dir, exist_ok=True)

        metadata = {
            "job_id": str(job_id),
            "release_id": str(version_id),  # FK still named release_id for backward compat
            "image_tag": "test-sync:v1",
            "build_type": "manual_test",
            "started_at": "2025-01-01T12:00:00",
            "completed_at": "2025-01-01T12:05:00",
            "build_args": []
        }
        
        with open(os.path.join(job_dir, "metadata.yaml"), "w") as f:
            yaml.dump(metadata, f)
            
        with open(os.path.join(job_dir, "Dockerfile"), "w") as f:
            f.write("# Dummy Dockerfile for Sync Test")
            
        print(f"Created archive at {job_dir}")
        
        # 4. Enforce NOT in DB (should be true since UUID is new)
        build = await db.get(DockerBuild, job_id)
        if build:
            print("ERROR: Job already exists in DB? That's impossible for random UUID.")
        else:
            print("Confirmed Job is NOT in DB.")
            
        print("\nSETUP COMPLETE.")
        print(f"Run 'docker compose restart backend' and then check if Job {job_id} appears in DB.")
        
        # Save ID to file for next step
        with open("test_job_id.txt", "w") as f:
            f.write(str(job_id))

if __name__ == "__main__":
    import sys
    # Add backend to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
    asyncio.run(setup_test())
