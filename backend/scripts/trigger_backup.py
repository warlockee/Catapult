
import asyncio
import logging

from app.services.snapshot_service import snapshot_service

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    print("Triggering manual backup...")
    path = await snapshot_service.backup()
    if path:
        print(f"Backup SUCCESS: {path}")
    else:
        print("Backup FAILED")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
