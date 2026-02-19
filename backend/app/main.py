"""
FastAPI main application entry point.
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import engine
from app.core.event_handlers import register_all_handlers
from app.core.exception_handlers import register_exception_handlers
from app.services.docker_service import docker_service
from app.services.filesystem_sync_service import fs_sync_service
from app.services.garbage_collector import cleanup_artifacts
from app.services.snapshot_service import snapshot_service
from app.services.storage_service import storage_service

# Create scheduler
scheduler = AsyncIOScheduler()

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Docker Release Registry API for managing ML model versions and deployments",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json",
)

# Register domain exception handlers
register_exception_handlers(app)

# Configure CORS with specific allowed methods and headers for security
# When allow_credentials=True, origins must be specific (not ["*"])
cors_origins = settings.get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
    expose_headers=["Content-Disposition"],
    max_age=600,  # Cache preflight requests for 10 minutes
)


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    """
    # Register event handlers
    register_all_handlers()

    # Check database connection
    try:
        from sqlalchemy import text
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection failed: {e}")

    # Check Ceph filesystem
    if storage_service.is_ceph_mounted():
        print("Ceph filesystem is mounted and accessible")
    else:
        print("WARNING: Ceph filesystem is not mounted or not accessible")

    # Start scheduler
    scheduler.add_job(cleanup_artifacts, 'interval', days=1)
    # Docker cleanup runs as Celery task (worker has Docker socket access)
    from app.services.task_dispatcher import task_dispatcher
    scheduler.add_job(lambda: task_dispatcher.dispatch_docker_cleanup(), 'interval', days=1)
    scheduler.add_job(snapshot_service.backup, 'interval', hours=1)  # Periodic DB backup
    # Deployment health checks run as Celery task (worker has Docker socket access)
    scheduler.add_job(
        lambda: task_dispatcher.dispatch_health_check(),
        'interval',
        seconds=settings.DEPLOYMENT_HEALTH_CHECK_INTERVAL,
    )
    scheduler.start()
    print("Scheduler started with cleanup, backup, and health check jobs")
    
    # Snapshot Restore & File System Sync
    try:
        print("Checking for snapshot restore...")
        await snapshot_service.restore_if_empty()
        
        print("Starting file system sync...")
        await fs_sync_service.sync_storage()
        
        import asyncio
        print("Syncing archived Docker build jobs (async)...")
        asyncio.create_task(docker_service.sync_archived_jobs())
    except Exception as e:
        print(f"Error during startup sync/restore: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    """
    await engine.dispose()
    print("Application shutdown complete")
    scheduler.shutdown()


@app.get("/api/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status and component checks
    """
    from sqlalchemy import text

    # Check database
    db_healthy = False
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        db_healthy = True
    except Exception:
        pass

    # Check storage filesystem (non-blocking — local storage may not need Ceph)
    storage_healthy = storage_service.is_ceph_mounted()

    # Only DB is required for health; storage is informational
    overall_status = "healthy" if db_healthy else "unhealthy"

    return JSONResponse(
        status_code=status.HTTP_200_OK if overall_status == "healthy" else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": overall_status,
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "storage": "healthy" if storage_healthy else "degraded",
            },
            "version": settings.APP_VERSION,
        }
    )


@app.get("/api/v1/info", status_code=status.HTTP_200_OK)
async def info():
    """
    API information endpoint.

    Returns:
        API version and system information
    """
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "api_version": "v1",
    }


# Include API v1 router
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint.

    Returns:
        Welcome message with API documentation link
    """
    return {
        "message": "Docker Release Registry API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }
