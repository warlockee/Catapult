import os
import shutil
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, unquote

from app.core.config import settings
from app.core.database import async_session_maker
from sqlalchemy import text

logger = logging.getLogger(__name__)


def parse_database_url(db_url: str) -> dict:
    """
    Parse database URL using urllib.parse for robust handling of special characters.

    Args:
        db_url: Database URL in format postgresql+asyncpg://user:pass@host:port/dbname

    Returns:
        dict with user, password, host, port, dbname keys
    """
    # Remove the async driver prefix if present (e.g., postgresql+asyncpg -> postgresql)
    if "+asyncpg" in db_url:
        db_url = db_url.replace("+asyncpg", "")

    parsed = urlparse(db_url)

    return {
        "user": unquote(parsed.username) if parsed.username else "",
        "password": unquote(parsed.password) if parsed.password else "",
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port) if parsed.port else "5432",
        "dbname": parsed.path.lstrip("/") if parsed.path else "",
    }

class SnapshotService:
    def __init__(self):
        self.snapshot_dir = Path(settings.SNAPSHOT_DIR)
        
        # Maximum number of backups to keep
        self.max_backups = 5

    def _ensure_dir(self):
         try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
         except Exception:
             pass

    async def backup(self) -> Optional[str]:
        """
        Create a database backup using pg_dump.
        Returns the path to the backup file if successful, None otherwise.
        """
        self._ensure_dir()
        timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
        filename = f"db_backup_{timestamp}.sql"
        filepath = self.snapshot_dir / filename
        
        # Parse database URL to get connection details
        try:
            db_config = parse_database_url(settings.DATABASE_URL)

            # Prepare environment with password
            env = os.environ.copy()
            env["PGPASSWORD"] = db_config["password"]

            # Construct command
            # Dump ONLY the application schema
            cmd = [
                "pg_dump",
                "-h", db_config["host"],
                "-p", db_config["port"],
                "-U", db_config["user"],
                "-d", db_config["dbname"],
                "-n", settings.DB_SCHEMA,  # Only dump the specific schema
                "-c",            # Clean (drop) schema prior to create
                "--if-exists",   # Use drop table if exists
                "-f", str(filepath)
            ]
            
            logger.info(f"Starting database backup (schema={settings.DB_SCHEMA}) to {filepath}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Backup failed: {stderr.decode()}")
                return None
            
            logger.info(f"Backup completed successfully: {filepath}")
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Backup failed with exception: {e}")
            return None

    def _cleanup_old_backups(self):
        """Keep only the last N backups."""
        try:
            backups = sorted(self.snapshot_dir.glob("db_backup_*.sql"), key=os.path.getmtime)
            if len(backups) > self.max_backups:
                to_remove = backups[:-self.max_backups]
                for p in to_remove:
                    try:
                        p.unlink()
                        logger.info(f"Removed old backup: {p}")
                    except Exception as e:
                        logger.error(f"Failed to remove old backup {p}: {e}")
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")

    async def restore_if_empty(self) -> bool:
        """
        Restore the latest snapshot if the database is empty.
        Returns True if restore was performed, False otherwise.
        """
        self._ensure_dir()

        # Serialize using advisory lock to prevent race conditions (e.g. multiple workers)
        async with async_session_maker() as db:
            # 12345 is an arbitrary lock ID for restore synchronization
            await db.execute(text("SELECT pg_advisory_lock(12345)"))
            try:
                # 1. Check if DB is empty (IN THE TARGET SCHEMA)
                is_empty = False
                schema = settings.DB_SCHEMA
                try:
                    # Check multiple tables to be sure
                    # We strictly query the target schema
                    result = await db.execute(text(f"SELECT count(*) FROM {schema}.releases"))
                    releases_count = result.scalar()
                    
                    result = await db.execute(text(f"SELECT count(*) FROM {schema}.models"))
                    models_count = result.scalar()

                    result = await db.execute(text(f"SELECT count(*) FROM {schema}.api_keys"))
                    keys_count = result.scalar()
                    
                    if releases_count == 0 and models_count == 0 and keys_count == 0:
                        is_empty = True
                except Exception as e:
                    await db.rollback()
                    # Only consider empty if tables don't exist (fresh install or schema missing)
                    if "does not exist" in str(e):
                        logger.warning(f"Tables do not exist in schema '{schema}', treating DB as empty/fresh.")
                        is_empty = True
                    else:
                        logger.error(f"Failed to check DB state: {e}. Skipping restore for safety.")
                        is_empty = False

                if not is_empty:
                    logger.info("Database is not empty (in target schema). Skipping restore.")
                    return False

                # 1.5 CHECK FOR LEGACY DATA IN PUBLIC BEFORE RESTORING
                # If we have live data in public, we prefer migrating it over restoring an old backup
                move_tables = ["models", "releases", "artifacts", "docker_builds", "deployments", "api_keys", "audit_logs", "alembic_version"]
                try:
                    check_public = await db.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'releases'"))
                    if check_public.scalar() > 0:
                        # Public tables exist. Check if they have data.
                        p_count = await db.execute(text("SELECT count(*) FROM public.releases"))
                        if p_count.scalar() > 0:
                            logger.info("Found existing data in 'public' schema. Migrating to target schema instead of restoring backup...")
                            # Create target schema if needed
                            await db.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                            
                            for tbl in move_tables:
                                try:
                                    # Move table to new schema
                                    await db.execute(text(f"ALTER TABLE public.{tbl} SET SCHEMA {schema}"))
                                except Exception as e:
                                    logger.warning(f"Could not migrate table {tbl}: {e}")
                            
                            await db.commit()
                            logger.info("Legacy data migration completed successfully.")
                            return True
                except Exception as e:
                    logger.error(f"Failed during legacy data check: {e}") 

                # 2. Find latest snapshot
                if not self.snapshot_dir.exists():
                     return False

                backups = sorted(self.snapshot_dir.glob("db_backup_*.sql"), key=os.path.getmtime, reverse=True)
                if not backups:
                    logger.info("No snapshots found in storage. Skipping restore.")
                    return False
                    
                latest_backup = backups[0]
                logger.info(f"Found snapshot: {latest_backup}. Attempting to restore...")
                
                # 3. Restore
                try:
                    db_config = parse_database_url(settings.DATABASE_URL)

                    env = os.environ.copy()
                    env["PGPASSWORD"] = db_config["password"]

                    # WIPE Schema to prevent duplicates (Fail-safe against append behavior)
                    # TARGET: strict schema wipe.
                    logger.info(f"Wiping schema '{schema}' before restore...")
                    wipe_cmd = [
                        "psql",
                        "-h", db_config["host"],
                        "-p", db_config["port"],
                        "-U", db_config["user"],
                        "-d", db_config["dbname"],
                        "-c", f"DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema}; GRANT ALL ON SCHEMA {schema} TO registry;"
                    ]
                    
                    wipe_proc = await asyncio.create_subprocess_exec(
                        *wipe_cmd,
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await wipe_proc.communicate()
                    
                    if wipe_proc.returncode != 0:
                        logger.warning("Failed to wipe schema, attempting restore anyway.")

                    # psql command to restore
                    cmd = [
                        "psql",
                        "-h", db_config["host"],
                        "-p", db_config["port"],
                        "-U", db_config["user"],
                        "-d", db_config["dbname"],
                        "-f", str(latest_backup)
                    ]
                    
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        env=env,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode != 0:
                        logger.error(f"Restore failed: {stderr.decode()}")
                        return False
                    
                    logger.info("Database restore command completed.")

                    # 4. POST-RESTORE MIGRATION for Legacy Backups
                    # If the backup was from 'public' schema (legacy), the tables are now in 'public', NOT 'model_registry'.
                    # We need to detect this and move them.
                    try:
                        move_tables = ["models", "releases", "artifacts", "docker_builds", "deployments", "api_keys", "audit_logs", "alembic_version"]
                        
                        # Check: Do tables exist in target schema?
                        check_schema = await db.execute(text(f"SELECT count(*) FROM information_schema.tables WHERE table_schema = '{schema}' AND table_name = 'releases'"))
                        if check_schema.scalar() == 0:
                            logger.info(f"Restore finished but tables missing in '{schema}'. Checking 'public' for legacy data...")
                            
                            check_public = await db.execute(text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'releases'"))
                            if check_public.scalar() > 0:
                                logger.info("Found legacy data in 'public'. Migrating to target schema...")
                                for tbl in move_tables:
                                    try:
                                        await db.execute(text(f"ALTER TABLE public.{tbl} SET SCHEMA {schema}"))
                                    except Exception as e:
                                        logger.warning(f"Could not migrate table {tbl} (might not exist): {e}")
                                await db.commit()
                                logger.info("Legacy data migration completed.")
                    except Exception as e:
                         logger.error(f"Post-restore migration failed: {e}")

                    return True
                    
                except Exception as e:
                    logger.error(f"Restore failed with exception: {e}")
                    return False

            finally:
                await db.execute(text("SELECT pg_advisory_unlock(12345)"))

snapshot_service = SnapshotService()
