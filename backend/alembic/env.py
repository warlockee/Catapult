"""Alembic environment configuration."""
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

# Import settings and Base
from app.core.config import settings
from app.core.database import Base

# Import all models so they're registered with Base
from app.models.model import Model
from app.models.version import Version
from app.models.deployment import Deployment
from app.models.api_key import ApiKey
from app.models.audit_log import AuditLog
from app.models.artifact import Artifact
from app.models.docker_build import DockerBuild
from app.models.docker_build_artifact import DockerBuildArtifact
from app.models.benchmark import Benchmark
from app.models.evaluation import Evaluation

# Alembic Config object
config = context.config

# Override sqlalchemy.url with our DATABASE_URL
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata object for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=settings.DB_SCHEMA,
        include_schemas=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with connection."""
    # Ensure schema exists and set search path
    connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.DB_SCHEMA}"))
    connection.execute(text(f"SET search_path TO {settings.DB_SCHEMA}"))
    connection.commit()  # Commit schema creation

    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=settings.DB_SCHEMA,
        include_schemas=True,
        transaction_per_migration=False,  # Use single transaction for all migrations
    )

    with context.begin_transaction():
        context.run_migrations()

    # Explicit commit to ensure alembic_version is updated
    connection.commit()


async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = settings.DATABASE_URL

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
