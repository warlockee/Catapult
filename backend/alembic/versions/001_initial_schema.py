"""Initial schema - squashed migration for open-source release

Revision ID: 0001
Revises:
Create Date: 2026-02-19 00:00:00.000000

This is a single squashed migration that creates the complete database schema.
It replaces all previous incremental migrations.

Tables:
  - models: ML model metadata
  - versions: Model versions (was 'releases', renamed)
  - deployments: Deployment records
  - artifacts: Build artifacts (wheels, tarballs, etc.)
  - api_keys: API key authentication with RBAC
  - audit_logs: Operation audit trail
  - docker_builds: Docker image build tracking
  - docker_build_artifacts: Junction table (docker_builds <-> artifacts)
  - benchmarks: Deployment performance benchmarks
  - evaluations: Quality evaluation results
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    schema = 'model_registry'

    # =========================================================================
    # 1. models
    # =========================================================================
    op.create_table(
        'models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('storage_path', sa.String(1000), nullable=False),
        sa.Column('repository', sa.String(500), nullable=True),
        sa.Column('company', sa.String(255), nullable=True),
        sa.Column('base_model', sa.String(100), nullable=True),
        sa.Column('parameter_count', sa.String(50), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('requires_gpu', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('server_type', sa.String(50), nullable=True),
        sa.Column('source', sa.String(50), nullable=False, server_default='filesystem'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        schema=schema,
    )
    # Unique index on name
    op.create_index('ix_models_name', 'models', ['name'], unique=True, schema=schema)
    # Searchable field indexes
    op.create_index('ix_models_company', 'models', ['company'], schema=schema)
    op.create_index('ix_models_base_model', 'models', ['base_model'], schema=schema)
    op.create_index('ix_models_parameter_count', 'models', ['parameter_count'], schema=schema)
    op.create_index('ix_models_server_type', 'models', ['server_type'], schema=schema)
    op.create_index('ix_models_source', 'models', ['source'], schema=schema)
    # GIN index on tags JSONB
    op.create_index('idx_models_tags', 'models', ['tags'], postgresql_using='gin', schema=schema)
    # Check constraint on source
    op.create_check_constraint(
        'ck_models_source',
        'models',
        "source IN ('filesystem', 'manual', 'orphaned')",
        schema=schema,
    )

    # =========================================================================
    # 2. versions (formerly 'releases')
    # =========================================================================
    op.create_table(
        'versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('image_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('version', sa.String(500), nullable=False),
        sa.Column('tag', sa.String(500), nullable=False),
        sa.Column('digest', sa.String(255), nullable=False),
        sa.Column('quantization', sa.String(50), nullable=True),
        sa.Column('size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('platform', sa.String(50), server_default='linux/amd64'),
        sa.Column('architecture', sa.String(50), server_default='amd64'),
        sa.Column('os', sa.String(50), server_default='linux'),
        sa.Column('status', sa.String(50), server_default='active'),
        sa.Column('release_notes', sa.String(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('ceph_path', sa.String(1000), nullable=True),
        sa.Column('mlflow_url', sa.String(1000), nullable=True),
        sa.Column('is_release', sa.Boolean(), server_default='false', nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ['image_id'], [f'{schema}.models.id'],
            name='versions_image_id_fkey', ondelete='CASCADE',
        ),
        schema=schema,
    )
    op.create_index('ix_versions_image_id', 'versions', ['image_id'], schema=schema)
    op.create_index('ix_versions_version', 'versions', ['version'], schema=schema)
    op.create_index('ix_versions_quantization', 'versions', ['quantization'], schema=schema)
    op.create_index('ix_versions_created_at', 'versions', ['created_at'], schema=schema)
    op.create_index('ix_versions_status', 'versions', ['status'], schema=schema)
    op.create_index('ix_versions_is_release', 'versions', ['is_release'], schema=schema)
    # Composite unique index (image_id, version, quantization)
    op.create_index(
        'idx_versions_image_version_quant', 'versions',
        ['image_id', 'version', 'quantization'], unique=True, schema=schema,
    )
    # Composite index for sorted release listings
    op.create_index(
        'ix_versions_is_release_created', 'versions',
        ['is_release', sa.text('created_at DESC')], schema=schema,
    )
    # Check constraint on status
    op.create_check_constraint(
        'ck_versions_status',
        'versions',
        "status IN ('active', 'deprecated', 'archived')",
        schema=schema,
    )

    # =========================================================================
    # 3. api_keys
    # =========================================================================
    op.create_table(
        'api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('prefix', sa.String(10), nullable=True),
        sa.Column('key_hash', sa.String(255), nullable=False, unique=True),
        sa.Column('role', sa.String(20), nullable=False, server_default='operator'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        schema=schema,
    )
    op.create_index('ix_api_keys_name', 'api_keys', ['name'], unique=True, schema=schema)
    op.create_index('ix_api_keys_prefix', 'api_keys', ['prefix'], schema=schema)
    op.create_index('ix_api_keys_role', 'api_keys', ['role'], schema=schema)
    # Check constraint on role
    op.create_check_constraint(
        'ck_api_keys_role',
        'api_keys',
        "role IN ('admin', 'operator', 'viewer')",
        schema=schema,
    )

    # =========================================================================
    # 4. deployments
    # =========================================================================
    op.create_table(
        'deployments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('release_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('environment', sa.String(100), nullable=False),
        sa.Column('cluster', sa.String(255), nullable=True),
        sa.Column('k8s_namespace', sa.String(255), nullable=True),
        sa.Column('endpoint_url', sa.String(500), nullable=True),
        sa.Column('replicas', sa.Integer(), nullable=True),
        sa.Column('deployed_by', sa.String(255), nullable=True),
        sa.Column('api_key_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('deployed_at', sa.DateTime(), nullable=False),
        sa.Column('terminated_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(50), server_default='success'),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
        # Deployment execution fields
        sa.Column('container_id', sa.String(64), nullable=True),
        sa.Column('host_port', sa.Integer(), nullable=True),
        sa.Column('deployment_type', sa.String(20), nullable=False, server_default='metadata'),
        sa.Column('health_status', sa.String(20), nullable=False, server_default='unknown'),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('stopped_at', sa.DateTime(), nullable=True),
        sa.Column('gpu_enabled', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('image_tag', sa.String(500), nullable=True),
        sa.ForeignKeyConstraint(
            ['release_id'], [f'{schema}.versions.id'],
            name='deployments_release_id_fkey', ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['api_key_id'], [f'{schema}.api_keys.id'],
            name='fk_deployments_api_key_id', ondelete='SET NULL',
        ),
        schema=schema,
    )
    op.create_index('ix_deployments_release_id', 'deployments', ['release_id'], schema=schema)
    op.create_index('ix_deployments_environment', 'deployments', ['environment'], schema=schema)
    op.create_index('ix_deployments_cluster', 'deployments', ['cluster'], schema=schema)
    op.create_index('ix_deployments_k8s_namespace', 'deployments', ['k8s_namespace'], schema=schema)
    op.create_index('ix_deployments_deployed_at', 'deployments', ['deployed_at'], schema=schema)
    op.create_index('ix_deployments_status', 'deployments', ['status'], schema=schema)
    op.create_index('ix_deployments_api_key_id', 'deployments', ['api_key_id'], schema=schema)
    op.create_index('ix_deployments_container_id', 'deployments', ['container_id'], schema=schema)
    op.create_index('ix_deployments_host_port', 'deployments', ['host_port'], schema=schema)
    # Check constraint on status
    op.create_check_constraint(
        'ck_deployments_status',
        'deployments',
        "status IN ('success', 'failed', 'pending', 'terminated', 'running')",
        schema=schema,
    )

    # =========================================================================
    # 5. artifacts
    # =========================================================================
    op.create_table(
        'artifacts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('release_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('artifact_type', sa.String(50), nullable=False),
        sa.Column('file_path', sa.String(1000), nullable=False),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('checksum', sa.String(128), nullable=False),
        sa.Column('checksum_type', sa.String(20), server_default='sha256'),
        sa.Column('platform', sa.String(100), nullable=True),
        sa.Column('python_version', sa.String(50), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('uploaded_by', sa.String(100), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ['model_id'], [f'{schema}.models.id'],
            name='fk_artifacts_model_id', ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['release_id'], [f'{schema}.versions.id'],
            name='artifacts_release_id_fkey', ondelete='CASCADE',
        ),
        schema=schema,
    )
    op.create_index('ix_artifacts_model_id', 'artifacts', ['model_id'], schema=schema)
    op.create_index('ix_artifacts_release_id', 'artifacts', ['release_id'], schema=schema)
    op.create_index('ix_artifacts_name', 'artifacts', ['name'], schema=schema)
    op.create_index('ix_artifacts_artifact_type', 'artifacts', ['artifact_type'], schema=schema)
    op.create_index('idx_artifacts_type', 'artifacts', ['artifact_type'], schema=schema)
    op.create_index('ix_artifacts_created_at', 'artifacts', ['created_at'], schema=schema)
    op.create_index('ix_artifacts_deleted_at', 'artifacts', ['deleted_at'], schema=schema)
    # Composite indexes
    op.create_index('ix_artifacts_model_id_name', 'artifacts', ['model_id', 'name'], schema=schema)
    op.create_index('ix_artifacts_release_id_name', 'artifacts', ['release_id', 'name'], schema=schema)
    # Unique partial index on file_path for non-deleted artifacts
    op.execute(f"""
        CREATE UNIQUE INDEX ix_{schema}_artifacts_file_path_unique
        ON {schema}.artifacts(file_path)
        WHERE deleted_at IS NULL;
    """)

    # =========================================================================
    # 6. audit_logs
    # =========================================================================
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('api_key_name', sa.String(255), nullable=True),
        sa.Column('api_key_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50), nullable=False),
        sa.Column('resource_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.Column('ip_address', postgresql.INET(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ['api_key_id'], [f'{schema}.api_keys.id'],
            name='fk_audit_logs_api_key_id', ondelete='SET NULL',
        ),
        schema=schema,
    )
    op.create_index('ix_audit_logs_api_key_name', 'audit_logs', ['api_key_name'], schema=schema)
    op.create_index('ix_audit_logs_api_key_id', 'audit_logs', ['api_key_id'], schema=schema)
    op.create_index('ix_audit_logs_created_at', 'audit_logs', ['created_at'], schema=schema)

    # =========================================================================
    # 7. docker_builds
    # =========================================================================
    op.create_table(
        'docker_builds',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('release_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('artifact_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('artifact_ids', sa.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('image_tag', sa.String(), nullable=False),
        sa.Column('build_type', sa.String(), nullable=False),
        sa.Column('log_path', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('dockerfile_content', sa.Text(), nullable=True),
        sa.Column('server_type', sa.String(50), nullable=True),
        sa.Column('celery_task_id', sa.String(255), nullable=True),
        sa.Column('superseded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('cleaned_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ['release_id'], [f'{schema}.versions.id'],
            name='docker_builds_release_id_fkey',
        ),
        sa.ForeignKeyConstraint(
            ['artifact_id'], [f'{schema}.artifacts.id'],
            name='docker_builds_artifact_id_fkey',
        ),
        schema=schema,
    )
    op.create_index('ix_docker_builds_id', 'docker_builds', ['id'], schema=schema)
    # GC index for finding builds due for cleanup
    op.execute(f"""
        CREATE INDEX ix_{schema}_docker_builds_superseded_at
        ON {schema}.docker_builds(superseded_at)
        WHERE superseded_at IS NOT NULL AND cleaned_at IS NULL;
    """)
    # Check constraint on status
    op.create_check_constraint(
        'ck_docker_builds_status',
        'docker_builds',
        "status IN ('pending', 'building', 'success', 'failed', 'cancelled')",
        schema=schema,
    )
    # Check constraint on build_type
    op.create_check_constraint(
        'ck_docker_builds_build_type',
        'docker_builds',
        "build_type IN ('organic', 'azure', 'test', 'optimized', 'asr-vllm', 'asr-allinone')",
        schema=schema,
    )

    # =========================================================================
    # 8. docker_build_artifacts (junction table)
    # =========================================================================
    op.create_table(
        'docker_build_artifacts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('docker_build_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('artifact_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(
            ['docker_build_id'], [f'{schema}.docker_builds.id'],
            name='docker_build_artifacts_docker_build_id_fkey', ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['artifact_id'], [f'{schema}.artifacts.id'],
            name='docker_build_artifacts_artifact_id_fkey', ondelete='CASCADE',
        ),
        sa.UniqueConstraint('docker_build_id', 'artifact_id', name='uq_docker_build_artifact'),
        schema=schema,
    )
    op.create_index(
        'ix_docker_build_artifacts_build_id', 'docker_build_artifacts',
        ['docker_build_id'], schema=schema,
    )
    op.create_index(
        'ix_docker_build_artifacts_artifact_id', 'docker_build_artifacts',
        ['artifact_id'], schema=schema,
    )

    # =========================================================================
    # 9. benchmarks
    # =========================================================================
    op.create_table(
        'benchmarks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('deployment_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('endpoint_url', sa.String(500), nullable=True),
        sa.Column('production_endpoint_id', sa.Integer(), nullable=True),
        # Configuration
        sa.Column('endpoint_path', sa.String(500), nullable=False, server_default='/health'),
        sa.Column('method', sa.String(10), nullable=False, server_default='GET'),
        sa.Column('concurrent_requests', sa.Integer(), nullable=False, server_default='10'),
        sa.Column('total_requests', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('timeout_seconds', sa.Float(), nullable=False, server_default='30.0'),
        # Execution mode
        sa.Column('execution_mode', sa.String(20), nullable=False, server_default='docker'),
        sa.Column('container_id', sa.String(64), nullable=True),
        sa.Column('log_path', sa.String(500), nullable=True),
        # Status
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('error_message', sa.String(1000), nullable=True),
        # Progress tracking
        sa.Column('current_stage', sa.String(50), nullable=True),
        sa.Column('stage_progress', sa.String(50), nullable=True),
        sa.Column('stages_completed', postgresql.JSONB(), server_default='[]', nullable=False),
        # Latency metrics (milliseconds)
        sa.Column('latency_avg_ms', sa.Float(), nullable=True),
        sa.Column('latency_min_ms', sa.Float(), nullable=True),
        sa.Column('latency_max_ms', sa.Float(), nullable=True),
        sa.Column('latency_p50_ms', sa.Float(), nullable=True),
        sa.Column('latency_p90_ms', sa.Float(), nullable=True),
        sa.Column('latency_p95_ms', sa.Float(), nullable=True),
        sa.Column('latency_p99_ms', sa.Float(), nullable=True),
        # Throughput metrics
        sa.Column('requests_per_second', sa.Float(), nullable=True),
        sa.Column('total_requests_sent', sa.Integer(), nullable=True),
        sa.Column('successful_requests', sa.Integer(), nullable=True),
        sa.Column('failed_requests', sa.Integer(), nullable=True),
        sa.Column('error_rate', sa.Float(), nullable=True),
        # Inference metrics (TTFT / TPS)
        sa.Column('ttft_avg_ms', sa.Float(), nullable=True),
        sa.Column('ttft_min_ms', sa.Float(), nullable=True),
        sa.Column('ttft_max_ms', sa.Float(), nullable=True),
        sa.Column('ttft_p50_ms', sa.Float(), nullable=True),
        sa.Column('ttft_p90_ms', sa.Float(), nullable=True),
        sa.Column('ttft_p95_ms', sa.Float(), nullable=True),
        sa.Column('ttft_p99_ms', sa.Float(), nullable=True),
        sa.Column('tokens_per_second_avg', sa.Float(), nullable=True),
        sa.Column('tokens_per_second_min', sa.Float(), nullable=True),
        sa.Column('tokens_per_second_max', sa.Float(), nullable=True),
        sa.Column('total_tokens_generated', sa.Integer(), nullable=True),
        sa.Column('model_id', sa.String(255), nullable=True),
        # ASR metrics (legacy, kept for backward compatibility)
        sa.Column('wer', sa.Float(), nullable=True),
        sa.Column('cer', sa.Float(), nullable=True),
        sa.Column('asr_samples_evaluated', sa.Integer(), nullable=True),
        sa.Column('asr_dataset_path', sa.String(500), nullable=True),
        # Timing
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        # Metadata
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
        sa.ForeignKeyConstraint(
            ['deployment_id'], [f'{schema}.deployments.id'],
            name='benchmarks_deployment_id_fkey', ondelete='CASCADE',
        ),
        schema=schema,
    )
    op.create_index('ix_benchmarks_deployment_id', 'benchmarks', ['deployment_id'], schema=schema)
    op.create_index('ix_benchmarks_status', 'benchmarks', ['status'], schema=schema)
    op.create_index('ix_benchmarks_created_at', 'benchmarks', ['created_at'], schema=schema)
    op.create_index('ix_benchmarks_production_endpoint_id', 'benchmarks', ['production_endpoint_id'], schema=schema)

    # =========================================================================
    # 10. evaluations
    # =========================================================================
    op.create_table(
        'evaluations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('deployment_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('production_endpoint_id', sa.Integer(), nullable=True),
        sa.Column('endpoint_url', sa.String(500), nullable=True),
        sa.Column('evaluation_type', sa.String(50), nullable=False),
        sa.Column('evaluator_name', sa.String(100), nullable=False),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('error_message', sa.String(1000), nullable=True),
        sa.Column('current_stage', sa.String(50), nullable=True),
        sa.Column('stage_progress', sa.String(50), nullable=True),
        sa.Column('primary_metric', sa.Float(), nullable=True),
        sa.Column('primary_metric_name', sa.String(50), nullable=True),
        sa.Column('secondary_metric', sa.Float(), nullable=True),
        sa.Column('secondary_metric_name', sa.String(50), nullable=True),
        sa.Column('wer', sa.Float(), nullable=True),
        sa.Column('cer', sa.Float(), nullable=True),
        sa.Column('samples_total', sa.Integer(), nullable=True),
        sa.Column('samples_evaluated', sa.Integer(), nullable=True),
        sa.Column('samples_with_errors', sa.Integer(), nullable=True),
        sa.Column('no_speech_count', sa.Integer(), nullable=True),
        sa.Column('dataset_path', sa.String(500), nullable=True),
        sa.Column('dataset_name', sa.String(255), nullable=True),
        sa.Column('config', postgresql.JSONB(), server_default='{}', nullable=False),
        sa.Column('results', postgresql.JSONB(), server_default='{}', nullable=False),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ['deployment_id'], [f'{schema}.deployments.id'],
            name='evaluations_deployment_id_fkey', ondelete='CASCADE',
        ),
        schema=schema,
    )
    op.create_index('ix_evaluations_deployment_id', 'evaluations', ['deployment_id'], schema=schema)
    op.create_index('ix_evaluations_production_endpoint_id', 'evaluations', ['production_endpoint_id'], schema=schema)
    op.create_index('ix_evaluations_evaluation_type', 'evaluations', ['evaluation_type'], schema=schema)
    op.create_index('ix_evaluations_status', 'evaluations', ['status'], schema=schema)
    op.create_index('ix_evaluations_created_at', 'evaluations', ['created_at'], schema=schema)


def downgrade() -> None:
    schema = 'model_registry'

    # Drop in reverse dependency order
    op.drop_table('evaluations', schema=schema)
    op.drop_table('benchmarks', schema=schema)
    op.drop_table('docker_build_artifacts', schema=schema)
    op.drop_table('docker_builds', schema=schema)
    op.drop_table('audit_logs', schema=schema)
    op.drop_table('artifacts', schema=schema)
    op.drop_table('deployments', schema=schema)
    op.drop_table('api_keys', schema=schema)
    op.drop_table('versions', schema=schema)
    op.drop_table('models', schema=schema)
