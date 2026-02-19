# Catapult

- `backend/`: FastAPI application
- `frontend/`: React frontend application
- `infrastructure/`: Nginx and other config
- `sdk/`: Python client SDK

A lightweight, Python-first MLOps platform tailored for PyTorch/ML teams. Provides version management, metadata tracking, and deployment coordination for containerized ML applications.

## Features

- 🚀 **Simple Deployment**: Single `docker-compose up` command
- 🐍 **Python-First**: FastAPI backend + Python SDK
- 📦 **Version Management**: Track Docker images, model versions, and deployments
- 🔍 **Metadata Tracking**: Store training metrics, model info, and custom metadata
- 🔐 **API Key Authentication**: Simple and secure
- 🌐 **Web UI**: Clean React interface for browsing and management
- 📊 **Audit Logging**: Complete history of all operations
- 💾 **Ceph Integration**: Support for shared storage filesystems

## Quick Start

### Prerequisites

- Docker and Docker Compose
- (Optional) Ceph filesystem mount

### 1. Clone and Configure

```bash
git clone https://github.com/warlockee/Catapult.git
cd Catapult

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

**Required Configuration:**
- `POSTGRES_PASSWORD`: Set a secure database password
- `API_KEY_SALT`: Set a random 32+ character salt
- `CEPH_MOUNT_PATH`: Path to Ceph mount (or use `/tmp/registry-storage` for testing)

### 2. Deploy

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

The script will:
- Build Docker images
- Start all services (PostgreSQL, Backend, Frontend, Nginx)
- Run database migrations
- Create an initial API key

### 3. Access the Application

- **Web UI**: http://localhost
- **API Documentation**: http://localhost/docs
- **API Endpoint**: http://localhost/api

### 4. Configure API Key

When you first access the web UI, you'll need to enter your API key. Use the key generated during deployment (shown in the deploy script output).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client Layer                         │
├──────────────────────────────┬──────────────────────────────┤
│      Web UI (React)          │       Python SDK             │
└──────────────────────────────┴──────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Nginx Proxy  │
                    └───────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        ┌──────────────┐        ┌──────────────┐
        │ React SPA    │        │  FastAPI     │
        │ (Static)     │        │  Backend     │
        └──────────────┘        └──────────────┘
                                        │
                                        ▼
                                ┌──────────────┐
                                │ PostgreSQL   │
                                └──────────────┘
```

**Components:**
- **Nginx**: Reverse proxy and static file server
- **React Frontend**: Modern web UI built with React + TypeScript
- **FastAPI Backend**: Python 3.11+ REST API
- **PostgreSQL**: Primary database
- **Python SDK**: Client library for programmatic access

## Python SDK Usage

### Installation

```bash
# From the repository
cd sdk/python
pip install -e .

# Or from PyPI (when published)
pip install catapult-sdk
```

### Basic Example

```python
from catapult import Registry

# Initialize client
registry = Registry(
    base_url="http://localhost/api",
    api_key="your-api-key"
)

# Create a model
model = registry.create_model(
    name="myorg/pytorch-model",
    storage_path="docker.io/myorg/pytorch-model",
    description="My PyTorch model"
)

# Create a version
version = registry.create_version(
    model_name="myorg/pytorch-model",
    version="1.0.0",
    tag="v1.0.0",
    digest="sha256:abc123...",
    metadata={
        "pytorch_version": "2.1.0",
        "accuracy": 0.95,
        "git_commit": "abc123"
    }
)

# Promote to official release
registry.promote_version(version.id, is_release=True)

# Record a deployment
deployment = registry.deploy(
    release_id=version.id,
    environment="production",
    metadata={"replicas": 3}
)

# Get latest version
latest = registry.get_latest_version(
    model_name="myorg/pytorch-model"
)

print(f"Latest version: {latest.version}")
```

### Integration with Training Scripts

```python
import torch
from catapult import Registry

# Initialize from environment variables
# Set: REGISTRY_URL and REGISTRY_API_KEY
registry = Registry.from_env()

# Your training code
model = train_model()
accuracy = evaluate_model(model, test_loader)

# Register the version with training metadata
version = registry.create_version(
    model_name="myorg/sentiment-model",
    version="2.1.0",
    tag="v2.1.0",
    digest="sha256:...",  # From docker build
    metadata={
        "model_type": "BERT-base",
        "accuracy": accuracy,
        "training_samples": len(train_dataset),
        "pytorch_version": torch.__version__,
        "git_commit": get_git_commit(),
    }
)

print(f"Registered version: {version.id}")
```

## API Endpoints

### Health & Info
- `GET /api/health` - Health check
- `GET /api/v1/info` - API version and system info

### Models
- `GET /api/v1/models` - List models
- `POST /api/v1/models` - Create model
- `GET /api/v1/models/{id}` - Get model
- `PUT /api/v1/models/{id}` - Update model
- `DELETE /api/v1/models/{id}` - Delete model
- `GET /api/v1/models/{id}/versions` - List versions for model

### Versions
- `POST /api/v1/versions` - Create version
- `GET /api/v1/versions` - List versions (use `is_release=true` for official releases)
- `GET /api/v1/versions/latest` - Get latest version
- `GET /api/v1/versions/{id}` - Get version
- `PUT /api/v1/versions/{id}` - Update version (including promote/demote via `is_release`)
- `DELETE /api/v1/versions/{id}` - Delete version
- `GET /api/v1/versions/{id}/deployments` - List deployments for version

### Deployments
- `POST /api/v1/deployments` - Record deployment
- `GET /api/v1/deployments` - List deployments
- `GET /api/v1/deployments/{id}` - Get deployment

### API Keys
- `POST /api/v1/api-keys` - Create API key
- `GET /api/v1/api-keys` - List API keys
- `DELETE /api/v1/api-keys/{id}` - Revoke API key

### Audit Logs
- `GET /api/v1/audit-logs` - List audit logs

## Database Schema

The system uses 5 core tables:

- **models**: ML models/images
- **versions**: Model versions with metadata (can be promoted to official releases via `is_release` flag)
- **deployments**: Deployment history
- **api_keys**: Authentication keys
- **audit_logs**: Operation history

See [PROJECT_DESIGN.md](PROJECT_DESIGN.md) for detailed schema documentation.

## Management Commands

### Create API Key
```bash
docker-compose exec backend python scripts/create_api_key.py --name "my-key"
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Database Migrations
```bash
# Run migrations
docker-compose exec backend alembic upgrade head

# Create new migration
docker-compose exec backend alembic revision --autogenerate -m "description"
```

### Backup Database
```bash
docker-compose exec postgres pg_dump -U registry registry > backup.sql
```

### Restore Database
```bash
docker-compose exec -T postgres psql -U registry registry < backup.sql
```

## Testing

### End-to-End Tests

```bash
# 1. Start the application
docker-compose up -d

# 2. Create a test API key
TEST_KEY=$(docker-compose exec backend python scripts/create_api_key.py --name test-key | grep "Key:" | awk '{print $2}')

# 3. Run tests
export TEST_API_KEY=$TEST_KEY
python tests/e2e_test.py
```

The test suite validates:
- Health checks and connectivity
- Model CRUD operations
- Version creation and querying
- Deployment tracking
- API key management
- Complete workflows (model → version → deployment)
- Error handling (duplicates, not found, etc.)

### Manual Testing with curl

```bash
# Health check
curl http://localhost/api/health

# List images (requires API key)
curl -H "X-API-Key: your-key" http://localhost/api/v1/images

# Create image
curl -X POST http://localhost/api/v1/images \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-image",
    "repository": "docker.io/org/test-image",
    "description": "Test image"
  }'
```

## Troubleshooting

### Backend can't connect to database
```bash
# Check if postgres is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Verify connection
docker-compose exec postgres psql -U registry -d registry -c "SELECT 1;"
```

### Frontend can't reach API
```bash
# Check nginx logs
docker-compose logs nginx

# Verify backend is healthy
curl http://localhost/api/health

# Check CORS settings in .env
```

### Storage path not accessible
```bash
# Verify mount on host
ls -la $CEPH_MOUNT_PATH

# Check permissions
stat $CEPH_MOUNT_PATH

# For testing, use local directory
CEPH_MOUNT_PATH=/tmp/registry-storage
```

## Development

### Local Development Setup

```bash
# 1. Start database only
docker-compose up -d postgres

# 2. Backend
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload

# 3. Frontend (in another terminal)
cd frontend
npm install
npm run dev

# 4. SDK development
cd sdk/python
pip install -e .
```

### Running Individual Services

```bash
# Backend only
docker-compose up -d postgres
docker-compose up backend

# Frontend only
cd frontend
npm run dev

# All except nginx
docker-compose up -d postgres backend
```

## Production Deployment

For production deployment:

1. **Use strong passwords** in `.env`
2. **Enable HTTPS** by configuring SSL certificates in `infrastructure/nginx/ssl/`
3. **Configure backup strategy** for PostgreSQL and Ceph storage
4. **Set resource limits** in docker-compose.yml
5. **Configure monitoring** (optional: add Prometheus/Grafana)
6. **Use external Ceph mount** instead of local directory
7. **Review security settings** in PROJECT_DESIGN.md

## License

MIT License - See LICENSE file for details.

## Support

- Documentation: See [PROJECT_DESIGN.md](PROJECT_DESIGN.md)
- Issues: [GitHub Issues](https://github.com/warlockee/Catapult/issues)
- API Docs: http://localhost/docs (when running)

## Project Structure

```
Catapult/
├── backend/                # FastAPI backend
│   ├── app/
│   │   ├── api/           # API endpoints
│   │   ├── core/          # Config, database, security
│   │   ├── models/        # SQLAlchemy models
│   │   ├── schemas/       # Pydantic schemas
│   │   └── services/      # Business logic
│   ├── alembic/           # Database migrations
│   └── scripts/           # Utility scripts
├── frontend/                 # React frontend
│   └── src/
│       ├── components/    # React components
│       └── lib/           # API client, utilities
├── sdk/python/            # Python SDK
│   └── catapult/
├── infrastructure/        # Nginx configs
├── tests/                 # E2E tests
├── docker-compose.yml     # Main deployment config
└── deploy.sh             # Deployment script
```
