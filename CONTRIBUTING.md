# Contributing to Catapult

We welcome contributions! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/warlockee/Catapult
cd Catapult
cp .env.example .env    # Configure database, storage paths
./deploy.sh             # Starts all services with hot-reload
```

## Project Structure

The codebase follows clear patterns:

- **Repository layer** — Data access (`backend/app/repositories/`)
- **Service layer** — Business logic (`backend/app/services/`)
- **Pydantic schemas** — Validation (`backend/app/schemas/`)
- **React Query hooks** — Frontend data fetching (`frontend/src/lib/api.ts`)

New features typically touch one file in each layer.

## Making Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest backend/tests/`)
5. Commit your changes (`git commit -m 'Add my feature'`)
6. Push to the branch (`git push origin feature/my-feature`)
7. Open a Pull Request

## Adding Dockerfile Templates

Drop a `Dockerfile.<name>` into `kb/dockers/` and it will appear in the build UI automatically.

## Adding Evaluators

Implement the `BaseEvaluator` interface, register via the factory in `backend/app/services/eval/factory.py`, and your evaluation type is available across the platform.

## Code Style

- Python: Follow existing patterns. Use type hints. Async where possible.
- TypeScript: Follow existing patterns. Use TypeScript interfaces for API types.
- Both: Keep it simple. Don't over-engineer.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
