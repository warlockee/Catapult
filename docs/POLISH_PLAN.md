# Catapult OSS Polish Plan

Fix every issue a skeptical GitHub visitor would notice. Each section is an independent task for a subagent.

---

## Agent 1: MCP Rename (`model-registry-mcp` → `catapult-mcp`)

**Goal**: Complete rebrand of the MCP server package.

**Files to change**:
- `mcp/pyproject.toml` — package name `model-registry-mcp` → `catapult-mcp`, entrypoint `model-registry-mcp` → `catapult-mcp`
- `mcp/README.md` — all references
- `mcp/test_integration.py` — all references
- `mcp/src/model_registry_mcp/` → `mcp/src/catapult_mcp/` (rename directory)
- Every `.py` file inside that references `model_registry_mcp` in imports — update to `catapult_mcp`
- `mcp/src/catapult_mcp/server.py` — FastMCP name from `"Model Registry"` to `"Catapult"`
- `README.md` (root) — MCP config section: command `model-registry-mcp` → `catapult-mcp`

**Verification**: `grep -r "model.registry.mcp\|model_registry_mcp\|model-registry-mcp" mcp/` returns nothing.

---

## Agent 2: Docker Compose & Worker Fixes

**Goal**: Fix docker-compose issues that signal sloppiness.

**Changes**:
1. Remove `version: '3.8'` line from `docker-compose.yml` (obsolete, triggers warning)
2. Change worker `APP_NAME` from `"Model Registry Worker"` to `"Catapult Worker"` (line ~156)
3. Add worker healthcheck:
   ```yaml
   healthcheck:
     test: ["CMD-SHELL", "celery -A app.core.celery_app inspect ping --timeout 5 || exit 1"]
     interval: 30s
     timeout: 10s
     retries: 3
     start_period: 30s
   ```
4. Check redis also has a healthcheck (add if missing):
   ```yaml
   healthcheck:
     test: ["CMD", "redis-cli", "ping"]
     interval: 10s
     timeout: 5s
     retries: 5
   ```

**Verification**: `docker-compose config` succeeds without warnings. All services have healthchecks.

---

## Agent 3: Git Hygiene (`.agent/`, `.gitignore`)

**Goal**: Remove committed junk, harden `.gitignore`.

**Changes**:
1. Add to `.gitignore`:
   ```
   # AI agent state
   .agent/
   .cursor/

   # Test cache
   .pytest_cache/
   ```
2. Remove `.agent/` from git tracking: `git rm -r --cached .agent/`
3. Remove `.pytest_cache/` if tracked: `git rm -r --cached .pytest_cache/` (if exists)
4. Verify no `.env` file is tracked (only `.env.example` should be)

**Verification**: `git status` shows clean after re-add. `git ls-files .agent` returns nothing.

---

## Agent 4: CI/CD — GitHub Actions

**Goal**: Add a basic CI workflow so the repo has a green badge.

**Create `.github/workflows/ci.yml`**:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  backend-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff
      - run: ruff check backend/

  frontend-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "22"
      - run: cd frontend && npm ci && npm run build

  sdk-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: cd sdk/python && pip install -e . && python -c "from catapult import Registry; print('SDK OK')"

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker compose build
```

**Also add badges to README.md** (top, after the title):
```markdown
[![CI](https://github.com/warlockee/Catapult/actions/workflows/ci.yml/badge.svg)](https://github.com/warlockee/Catapult/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
```

**Verification**: Workflow file valid YAML. Badge markdown renders.

---

## Agent 5: Security & Community Files

**Goal**: Add governance files that signal a serious project.

**Create**:

1. **`SECURITY.md`**:
   ```markdown
   # Security Policy

   ## Reporting a Vulnerability

   If you discover a security vulnerability, please report it responsibly:

   1. **Do NOT open a public GitHub issue**
   2. Email: security@catapult-mlops.dev (or open a private security advisory on GitHub)
   3. Include: description, steps to reproduce, potential impact

   We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

   ## Supported Versions

   | Version | Supported |
   |---------|-----------|
   | 1.x     | ✅        |

   ## Security Best Practices

   - Change default `API_KEY_SALT` and `POSTGRES_PASSWORD` in `.env`
   - Use HTTPS in production (configure SSL in `infrastructure/nginx/ssl/`)
   - Restrict Docker socket access to trusted users
   - Review API key permissions (admin vs viewer roles)
   ```

2. **`.github/ISSUE_TEMPLATE/bug_report.md`**:
   ```markdown
   ---
   name: Bug Report
   about: Report a bug
   labels: bug
   ---

   **Describe the bug**
   A clear description of what the bug is.

   **To Reproduce**
   Steps to reproduce the behavior.

   **Expected behavior**
   What you expected to happen.

   **Environment**
   - OS:
   - Docker version:
   - Catapult version:

   **Logs**
   ```
   Paste relevant logs here
   ```
   ```

3. **`.github/ISSUE_TEMPLATE/feature_request.md`**:
   ```markdown
   ---
   name: Feature Request
   about: Suggest a feature
   labels: enhancement
   ---

   **Problem**
   What problem does this solve?

   **Proposed Solution**
   How should it work?

   **Alternatives Considered**
   Other approaches you've thought about.
   ```

**Verification**: Files exist and are valid markdown.

---

## Agent 6: Test Cleanup

**Goal**: Remove hardcoded internal IPs and make tests runnable by outsiders.

**Files**:
- `tests/verify_deployment.py` line 15 — change `http://172.202.29.125:26000/v1` to `http://localhost:8080/api/v1` (or make it env-var configurable)
- Scan ALL test files for hardcoded IPs: `grep -rn "172\.\|10\.0\.\|192\.168\.\|boson\|higgs" tests/`
- Replace any hardcoded API keys with placeholder or env var
- Remove any test that references internal-only infrastructure and can't work for OSS users
- Ensure `tests/e2e_test.py` works against a fresh `./deploy.sh` deployment

**Verification**: `grep -rn "172\.\|10\.0\.\|192\.168\.\|boson\|higgs" tests/` returns nothing.

---

## Agent 7: README Final Pass

**Goal**: Update README to reflect all changes from agents 1-6.

**Changes**:
1. Add CI badge and license badge after `# Catapult` title
2. Fix MCP install section — command is now `catapult-mcp` not `model-registry-mcp`
3. Fix MCP config JSON — entrypoint is `catapult-mcp`
4. Remove "when published" from `pip install catapult-sdk` (just show source install)
5. Add link to `/docs` (Swagger UI) in the Install section
6. Mention `SECURITY.md` somewhere
7. Keep it concise — don't re-bloat

**Verification**: No references to `model-registry-mcp` in README. Badges present.

---

## Execution Order

Agents 1-6 are **independent** — run in parallel.
Agent 7 depends on all others (runs last).

```
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │ Agent 4 │ │ Agent 5 │ │ Agent 6 │
│ MCP     │ │ Docker  │ │ Git     │ │ CI/CD   │ │Security │ │ Tests   │
│ Rename  │ │ Compose │ │ Hygiene │ │ Actions │ │ Files   │ │ Cleanup │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │           │           │
     └───────────┴───────────┴─────┬─────┴───────────┴───────────┘
                                   │
                             ┌─────┴─────┐
                             │  Agent 7  │
                             │  README   │
                             │  Final    │
                             └───────────┘
```

## Post-Swarm

After all agents complete:
1. `grep -ri "model.registry.mcp\|model_registry_mcp\|172\.\|boson" .` — final audit
2. `docker compose build` — verify everything still builds
3. `./deploy.sh` — verify deployment works
4. Commit and push
