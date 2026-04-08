# Repository Guidelines

## Project Structure & Module Organization
The package is rooted at the repository top level. Core models live in `models.py`, the baseline agent in `inference.py`, the client helper in `client.py`, and topology analysis in `graph_analysis.py`. The FastAPI/OpenEnv server lives in `server/` with `app.py`, `environment.py`, `tasks.py`, `graders.py`, and logging helpers. Tests are in `tests/`, reference docs in `docs/` and `architecture/`, and submission utilities in `submission/`. Runtime artifacts go under `outputs/logs/` and `outputs/evals/`.

## Build, Test, and Development Commands
Use `uv` for local work.

- `env UV_CACHE_DIR=/tmp/uv-cache uv run --no-dev server --port 8000` starts the FastAPI server declared in `openenv.yaml`.
- `env UV_CACHE_DIR=/tmp/uv-cache uv run --no-dev grid2op-smoke --task-id single_fault --steps 1` runs a quick environment smoke test.
- `env UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev pytest tests/test_grid2op_env.py -q` runs the current pytest suite.
- `docker build -t grid2op-env:local -f server/Dockerfile .` builds the local container image.
- `bash submission/pre_validation.sh` runs submission checks before packaging.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints, `from __future__ import annotations`, and compact module-level imports. Use `snake_case` for functions, variables, and modules, `PascalCase` for Pydantic models, and `UPPER_CASE` for constants like `TASKS`. Keep OpenEnv payloads strongly typed with Pydantic models instead of raw dicts when practical. No formatter or linter config is committed, so match surrounding code and keep diffs minimal.

## Testing Guidelines
Tests use `pytest`. Add new coverage in `tests/test_grid2op_env.py` or split into `tests/test_<feature>.py` as the suite grows. Prefer deterministic assertions over probabilistic checks; this repository already tests grader determinism, task resets, proposal parsing, and graph-analysis output. Run the smoke command plus pytest before opening a PR.

## Commit & Pull Request Guidelines
Recent commits use short, direct subjects such as `docs updated` and `task 3 refining`. Keep commit titles imperative, lowercase is acceptable, and stay under roughly 60 characters. PRs should describe the affected task or subsystem, list validation commands run, and include baseline or API behavior changes when relevant. Add screenshots only for UI or HTTP response examples.

## Configuration & Runtime Notes
`openenv.yaml` points to `server.app:app` on port `8000`. Keep API credentials in environment variables or `.env`; do not hardcode secrets. If you change server routes or environment logic, restart the server before rerunning `inference.py`.
