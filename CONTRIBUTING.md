# Contributing to Pocket Nutrition Ingest

Thanks for your interest in improving the training pipeline. This document covers local
development setup, how to run the pipeline steps individually, testing, and the PR process.

## Development Setup

```bash
git clone https://github.com/pocket-nutrition/ingest.git
cd ingest

python3.12 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt pytest
```

Copy `.env.example` to `.env` and adjust values for your local Redis / PostgreSQL /
Elasticsearch instances (see `CLAUDE.md` for what each variable controls and what upstream
service is expected to have populated the data):

```bash
cp .env.example .env
```

This repository is a pure **consumer**: it reads a knowledge cache materialised into Redis by
`pocket-nutrition/community` and raw nutrient tables loaded into PostgreSQL by
`pocket-nutrition/data`. You need at least one of those populated to exercise the pipeline
end-to-end locally; unit tests do not require either.

## Running Tests

```bash
pytest tests/ -v
```

Tests are unit-level only — no live database, Redis, or Elasticsearch connection is required.
External clients (`SessionLocal`, the Redis client, Elasticsearch) are mocked with
`unittest.mock`. Keep new tests in that style: mock the client boundary, assert on the
transformation logic around it.

## Running the Pipeline Steps

The full pipeline runs all three steps in order:

```bash
python -m pipeline.main
```

Each step can also be run in isolation once its prerequisites are met:

```bash
# Step 1 — build training_pairs from the Redis knowledge cache + PostgreSQL nutrient tables
python -c "from pipeline.transform import transform_and_persist; transform_and_persist()"

# Step 2 — train the XGBoost model and export the artifact (needs training_pairs populated)
python -c "from pipeline.train import train_and_export; train_and_export()"

# Step 3 — rebuild the product_search Elasticsearch index (useful when the model itself
# does not need retraining)
python -m pipeline.index_search
```

`pipeline/main.py` is the only module with a `__main__` entry point; `transform.py` and
`train.py` are invoked as library functions (see the snippets above), while
`index_search.py` also exposes a `run()` function callable the same way.

## Code Style

- Python 3.12, type-annotated function signatures.
- Keep new pipeline logic in small, focused functions — favor unit-testable helpers
  (like `_derive_cooked_profile`, `_confidence_weighted_median`) over large monolithic
  functions.
- No live I/O in unit tests. Mock `SessionLocal`, the Redis client, and the Elasticsearch
  client at their import boundary.

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your change, adding or updating tests under `tests/` for any behavior change.
3. Run `pytest tests/ -v` and confirm everything passes.
4. Open a PR describing the change, why it's needed, and how it was tested. Reference any
   related issue.
5. Keep PRs focused — one logical change per PR is easier to review and merge.

## Reporting Issues

Use the GitHub issue templates (bug report / feature request) under
`.github/ISSUE_TEMPLATE/`. Include:

- What you ran (`python -m pipeline.main`, a specific step, `pytest tests/`, ...)
- What you expected vs. what happened
- Relevant log output (this pipeline logs via the standard `logging` module at `INFO` level)

## CI and releases

- Every pull request runs `.github/workflows/ci.yml`. The `ci-success` check must be green before a pull
  request can be merged, the branch must be up to date with `main`, and pull requests are squash-merged
  so history stays linear.
- If you contribute from a fork, the workflow waits for a maintainer to approve it. It runs with a
  read-only token and no secrets.
- Every commit merged to `main` is released automatically with a CalVer version, `vYYYY.M.N`: the year, the
  month, then a counter that restarts at 1 each month (for example `v2026.11.1`). You never pick a version
  number or edit a changelog; release notes are generated from the merged pull requests.
- A release publishes the container image `ghcr.io/pocket-nutrition/ingest`.
