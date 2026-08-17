# Pocket Nutrition Ingest

Python training pipeline for [Pocket Nutrition](https://pocketnutrition.org). Reads the ingredient knowledge base out of Redis and raw nutrient source tables out of PostgreSQL, trains an XGBoost model that predicts post-cooking nutrient values, and publishes an Elasticsearch search index used to resolve ingredient names to canonical profiles.

This repository is a pure **consumer** in the wider Pocket Nutrition pipeline: it never writes back to Redis and never modifies the raw source tables it reads from (`ciqual_foods`, `mccance_foods`, `usda_foods`, `off_products`, ...). Its only outputs are a derived PostgreSQL table (`training_pairs`), a model artifact file, and an Elasticsearch index.

## Prerequisites

| Dependency | Purpose |
|------------|---------|
| Redis | Knowledge cache (ingredients, source mappings, cooking relationships), materialised by `pocket-nutrition/community` |
| PostgreSQL | Raw nutrient source tables, loaded by `pocket-nutrition/data` |
| Elasticsearch | Destination for the `product_search` index |

See `.env.example` for the full list of environment variables (`DATABASE_URL`, `REDIS_URL`, `ELASTICSEARCH_URL`, `MODEL_OUTPUT_PATH`).

## Pipeline steps

The entry point is `pipeline/main.py`, which runs three steps in order (`python -m pipeline.main`):

| Step | Module | What it does | Output |
|------|--------|---------------|--------|
| 1. transform | `pipeline/transform.py` | Builds raw→cooked nutrient training pairs from Redis cooking relationships + nutrient profiles resolved via `pipeline/sync_variants.py`'s `build_nutrient_lookup()`. Also imputes "theoretical" pairs (group-median yield factors) for ingredient/method combinations with a profile but no explicit yield. | `training_pairs` table in PostgreSQL (dropped and recreated each run) |
| 2. train | `pipeline/train.py` | Runs an Optuna hyperparameter search over an XGBoost `MultiOutputRegressor`, evaluated with grouped k-fold cross-validation (grouped by ingredient so no ingredient leaks between folds), then fits a final model on the full training set. | Model artifact (`joblib` pickle containing the model, scaler, encoders, food/raw nutrient indices, and yield factors) |
| 3. index_search | `pipeline/index_search.py` | Rebuilds the `product_search` Elasticsearch index: curated knowledge ingredients (only those with at least one source mapping), recipes, and a full stream of Open Food Facts products, all as one alias-swapped index. | `product_search` Elasticsearch index |

Note: `pipeline/sync_variants.py` also contains a `sync_variants()` function, but it is a retired no-op compatibility shim — nutrient profiles are now looked up on demand and embedded directly into `product_search` by the `index_search` step. The active pipeline is the three steps above.

## Running locally

```bash
pip install -r requirements.txt

export DATABASE_URL=postgresql://user:password@localhost:5432/pocket_nutrition
export REDIS_URL=redis://localhost:6379/2
export ELASTICSEARCH_URL=http://localhost:9200
export MODEL_OUTPUT_PATH=./models/model.joblib

# Full pipeline (all 3 steps)
python -m pipeline.main

# Search index only (step 3, useful when the model doesn't need retraining)
python -m pipeline.index_search
```

Or with Docker:

```bash
docker build -t pocket-nutrition-ingest .
docker run --env-file .env pocket-nutrition-ingest
```

## Testing

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

Tests are unit-level only (no live database, Redis, or Elasticsearch required) — external clients are mocked.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, running individual pipeline steps, and
the PR workflow.

## License

AGPL-3.0. See [LICENSE](./LICENSE).
