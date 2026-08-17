# Pocket Nutrition Ingest

Python pipeline that trains the XGBoost cooking-transform model and indexes the Knowledge catalogue into Elasticsearch for product search.

## Pipeline steps

| Step | Module | Output |
|------|--------|--------|
| 1. transform | `pipeline/transform.py` | `training_pairs` in PostgreSQL |
| 2. train | `pipeline/train.py` | `model.joblib` on shared volume |
| 3. index_search | `pipeline/index_search.py` | `product_search` index in Elasticsearch |

### Step 3 — `product_search` Elasticsearch index

Reads ingredients from the Knowledge Redis cache (`knowledge:ingredient:*` keys, db 2) populated by `pocket-nutrition/community`. Each ingredient becomes one ES document with nested multilingual labels:

```json
{
  "product_id": "poulet_filet",
  "product_type": "ingredient",
  "food_group": "meat",
  "boost": 1.0,
  "labels": [
    { "lang": "fr", "text": "Poulet, filet", "normalized": "poulet filet" },
    { "lang": "en", "text": "Chicken breast", "normalized": "chicken breast" }
  ]
}
```

The index is deleted and recreated on each run. `pocket-nutrition/api` queries this index for ingredient search and resolution.

## Prerequisites

| Dependency | Env var | Default |
|------------|---------|---------|
| PostgreSQL | `DATABASE_URL` | `postgresql://pocket_nutrition:pocket_nutrition@postgresql:5432/pocket_nutrition` |
| Redis (Knowledge cache) | `REDIS_URL` | `redis://redis:6379/2` |
| Elasticsearch | `ELASTICSEARCH_URL` | `http://pocket-nutrition-elasticsearch:9200` |
| Model output path | `MODEL_OUTPUT_PATH` | `/models/model.joblib` |

Redis must be populated by `pocket-nutrition/community` before running. PostgreSQL must be populated by `pocket-nutrition/data` before running.

## Local run

```bash
# Full pipeline (all 3 steps)
python -m pipeline.main

# Search index only (step 3)
python -m pipeline.index_search
```

## Key modules

| Module | Purpose |
|--------|---------|
| `pipeline/main.py` | Entry point — runs all 3 steps |
| `pipeline/transform.py` | Builds training pairs from Knowledge Redis |
| `pipeline/train.py` | XGBoost + Optuna, exports model artifact |
| `pipeline/index_search.py` | Indexes Knowledge catalogue into Elasticsearch |
| `pipeline/knowledge_store.py` | Redis client — `fetch_ingredients()` |
| `pipeline/utils.py` | `slugify()` — must match the equivalent slugify helper in `pocket-nutrition/api` |
