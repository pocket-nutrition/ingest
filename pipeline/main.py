"""pocket-nutrition-ingest pipeline entry point.

Prerequisites:
  - Redis must be populated by pocket-nutrition-community (knowledge cache loaded)
  - ciqual_foods / mccance_foods / usda_foods must be loaded by pocket-nutrition-data
  - Elasticsearch must be reachable at ELASTICSEARCH_URL

Steps:
  1. transform    — fetch pairs from Redis knowledge cache, write training_pairs
  2. train        — XGBoost + Optuna, export model.joblib
  3. index_search — build product_search (labels + nutrients for canonical + OFF products)
"""

import logging

from pipeline.index_search import run as index_search
from pipeline.train import train_and_export
from pipeline.transform import transform_and_persist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run() -> None:
    log.info("Step 1/3: building training pairs from knowledge cache")
    transform_and_persist()

    log.info("Step 2/3: training XGBoost model")
    train_and_export()

    log.info("Step 3/3: indexing Knowledge catalogue into Elasticsearch")
    index_search()

    log.info("Pipeline complete")


if __name__ == "__main__":
    run()
