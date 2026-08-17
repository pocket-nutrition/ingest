"""Index the product search catalogue into Elasticsearch from the Knowledge Redis cache.

Run: python -m pipeline.index_search

Prerequisites:
  - Redis (db 2) must be populated by pocket-nutrition-community
  - Elasticsearch must be reachable at ELASTICSEARCH_URL
"""
from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Iterator
from typing import Any

from elasticsearch import Elasticsearch, helpers
from sqlalchemy import text

from pipeline.db import NUTRIENT_COLUMNS, SessionLocal
from pipeline.knowledge_store import (
    fetch_ingredients,
    fetch_mappings,
    fetch_recipes,
    fetch_relationships,
)
from pipeline.sync_variants import build_nutrient_lookup
from pipeline.utils import slugify

log = logging.getLogger(__name__)

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://pocket-nutrition-elasticsearch:9200")
ES_TIMEOUT = int(os.getenv("ELASTICSEARCH_TIMEOUT", "60"))
ALIAS_NAME = "product_search"
INDEX_BASE = "product_search"
_OFF_MAX_PRODUCTS: int | None = (
    int(float(os.getenv("OFF_MAX_PRODUCTS"))) if os.getenv("OFF_MAX_PRODUCTS") else None
)
_OFF_CHUNK_SIZE = 5_000

def _f(value: float | None) -> float | None:
    """Return None for NaN/Inf — Elasticsearch rejects non-standard JSON floats."""
    if value is None:
        return None
    try:
        return None if not math.isfinite(value) else value
    except (TypeError, ValueError):
        return None


def _first_brand(brands: str | None) -> str | None:
    """
    Reduce OFF's ``brands`` field to the one name a row can show.

    OFF stores it comma-separated and contributors are inconsistent about it — the same product may
    read ``"Coca Cola"``, ``"Coca-Cola,Coca Cola"``, or a distributor followed by a sub-brand. Only
    the first fits a suggestion row, and it is the one contributors put first. Whitespace-only and
    empty entries collapse to ``None`` so the client sees an absent brand rather than an empty
    string it would have to guard against.
    """
    if not brands:
        return None
    first = brands.split(",")[0].strip()
    return first or None


_QTY_MULTI_RE = re.compile(r'^\d+\s*[xX]\s*(\d+(?:[.,]\d+)?)\s*(g|kg|ml|cl|l)\b', re.IGNORECASE)
_QTY_SINGLE_RE = re.compile(r'^(\d+(?:[.,]\d+)?)\s*(g|kg|ml|cl|l)\b', re.IGNORECASE)


def _parse_package_quantity(quantity: str | None) -> tuple[float, str] | None:
    """Parse OFF 'quantity' field → (value_in_native_unit, 'g'|'ml') or None.

    Multi-pack "6 x 125 g" → individual unit (125, 'g').
    Handles g, kg, ml, cl, l. Returns None if unparseable.
    """
    if not quantity:
        return None
    m = _QTY_MULTI_RE.match(quantity.strip()) or _QTY_SINGLE_RE.match(quantity.strip())
    if not m:
        return None
    val = float(m.group(1).replace(',', '.'))
    unit = m.group(2).lower()
    if unit == 'kg':
        return (val * 1000.0, 'g')
    if unit == 'cl':
        return (val * 10.0, 'ml')
    if unit == 'l':
        return (val * 1000.0, 'ml')
    return (val, unit)  # 'g' or 'ml'


def _compute_default_portion(qty: float, max_qty: float = 400.0) -> float:
    """Halve qty until ≤ max_qty, return rounded result."""
    portion = qty
    while portion > max_qty:
        portion /= 2.0
    return round(portion)


_MAPPING: dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "product_id":        {"type": "keyword"},
            "product_type":      {"type": "keyword"},
            "image_url":         {"type": "keyword", "index": False},
            # Display-only, like image_url: the client shows it to tell two identically-named
            # products apart. Never searched, so it stays out of the inverted index.
            "brand":             {"type": "keyword", "index": False},
            "food_group":        {"type": "keyword"},
            "boost":             {"type": "float"},
            "has_training_data": {"type": "boolean"},
            "typical_state":          {"type": "keyword"},
            "portion_measured_state": {"type": "keyword"},
            "can_eat_raw":       {"type": "boolean"},
            "can_eat_cooked":    {"type": "boolean"},
            "yield_factors":     {"type": "object", "dynamic": True},
            "density_g_ml":      {"type": "float"},
            "portion_g":         {"type": "float"},
            "states": {
                "type": "object",
                "properties": {
                    "raw":       {"type": "object", "properties": {"portion_g": {"type": "float"}, "density_g_ml": {"type": "float"}}},
                    "cooked":    {"type": "object", "properties": {"portion_g": {"type": "float"}, "density_g_ml": {"type": "float"}}},
                    "processed": {"type": "object", "properties": {"portion_g": {"type": "float"}, "density_g_ml": {"type": "float"}}},
                },
            },
            "serving_unit_types": {"type": "keyword"},
            "display_unit":       {"type": "keyword"},
            "serving_mode":       {"type": "keyword"},
            "unit_label_fr":      {"type": "keyword"},
            "unit_label_en":      {"type": "keyword"},
            "package_size_ml":    {"type": "float"},
            "package_size_g":     {"type": "float"},
            "serving_size_g":     {"type": "float"},
            # Recipe components — stored in _source only, never queried.
            "components":         {"type": "object", "enabled": False},
            **{col: {"type": "float"} for col in NUTRIENT_COLUMNS},
            "labels": {
                "type": "nested",
                "properties": {
                    "lang":       {"type": "keyword"},
                    "text":       {"type": "text", "analyzer": "french"},
                    "normalized": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                },
            },
        }
    },
}


def _select_curated(
    ingredients: dict[str, dict[str, Any]], mapped: set[str]
) -> dict[str, dict[str, Any]]:
    """Keep only ingredients linked to at least one source mapping (curated).

    An ingredient with no source mapping (no Ciqual/OFF/USDA/McCance link) has no
    nutrient profile and is useless to callers, so it must not be indexed.
    """
    return {slug: ing for slug, ing in ingredients.items() if slug in mapped}


def _build_labels(entity: dict[str, Any], fallback_id: str) -> list[dict[str, str]]:
    """Build the nested multilingual labels array from name_* and aliases_* fields."""
    labels = [
        {
            "lang": key.removeprefix("name_"),
            "text": value,
            "normalized": slugify(value).replace("_", " "),
        }
        for key, value in entity.items()
        if key.startswith("name_") and value
    ]
    for key in ("aliases_fr", "aliases_en"):
        lang = key.removeprefix("aliases_")
        for alias in entity.get(key) or []:
            labels.append({
                "lang": lang,
                "text": alias,
                "normalized": slugify(alias).replace("_", " "),
            })
    if not labels:
        fallback = fallback_id.replace("_", " ")
        labels = [{"lang": "fr", "text": fallback, "normalized": fallback}]
    return labels


def _build_document(
    slug: str,
    ingredient: dict[str, Any],
    trained_set: set[str],
    yield_factors: dict[str, float] | None = None,
) -> dict[str, Any]:
    labels = _build_labels(ingredient, slug)
    states: dict[str, dict] = {
        k: v for k, v in (ingredient.get("states") or {}).items()
        if isinstance(v, dict)
    }
    _portion_state = next(
        (k for k, v in states.items() if v.get("portion_g") is not None),
        next(iter(states), "raw"),
    )
    _state_meta = states.get(_portion_state) or {}
    _portion_g = _state_meta.get("portion_g")
    _density_g_ml = next(
        (v.get("density_g_ml") for v in states.values() if v.get("density_g_ml") is not None),
        None,
    )

    return {
        "_id":               slug,
        "product_id":        slug,
        "product_type":      "ingredient",
        "food_group":        ingredient.get("food_group") or "other",
        "boost":             1.0,
        "has_training_data": slug in trained_set,
        "typical_state":          ingredient.get("typical_state"),
        "portion_measured_state": _portion_state,
        "can_eat_raw":       ingredient.get("can_eat_raw"),
        "can_eat_cooked":    ingredient.get("can_eat_cooked"),
        "density_g_ml":      _density_g_ml,
        "portion_g":         _portion_g,
        "serving_unit_types": ingredient.get("serving_unit_types") or [],
        "display_unit":       ingredient.get("display_unit"),
        "serving_mode":       ingredient.get("serving_mode"),
        "unit_label_fr":      ingredient.get("unit_label_fr"),
        "unit_label_en":      ingredient.get("unit_label_en"),
        "yield_factors":     yield_factors or {},
        "states":            {k: {kk: vv for kk, vv in v.items() if vv is not None} for k, v in states.items() if isinstance(v, dict)},
        "labels":            labels,
    }


def _build_recipe_document(
    recipe_id: str,
    recipe: dict[str, Any],
    curated_slugs: set[str],
) -> dict[str, Any] | None:
    """Build a recipe search document, or None if any component is not indexed.

    A recipe referencing a non-curated ingredient would hand the API a dangling
    component id, so the whole recipe is skipped (with a warning) instead.
    """
    components = recipe.get("components") or []
    missing = [c["ingredient_id"] for c in components if c.get("ingredient_id") not in curated_slugs]
    if missing or not components:
        log.warning(
            "Skipping recipe '%s' — components not in curated catalogue: %s",
            recipe_id, missing or "(no components)",
        )
        return None

    return {
        "_id":          f"recipe:{recipe_id}",
        # Prefixed like OFF products ("off:{code}") so a recipe never collides
        # with a same-named ingredient (e.g. couscous) in dedup or profile lookup.
        "product_id":   f"recipe:{recipe_id}",
        "product_type": "recipe",
        "food_group":   "other",
        # Recipes outrank same-label ingredients (e.g. 'raclette' the recipe vs
        # raclette_cheese whose name_fr is also 'raclette').
        "boost":        1.5,
        "has_training_data": False,
        "labels":       _build_labels(recipe, recipe_id),
        "components": [
            {
                "ingredient_id":  c["ingredient_id"],
                "weight_g":       float(c["weight_g"]),
                "cooking_method": c["cooking_method"],
            }
            for c in components
        ],
    }


def _build_off_document(row: dict[str, Any]) -> dict[str, Any]:
    labels = []
    if row.get("product_name_fr"):
        text_fr = row["product_name_fr"]
        labels.append({"lang": "fr", "text": text_fr, "normalized": slugify(text_fr).replace("_", " ")})
    if row.get("product_name"):
        text_en = row["product_name"]
        labels.append({"lang": "en", "text": text_en, "normalized": slugify(text_en).replace("_", " ")})
    if not labels:
        code = str(row["code"])
        labels = [{"lang": "fr", "text": code, "normalized": code}]
    sodium_g = _f(row.get("sodium_100g"))
    _parsed = _parse_package_quantity(row.get("quantity"))
    if _parsed is not None:
        _entity_qty, _display_unit = _parsed
        _portion_g: float | None = _compute_default_portion(_entity_qty)
    else:
        _portion_g = None
        _display_unit = None
    # OFF's own normalized per-serving weight, already in grams — no unit parsing needed, unlike
    # `quantity`. This is what gives a pizza or tart a real "1 part" shortcut: package_size_g above
    # is the whole product, but a multi-slice product's usable portion is the slice, not the box.
    _serving_g = _f(row.get("serving_quantity"))
    if _serving_g is not None and _serving_g <= 0:
        _serving_g = None
    return {
        "_id":               f"off:{row['code']}",
        "product_id":        f"off:{row['code']}",
        "product_type":      "off_product",
        "food_group":        "other",
        "boost":             0.5,
        "has_training_data": False,
        "typical_state":          None,
        "portion_measured_state": None,
        "can_eat_raw":       None,
        "can_eat_cooked":    None,
        "density_g_ml":      None,
        "portion_g":         _portion_g,
        "serving_unit_types": [],
        "display_unit":       _display_unit,
        "package_size_ml":   _entity_qty if (_parsed is not None and _display_unit == 'ml') else None,
        "package_size_g":    _entity_qty if (_parsed is not None and _display_unit == 'g') else None,
        "serving_size_g":    _serving_g,
        "labels":            labels,
        "image_url":         row.get("image_front_url") or None,
        "brand":             _first_brand(row.get("brand")),
        "energy_kcal":       _f(row.get("energy_kcal_100g")),
        "protein_g":         _f(row.get("proteins_100g")),
        "fat_g":             _f(row.get("fat_100g")),
        "carbohydrates_g":   _f(row.get("carbohydrates_100g")),
        "sugars_g":          _f(row.get("sugars_100g")),
        "fiber_g":           _f(row.get("fiber_100g")),
        "sodium_mg":         round(sodium_g * 1000, 4) if sodium_g is not None else None,
    }


def _stream_off_products_from_db() -> Iterator[dict[str, Any]]:
    """Stream OFF product documents one chunk at a time to avoid OOM on large tables."""
    session = SessionLocal()
    last_code = ""
    count = 0
    try:
        while _OFF_MAX_PRODUCTS is None or count < _OFF_MAX_PRODUCTS:
            remaining = None if _OFF_MAX_PRODUCTS is None else _OFF_MAX_PRODUCTS - count
            chunk_size = _OFF_CHUNK_SIZE if remaining is None else min(_OFF_CHUNK_SIZE, remaining)
            rows = session.execute(
                text("""
                    SELECT code, product_name, product_name_fr, quantity, serving_quantity,
                           image_front_url, brand,
                           energy_kcal_100g, proteins_100g, fat_100g, carbohydrates_100g,
                           sugars_100g, fiber_100g, sodium_100g
                    FROM off_products
                    WHERE (product_name IS NOT NULL OR product_name_fr IS NOT NULL)
                      AND code > :last_code
                    ORDER BY code
                    LIMIT :lim
                """),
                {"lim": chunk_size, "last_code": last_code},
            ).mappings().all()
            if not rows:
                break
            for row in rows:
                if _OFF_MAX_PRODUCTS is not None and count >= _OFF_MAX_PRODUCTS:
                    return
                yield _build_off_document(dict(row))
                count += 1
                if count % 100_000 == 0:
                    log.info("Streamed %d OFF products so far...", count)
            last_code = rows[-1]["code"]
            if (_OFF_MAX_PRODUCTS is not None and count >= _OFF_MAX_PRODUCTS) or len(rows) < chunk_size:
                break
    finally:
        session.close()
        log.info("Finished streaming %d OFF products", count)


def run() -> None:
    import time
    es = Elasticsearch(ES_URL, request_timeout=ES_TIMEOUT)

    # First run: product_search may be a physical index (not yet an alias). Delete it
    # so the alias can be created in its place after the new index is populated.
    if es.indices.exists(index=ALIAS_NAME) and not es.indices.exists_alias(name=ALIAS_NAME):
        es.indices.delete(index=ALIAS_NAME)
        log.info("Deleted legacy physical index '%s' (replacing with alias)", ALIAS_NAME)

    # Create a versioned index — the old one remains live while we populate the new one.
    new_index = f"{INDEX_BASE}_{int(time.time())}"
    try:
        # wait_for_active_shards=0: return immediately after the cluster state update;
        # don't block waiting for the shard to be written to disk (NFS can be slow).
        es.indices.create(index=new_index, body=_MAPPING, wait_for_active_shards=0)
    except Exception as exc:
        if "resource_already_exists_exception" in str(exc):
            log.warning("Index '%s' already exists (race) — deleting and retrying", new_index)
            es.indices.delete(index=new_index)
            es.indices.create(index=new_index, body=_MAPPING, wait_for_active_shards=0)
        else:
            raise
    log.info("Created index '%s', waiting for primary shard to become active", new_index)
    # Poll until the primary shard is active (green). Separate from create so NFS
    # latency on shard initialisation doesn't block the HTTP request.
    es.cluster.health(
        index=new_index,
        wait_for_status="green",
        timeout=f"{ES_TIMEOUT}s",
        request_timeout=ES_TIMEOUT + 5,
    )
    log.info("Index '%s' is green", new_index)

    ingredients = fetch_ingredients()
    if not ingredients:
        raise RuntimeError(
            "No ingredients found in Redis — ensure pocket-nutrition-community is loaded "
            "(Reload button in UI or POST /admin/reload)."
        )

    # Fetch mappings + relationships once and reuse for both the curation filter and trained_set.
    mapped = {m["canonical_ingredient_id"] for m in fetch_mappings()}
    relationships = fetch_relationships()
    # trained_set: ingredients with both a mapping and a cooking_relationship (mirrors
    # the join condition in transform.py Step 1). Used only for the has_training_data flag.
    trained_set = mapped & {r["raw_id"] for r in relationships}
    log.info("%d ingredients are model-ready (have mapping + cooking_relationship)", len(trained_set))

    # Only index curated ingredients — those linked to at least one source mapping.
    curated = _select_curated(ingredients, mapped)
    skipped = len(ingredients) - len(curated)
    if skipped:
        log.info("Skipped %d non-curated ingredients (no source mapping)", skipped)
    log.info("Indexing %d curated ingredients", len(curated))

    yield_lookup: dict[str, dict[str, float]] = {}
    for rel in relationships:
        raw_id = rel.get("raw_id")
        method = rel.get("cooking_method")
        yf = rel.get("yield_factor")
        if raw_id and method and yf is not None:
            yield_lookup.setdefault(raw_id, {})[method] = float(yf)

    docs = [_build_document(slug, ing, trained_set, yield_lookup.get(slug)) for slug, ing in curated.items()]
    log.info("Built %d ingredient documents", len(docs))

    nutrient_lookup = build_nutrient_lookup()
    nutrient_count = 0
    for doc in docs:
        nutrients = nutrient_lookup.get(doc["product_id"])
        if nutrients:
            doc.update({k: _f(v) for k, v in nutrients.items()})
            nutrient_count += 1
    log.info("Merged nutrient profiles into %d / %d ingredient documents", nutrient_count, len(docs))

    curated_slugs = set(curated)
    recipe_docs = [
        doc
        for recipe_id, recipe in fetch_recipes().items()
        if (doc := _build_recipe_document(recipe_id, recipe, curated_slugs)) is not None
    ]
    log.info("Built %d recipe documents", len(recipe_docs))

    def _all_documents() -> Iterator[dict[str, Any]]:
        for doc in docs:
            yield {"_index": new_index, **doc}
        for doc in recipe_docs:
            yield {"_index": new_index, **doc}
        for doc in _stream_off_products_from_db():
            yield {"_index": new_index, **doc}

    log.info("Indexing ingredient documents + streaming OFF products into '%s'", new_index)
    successes = 0
    errors: list[Any] = []
    for ok, info in helpers.parallel_bulk(
        es,
        _all_documents(),
        raise_on_error=False,
        raise_on_exception=False,
    ):
        if ok:
            successes += 1
        else:
            errors.append(info)

    if errors:
        log.error("%d documents failed to index", len(errors))
        for sample in errors[:10]:
            log.error("  sample error: %s", sample)
    log.info("Indexed %d documents into '%s'", successes, new_index)
    es.indices.refresh(index=new_index)

    # Atomic alias swap: find current target, remove + add in one request.
    old_index = None
    if es.indices.exists_alias(name=ALIAS_NAME):
        old_index = next(iter(es.indices.get_alias(name=ALIAS_NAME)))

    actions: list[dict[str, Any]] = []
    if old_index:
        actions.append({"remove": {"index": old_index, "alias": ALIAS_NAME}})
    actions.append({"add": {"index": new_index, "alias": ALIAS_NAME}})
    es.indices.update_aliases(body={"actions": actions})
    log.info("Alias '%s' → '%s'", ALIAS_NAME, new_index)

    if old_index:
        es.indices.delete(index=old_index)
        log.info("Deleted old index '%s'", old_index)

    log.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
