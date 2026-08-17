"""Build ingredient nutrient profiles from Redis source_mappings + raw PG tables.

The resulting dict (ingredient_id → nutrients) is consumed by index_search.py,
which embeds nutrients directly into the product_search Elasticsearch index.
The intermediate ingredient_profiles ES index has been removed.
"""
from __future__ import annotations

import logging
import math

from sqlalchemy import text

from pipeline.db import NUTRIENT_COLUMNS, SessionLocal
from pipeline.knowledge_store import fetch_mappings

log = logging.getLogger(__name__)

_CIQUAL_REMAP: dict[str, str] = {"proteins_g": "protein_g"}
_MCCANCE_REMAP: dict[str, str] = {"carbohydrate_g": "carbohydrates_g", "fibre_g": "fiber_g"}
_USDA_COLS: frozenset[str] = frozenset(
    {"energy_kcal", "protein_g", "fat_g", "carbohydrates_g", "fiber_g", "sodium_mg"}
)
_USDA_EXTENDED_COLS: frozenset[str] = frozenset({
    "energy_kcal", "protein_g", "fat_g", "carbohydrates_g", "fiber_g",
    "sodium_mg", "calcium_mg", "iron_mg", "magnesium_mg", "phosphorus_mg",
    "potassium_mg", "zinc_mg", "vitamin_c_mg", "vitamin_d_ug", "vitamin_a_ug",
    "vitamin_b6_mg", "vitamin_b12_ug", "niacin_mg",
    "saturated_fat_g", "monounsaturated_fat_g", "polyunsaturated_fat_g", "cholesterol_mg",
})

# Lower value = preferred source
_SOURCE_PRIORITY: dict[str, int] = {
    "ciqual": 0,
    "mccance": 1,
    "usda_foundation": 2,
    "usda": 3,
    "usda_sr": 4,
}


def _row_to_nutrients(row: dict, remap: dict[str, str], available: frozenset[str] | None = None) -> dict:
    reverse = {v: k for k, v in remap.items()}
    nutrients: dict = {}
    for col in NUTRIENT_COLUMNS:
        src_col = reverse.get(col, col)
        if available is not None and src_col not in available:
            nutrients[col] = None
        else:
            nutrients[col] = row.get(src_col)
    return nutrients


def _fetch_ciqual(session, source_id: str) -> dict | None:
    row = session.execute(
        text("SELECT * FROM ciqual_foods WHERE alim_code = :code"),
        {"code": int(source_id)},
    ).mappings().first()
    return None if row is None else _row_to_nutrients(dict(row), _CIQUAL_REMAP)


def _fetch_mccance(session, source_id: str) -> dict | None:
    row = session.execute(
        text("SELECT * FROM mccance_foods WHERE food_code = :code"),
        {"code": source_id},
    ).mappings().first()
    return None if row is None else _row_to_nutrients(dict(row), _MCCANCE_REMAP)


def _fetch_usda(session, source_id: str) -> dict | None:
    row = session.execute(
        text("SELECT * FROM usda_foods WHERE fdc_id = :fdc_id"),
        {"fdc_id": int(source_id)},
    ).mappings().first()
    return None if row is None else _row_to_nutrients(dict(row), {}, available=_USDA_COLS)


def _fetch_usda_foundation(session, source_id: str) -> dict | None:
    row = session.execute(
        text("SELECT * FROM usda_foundation_foods WHERE fdc_id = :fdc_id"),
        {"fdc_id": int(source_id)},
    ).mappings().first()
    return None if row is None else _row_to_nutrients(dict(row), {}, available=_USDA_EXTENDED_COLS)


def _fetch_usda_sr(session, source_id: str) -> dict | None:
    row = session.execute(
        text("SELECT * FROM usda_sr_foods WHERE fdc_id = :fdc_id"),
        {"fdc_id": int(source_id)},
    ).mappings().first()
    return None if row is None else _row_to_nutrients(dict(row), {}, available=_USDA_EXTENDED_COLS)


_SOURCE_FETCHERS = {
    "ciqual": _fetch_ciqual,
    "mccance": _fetch_mccance,
    "usda_foundation": _fetch_usda_foundation,
    "usda": _fetch_usda,
    "usda_sr": _fetch_usda_sr,
}


def _sort_key(m: dict) -> tuple[int, int]:
    source = m.get("source", "")
    label = m.get("label")
    return (_SOURCE_PRIORITY.get(source, 99), 0 if label is None else 1)


def _confidence_weighted_median(candidates: list[tuple[dict, float]]) -> dict:
    """Merge multiple nutrient dicts into one using confidence-weighted median per column.

    Each candidate is (nutrients_dict, confidence_weight). For each nutrient column,
    collects non-None/non-NaN values, sorts them, then picks the weighted median
    (smallest value whose cumulative weight reaches >= 50% of total weight).
    Columns with no valid value across all candidates are omitted from the result.
    """
    all_cols: set[str] = set()
    for nutrients, _ in candidates:
        all_cols.update(nutrients.keys())

    result: dict = {}
    for col in all_cols:
        pairs: list[tuple[float, float]] = []
        for nutrients, weight in candidates:
            v = nutrients.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            pairs.append((float(v), weight))

        if not pairs:
            continue

        pairs.sort(key=lambda x: x[0])
        total_w = sum(w for _, w in pairs)
        cumulative = 0.0
        for val, w in pairs:
            cumulative += w
            if cumulative >= total_w / 2:
                result[col] = round(val, 4)
                break

    return result


def build_nutrient_lookup() -> dict[str, dict]:
    """Return ingredient_id → {nutrient_col: value} from Redis mappings + PG source tables.

    Only raw-state mappings (cooking_method == "raw") are used. When multiple raw mappings
    exist for the same ingredient, nutrient values are merged via confidence-weighted median.
    No Elasticsearch I/O — consumed by index_search.run() and embedded into product_search.
    """
    mappings = fetch_mappings()
    if not mappings:
        log.warning("No source mappings found in Redis — nutrient lookup will be empty")
        return {}

    log.info("Building nutrient lookup from %d source mapping(s)", len(mappings))

    by_ingredient: dict[str, list[dict]] = {}
    for m in mappings:
        canonical_id = m["canonical_ingredient_id"]
        by_ingredient.setdefault(canonical_id, []).append(m)

    for ing_mappings in by_ingredient.values():
        ing_mappings.sort(key=_sort_key)

    session = SessionLocal()
    result: dict[str, dict] = {}
    skipped = 0
    try:
        for ingredient_id, ing_mappings in by_ingredient.items():
            candidates: list[tuple[dict, float]] = []
            for m in ing_mappings:
                if m.get("cooking_method", "raw") != "raw":
                    continue
                source = m.get("source", "")
                source_id = str(m.get("source_id", ""))
                fetcher = _SOURCE_FETCHERS.get(source)
                if fetcher is None:
                    continue
                nutrients = fetcher(session, source_id)
                if nutrients is None:
                    continue
                confidence = float(m.get("confidence") or 1.0)
                candidates.append((nutrients, confidence))

            if candidates:
                result[ingredient_id] = _confidence_weighted_median(candidates)
            else:
                log.debug("No raw nutrient data found for %s — skipping", ingredient_id)
                skipped += 1
    finally:
        session.close()

    log.info("Built nutrient lookup for %d ingredients (%d skipped — no source data)", len(result), skipped)
    return result


def sync_variants() -> None:
    """Compatibility shim — nutrients are now embedded into product_search by index_search.run()."""
    log.info("sync_variants: retired — nutrients embedded into product_search; no ingredient_profiles index created")
