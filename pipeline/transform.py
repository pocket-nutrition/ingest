"""Build training pairs from Redis knowledge cache + PostgreSQL nutrient profiles."""
from __future__ import annotations

import json
import logging
from statistics import median as _median

from sqlalchemy.dialects.postgresql import insert

from pipeline.db import NUTRIENT_COLUMNS, TrainingPair, engine, init_training_tables
from pipeline.knowledge_store import fetch_ingredients, fetch_relationships
from pipeline.sync_variants import build_nutrient_lookup

log = logging.getLogger(__name__)

_COOKED_REMAP: dict[str, str] = {
    "meat": "roasted",
    "fish": "roasted",
    "egg": "roasted",
    "fat": "roasted",
    "vegetable": "boiled",
    "cereal": "boiled",
    "dairy": "boiled",
    "shellfish": "boiled",
    "other": "boiled",
}

_THEORETICAL_METHODS = ["boiled", "grilled", "roasted", "fried", "steamed"]
_MIN_YIELD_GROUP_COUNT = 3


def _derive_cooked_profile(
    raw_profile: dict,
    yield_f: float,
    nutrient_retention: dict[str, float] | None = None,
) -> dict:
    """Cooked profile of 100 g of RAW input — the basis ml scales its predictions by.

    100 g raw becomes `100 × yield_f` g cooked. Cooking creates no nutrient, so each nutrient
    is conserved per raw gram, minus what does not survive (USDA SR28 retention). The mass
    change is water: lost when yield < 1, absorbed when yield > 1.

    Do not divide by the yield here: ml already converts a cooked weight to its raw
    equivalent with the same yield, so a per-100 g-cooked target concentrates twice.
    """
    if yield_f <= 0:
        return dict(raw_profile)
    result = {}
    for col, val in raw_profile.items():
        if val is None:
            result[col] = None
        elif col == "water_g":
            result[col] = max(0.0, val - 100.0 * (1.0 - yield_f))
        else:
            retention = (nutrient_retention or {}).get(col, 1.0)
            result[col] = round(val * retention, 4)
    return result


def _load_profiles() -> dict[str, dict]:
    """Load ingredient nutrient profiles from Redis mappings + PostgreSQL source tables."""
    return build_nutrient_lookup()


def _compute_group_yields(
    relationships: list[dict],
    ingredient_meta: dict[str, dict],
    min_count: int = _MIN_YIELD_GROUP_COUNT,
) -> dict[tuple[str, str], float]:
    """Return {(food_group, cooking_method): median_yield} for groups with >= min_count known yields.

    Generic 'cooked' relationships are excluded — they are remapped to specific
    methods by _COOKED_REMAP and would introduce noise in the group statistics.
    """
    buckets: dict[tuple[str, str], list[float]] = {}
    for rel in relationships:
        yf = rel.get("yield_factor")
        rid = rel.get("raw_id")
        method = rel.get("cooking_method")
        if yf is None or not rid or not method or method == "cooked":
            continue
        fg = ingredient_meta.get(rid, {}).get("food_group")
        if not fg:
            continue
        buckets.setdefault((fg, method), []).append(float(yf))
    return {
        key: _median(vals)
        for key, vals in buckets.items()
        if len(vals) >= min_count
    }


def _generate_theoretical_pairs(
    known_fm: set[tuple[str, str]],
    profile_map: dict[str, dict],
    ingredient_meta: dict[str, dict],
    group_yields: dict[tuple[str, str], float],
) -> list[dict]:
    """Generate training pairs using group-median yields for ingredient+method combos lacking explicit data."""
    pairs = []
    for slug, profile_doc in profile_map.items():
        fg = ingredient_meta.get(slug, {}).get("food_group")
        if not fg:
            continue
        raw_profile = {col: profile_doc.get(col) for col in NUTRIENT_COLUMNS}
        for method in _THEORETICAL_METHODS:
            if (slug, method) in known_fm:
                continue
            yf = group_yields.get((fg, method))
            if yf is None:
                continue
            cooked_profile = _derive_cooked_profile(raw_profile, yf, nutrient_retention=None)
            pairs.append({
                "source": "theoretical",
                "food_name": slug,
                "cooking_method": method,
                "raw_profile": json.dumps(raw_profile),
                "cooked_profile": json.dumps(cooked_profile),
                "food_group": fg,
                "yield_f": yf,
                "is_theoretical": True,
            })
    return pairs


def transform_and_persist() -> None:
    relationships = fetch_relationships()
    if not relationships:
        raise RuntimeError(
            "No relationships found in Redis — ensure community is loaded "
            "(Reload button in UI or POST /admin/reload)."
        )
    log.info("Fetched %d relationships from knowledge cache", len(relationships))

    ingredients = fetch_ingredients()
    log.info("Fetched %d ingredient records from knowledge cache", len(ingredients))

    profile_map = _load_profiles()
    log.info("Loaded nutrient profiles for %d distinct ingredients", len(profile_map))

    init_training_tables()

    # Process generic 'cooked' relationships first so that explicit method
    # relationships (e.g. 'roasted') overwrite them when both exist for the
    # same ingredient.
    relationships_sorted = sorted(
        relationships, key=lambda r: (r.get("cooking_method") != "cooked")
    )

    pair_map: dict[tuple[str, str, str], dict] = {}
    skipped = 0
    for rel in relationships_sorted:
        raw_id: str | None = rel.get("raw_id")
        cooking_method: str | None = rel.get("cooking_method")
        yield_f: float | None = rel.get("yield_factor")

        if not raw_id or not cooking_method or yield_f is None:
            skipped += 1
            continue

        profile_doc = profile_map.get(raw_id)
        if not profile_doc:
            log.debug("No nutrient profile for slug=%s — skipping", raw_id)
            skipped += 1
            continue

        raw_profile = {col: profile_doc.get(col) for col in NUTRIENT_COLUMNS}
        nutrient_retention: dict[str, float] | None = rel.get("nutrient_retention") or None
        cooked_profile = _derive_cooked_profile(raw_profile, float(yield_f), nutrient_retention)
        food_group = ingredients.get(raw_id, {}).get("food_group")

        if cooking_method == "cooked":
            cooking_method = _COOKED_REMAP.get(food_group or "other", "boiled")

        key = ("knowledge", raw_id, cooking_method)
        pair_map[key] = {
            "source": "knowledge",
            "food_name": raw_id,
            "cooking_method": cooking_method,
            "raw_profile": json.dumps(raw_profile),
            "cooked_profile": json.dumps(cooked_profile),
            "food_group": food_group,
            "yield_f": float(yield_f),
            "is_theoretical": False,
        }

    known_pairs = list(pair_map.values())
    log.info(
        "Built %d known training pairs (%d skipped — missing profile or yield)",
        len(known_pairs), skipped,
    )

    # Theoretical yield imputation: cover ingredients with profiles but no explicit yield.
    known_fm: set[tuple[str, str]] = {(p["food_name"], p["cooking_method"]) for p in known_pairs}
    group_yields = _compute_group_yields(relationships, ingredients)
    if group_yields:
        log.info(
            "Computed theoretical yields for %d (food_group, cooking_method) combinations",
            len(group_yields),
        )
    else:
        log.warning(
            "No group yields could be computed — theoretical imputation disabled. "
            "Ensure relationships have yield_factor and ingredients have food_group set."
        )
    theoretical_pairs = _generate_theoretical_pairs(
        known_fm=known_fm,
        profile_map=profile_map,
        ingredient_meta=ingredients,
        group_yields=group_yields,
    )
    log.info("Generated %d theoretical pairs", len(theoretical_pairs))

    if not known_pairs and not theoretical_pairs:
        raise RuntimeError(
            "No training pairs produced — check that Redis knowledge cache is populated "
            "and PostgreSQL source tables contain nutrient data."
        )

    # Known pairs: upsert with overwrite (fresh run refreshes data).
    if known_pairs:
        stmt = insert(TrainingPair).values(known_pairs)
        stmt = stmt.on_conflict_do_update(
            index_elements=["source", "food_name", "cooking_method"],
            set_={
                "raw_profile":    stmt.excluded.raw_profile,
                "cooked_profile": stmt.excluded.cooked_profile,
                "food_group":     stmt.excluded.food_group,
                "yield_f":        stmt.excluded.yield_f,
                "is_theoretical": stmt.excluded.is_theoretical,
            },
        )
        with engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

    # Theoretical pairs: insert-or-skip — never overwrite known data.
    if theoretical_pairs:
        stmt = insert(TrainingPair).values(theoretical_pairs)
        stmt = stmt.on_conflict_do_nothing(index_elements=["source", "food_name", "cooking_method"])
        with engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

    log.info(
        "Upserted %d pairs into training_pairs (%d known, %d theoretical)",
        len(known_pairs) + len(theoretical_pairs), len(known_pairs), len(theoretical_pairs),
    )
