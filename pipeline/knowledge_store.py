"""Read cooking pairs and ingredient metadata directly from the Redis knowledge cache.

Key layout (same as pocket-nutrition-community redis_store.py):
  knowledge:ingredient:{slug}              → JSON ingredient dict
  knowledge:relationship:{raw_id}:{method} → JSON relationship dict
  knowledge:recipe:{id}                    → JSON recipe dict
"""
from __future__ import annotations

import json
import os

import redis as _redis

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/2")

_pool = _redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)


def _r() -> _redis.Redis:
    return _redis.Redis(connection_pool=_pool)


def _scan_keys(pattern: str) -> list[str]:
    r = _r()
    keys: list[str] = []
    cursor = 0
    while True:
        cursor, batch = r.scan(cursor, match=pattern, count=500)
        keys.extend(batch)
        if cursor == 0:
            break
    return keys


def fetch_relationships() -> list[dict]:
    """Return all cooking relationships from Redis as a list of dicts."""
    keys = _scan_keys("knowledge:relationship:*")
    if not keys:
        return []
    values = _r().mget(keys)
    return [json.loads(v) for v in values if v is not None]


def fetch_ingredients() -> dict[str, dict]:
    """Return all ingredients from Redis as {slug: ingredient_dict}."""
    keys = _scan_keys("knowledge:ingredient:*")
    if not keys:
        return {}
    slugs = [k.removeprefix("knowledge:ingredient:") for k in keys]
    values = _r().mget(keys)
    return {
        slug: json.loads(v)
        for slug, v in zip(slugs, values)
        if v is not None
    }


def fetch_recipes() -> dict[str, dict]:
    """Return all recipes from Redis as {recipe_id: recipe_dict}."""
    keys = _scan_keys("knowledge:recipe:*")
    if not keys:
        return {}
    recipe_ids = [k.removeprefix("knowledge:recipe:") for k in keys]
    values = _r().mget(keys)
    return {
        recipe_id: json.loads(v)
        for recipe_id, v in zip(recipe_ids, values)
        if v is not None
    }


def fetch_mappings() -> list[dict]:
    """Return all source mappings from Redis as a list of dicts.

    Key layout: knowledge:mapping:{source}:{source_id}
    Only returns mappings with a non-null canonical_ingredient_id.
    """
    keys = _scan_keys("knowledge:mapping:*")
    if not keys:
        return []
    values = _r().mget(keys)
    result = []
    for v in values:
        if v is None:
            continue
        m = json.loads(v)
        if m.get("canonical_ingredient_id"):
            result.append(m)
    return result
