"""Shared utilities for the pocket-nutrition-ingest pipeline."""
from __future__ import annotations

import math
import re
import unicodedata

_FR_LIGATURES = str.maketrans({"œ": "oe", "æ": "ae", "Œ": "OE", "Æ": "AE"})
_NON_WORD = re.compile(r"[^\w]+")

def _missing(v: float | None) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def impute_water_g(
    protein_g: float | None,
    fat_g: float | None,
    carbohydrates_g: float | None,
    fiber_g: float | None = None,
) -> float | None:
    if any(_missing(v) for v in (protein_g, fat_g, carbohydrates_g)):
        return None
    fiber = fiber_g if not _missing(fiber_g) else 0.0
    dry_matter = protein_g + fat_g + carbohydrates_g + fiber  # type: ignore[operator]
    return max(0.0, min(100.0, 100.0 - dry_matter))


def slugify(name: str) -> str:
    """Convert a food name to a stable ASCII slug used as ingredient_id."""
    s = name.lower().translate(_FR_LIGATURES)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = _NON_WORD.sub("_", s).strip("_")
    return s[:128]


def normalize_to_raw_basis(profile: dict, yield_f: float) -> dict:
    """Convert nutrient density from 'per 100g cooked' to 'per 100g raw input'."""
    return {k: (v * yield_f if v is not None else None) for k, v in profile.items()}
