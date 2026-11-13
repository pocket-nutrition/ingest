"""Unit tests for the pipeline helpers (no DB, no network)."""
import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.utils import (
    _missing,
    impute_water_g,
    normalize_to_raw_basis,
    slugify,
)
from pipeline.transform import _derive_cooked_profile


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------

def test_slugify_basic():
    assert slugify("Poulet, filet") == "poulet_filet"


def test_slugify_accents():
    assert slugify("bœuf, haché") == "boeuf_hache"


def test_slugify_ligature():
    assert slugify("épinards") == "epinards"


def test_slugify_length_cap():
    long_name = "a" * 200
    assert len(slugify(long_name)) == 128


def test_impute_water_g_basic():
    result = impute_water_g(20.0, 5.0, 10.0, 2.0)
    assert result == pytest.approx(63.0)


def test_impute_water_g_clamped():
    result = impute_water_g(0.0, 0.0, 0.0, 0.0)
    assert result == pytest.approx(100.0)


def test_impute_water_g_none_input():
    assert impute_water_g(None, 5.0, 10.0) is None


def test_normalize_to_raw_basis():
    profile = {"energy_kcal": 200.0, "protein_g": 30.0, "fat_g": None}
    result = normalize_to_raw_basis(profile, 0.8)
    assert result["energy_kcal"] == pytest.approx(160.0)
    assert result["protein_g"] == pytest.approx(24.0)
    assert result["fat_g"] is None


# ---------------------------------------------------------------------------
# transform helpers
# ---------------------------------------------------------------------------

# Training targets are expressed per 100 g of RAW input — the basis ml scales by
# (raw weight / 100, after converting a cooked weight with the yield). Cooking does not create
# nutrients, so per raw gram they are conserved (times retention); only water follows the mass
# change. Concentration per 100 g of cooked food is then recovered by ml via the yield.

def test_derive_cooked_profile_conserves_nutrients_per_raw_gram_and_loses_water():
    # yield=0.8: 100 g raw becomes 80 g cooked; the 20 g lost is water
    raw = {"energy_kcal": 165.0, "protein_g": 23.0, "water_g": 65.0}
    cooked = _derive_cooked_profile(raw, 0.8)
    assert cooked["water_g"] == pytest.approx(65.0 - 100.0 * 0.2)
    assert cooked["energy_kcal"] == pytest.approx(165.0)
    assert cooked["protein_g"] == pytest.approx(23.0)


def test_derive_cooked_profile_is_per_raw_basis_not_per_cooked():
    # Guard against re-introducing the per-cooked target (raw / yield): ml already divides
    # a cooked weight by the yield, so a per-cooked target concentrates nutrients twice.
    raw = {"energy_kcal": 110.0}
    assert _derive_cooked_profile(raw, 0.7)["energy_kcal"] != pytest.approx(110.0 / 0.7)


def test_derive_cooked_profile_water_clamped_to_zero():
    raw = {"water_g": 5.0, "energy_kcal": 300.0}
    cooked = _derive_cooked_profile(raw, 0.5)
    assert cooked["water_g"] == 0.0
    assert cooked["energy_kcal"] == pytest.approx(300.0)


def test_derive_cooked_profile_absorbed_water_when_yield_above_one():
    # yield=1.5: 100 g dry pasta absorbs 50 g water; nutrients per raw gram are unchanged
    raw = {"energy_kcal": 350.0, "protein_g": 12.0, "water_g": 12.0}
    cooked = _derive_cooked_profile(raw, 1.5)
    assert cooked["water_g"] == pytest.approx(12.0 + 50.0)
    assert cooked["energy_kcal"] == pytest.approx(350.0)
    assert cooked["protein_g"] == pytest.approx(12.0)


def test_derive_cooked_profile_no_water_key():
    raw = {"energy_kcal": 200.0, "protein_g": 10.0}
    cooked = _derive_cooked_profile(raw, 0.7)
    assert cooked["energy_kcal"] == pytest.approx(200.0)
    assert cooked["protein_g"] == pytest.approx(10.0)


def test_derive_cooked_profile_none_values_preserved():
    raw = {"energy_kcal": 100.0, "vitamin_c_mg": None}
    cooked = _derive_cooked_profile(raw, 0.8)
    assert cooked["vitamin_c_mg"] is None
    assert cooked["energy_kcal"] == pytest.approx(100.0)


def test_derive_cooked_profile_zero_yield_returns_raw():
    raw = {"energy_kcal": 100.0, "protein_g": 10.0}
    cooked = _derive_cooked_profile(raw, 0.0)
    assert cooked == raw


def test_derive_cooked_profile_applies_retention_factor():
    # Boiled spinach: vitamin_c_mg retention ≈ 0.55; energy_kcal has no retention factor
    raw = {"vitamin_c_mg": 28.0, "energy_kcal": 23.0}
    retention = {"vitamin_c_mg": 0.55}
    cooked = _derive_cooked_profile(raw, 0.8375, nutrient_retention=retention)
    assert cooked["vitamin_c_mg"] == pytest.approx(28.0 * 0.55, rel=1e-3)
    assert cooked["energy_kcal"] == pytest.approx(23.0, rel=1e-3)


def test_derive_cooked_profile_retention_missing_key_defaults_to_one():
    raw = {"protein_g": 20.0}
    cooked = _derive_cooked_profile(raw, 0.7, nutrient_retention={"vitamin_c_mg": 0.55})
    assert cooked["protein_g"] == pytest.approx(20.0, rel=1e-3)


def test_derive_cooked_profile_none_retention_ignored():
    raw = {"energy_kcal": 100.0}
    cooked_no_ret = _derive_cooked_profile(raw, 0.8, nutrient_retention=None)
    cooked_empty = _derive_cooked_profile(raw, 0.8, nutrient_retention={})
    assert cooked_no_ret == cooked_empty


# ---------------------------------------------------------------------------
# knowledge_store (mocked Redis)
# ---------------------------------------------------------------------------

def test_fetch_relationships_returns_parsed_dicts():
    from pipeline.knowledge_store import fetch_relationships

    rel = {"raw_id": "chicken_breast", "cooking_method": "grilled", "yield_factor": 0.74}
    mock_r = MagicMock()
    mock_r.scan.side_effect = [(0, ["knowledge:relationship:chicken_breast:grilled"])]
    mock_r.mget.return_value = [json.dumps(rel)]

    with patch("pipeline.knowledge_store._r", return_value=mock_r):
        result = fetch_relationships()

    assert len(result) == 1
    assert result[0]["raw_id"] == "chicken_breast"
    assert result[0]["yield_factor"] == pytest.approx(0.74)


def test_fetch_relationships_empty_redis():
    from pipeline.knowledge_store import fetch_relationships

    mock_r = MagicMock()
    mock_r.scan.side_effect = [(0, [])]

    with patch("pipeline.knowledge_store._r", return_value=mock_r):
        result = fetch_relationships()

    assert result == []


def test_fetch_ingredients_returns_slug_dict():
    from pipeline.knowledge_store import fetch_ingredients

    ing = {"id": "chicken_breast", "food_group": "meat", "name_en": "Chicken breast"}
    mock_r = MagicMock()
    mock_r.scan.side_effect = [(0, ["knowledge:ingredient:chicken_breast"])]
    mock_r.mget.return_value = [json.dumps(ing)]

    with patch("pipeline.knowledge_store._r", return_value=mock_r):
        result = fetch_ingredients()

    assert "chicken_breast" in result
    assert result["chicken_breast"]["food_group"] == "meat"
