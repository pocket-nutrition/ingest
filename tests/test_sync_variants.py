"""Tests for pipeline.sync_variants — no DB, no network."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from pipeline.sync_variants import (
    _MCCANCE_REMAP,
    _USDA_COLS,
    _confidence_weighted_median,
    _row_to_nutrients,
    build_nutrient_lookup,
    sync_variants,
)
from pipeline.db import NUTRIENT_COLUMNS


# ---------------------------------------------------------------------------
# _row_to_nutrients
# ---------------------------------------------------------------------------

def test_row_to_nutrients_ciqual_renames_proteins_g():
    row = {
        "proteins_g": 23.1,
        "energy_kcal": 165.0,
        "fat_g": 3.6,
        "carbohydrates_g": 0.0,
    }
    result = _row_to_nutrients(row, {"proteins_g": "protein_g"})
    assert result["protein_g"] == pytest.approx(23.1)
    assert result["energy_kcal"] == pytest.approx(165.0)
    assert result.get("proteins_g") is None  # original key must not appear in output


def test_row_to_nutrients_mccance_renames_carbohydrate_and_fibre():
    row = {
        "carbohydrate_g": 12.5,
        "fibre_g": 2.1,
        "protein_g": 5.0,
        "energy_kcal": 80.0,
    }
    result = _row_to_nutrients(row, _MCCANCE_REMAP)
    assert result["carbohydrates_g"] == pytest.approx(12.5)
    assert result["fiber_g"] == pytest.approx(2.1)
    assert result["protein_g"] == pytest.approx(5.0)


def test_row_to_nutrients_usda_nulls_unavailable_cols():
    row = {"energy_kcal": 200.0, "protein_g": 10.0, "fat_g": 5.0,
           "carbohydrates_g": 20.0, "fiber_g": 1.0, "sodium_mg": 50.0}
    result = _row_to_nutrients(row, {}, available=_USDA_COLS)
    assert result["energy_kcal"] == pytest.approx(200.0)
    assert result["protein_g"] == pytest.approx(10.0)
    # Columns not in USDA_COLS must be None
    assert result["calcium_mg"] is None
    assert result["vitamin_c_mg"] is None
    assert result["sugars_g"] is None


def test_row_to_nutrients_returns_all_nutrient_columns():
    result = _row_to_nutrients({}, {})
    assert set(result.keys()) == set(NUTRIENT_COLUMNS)


# ---------------------------------------------------------------------------
# build_nutrient_lookup — happy paths
# ---------------------------------------------------------------------------

def _make_session(source_row: dict | None = None):
    """Return a mock SQLAlchemy Session whose execute().mappings().first() returns source_row."""
    mock_result = MagicMock()
    mock_result.mappings.return_value.first.return_value = source_row
    session = MagicMock()
    session.execute.return_value = mock_result
    return session


def test_build_nutrient_lookup_returns_nutrients_for_ciqual_mapping():
    mapping = {
        "source": "ciqual",
        "source_id": "20001",
        "canonical_ingredient_id": "chicken_breast",
        "label": None,
    }
    ciqual_row = {
        "alim_code": 20001,
        "proteins_g": 23.1,
        "energy_kcal": 165.0,
        "fat_g": 3.6,
        "carbohydrates_g": 0.0,
        **{col: None for col in NUTRIENT_COLUMNS if col not in ("energy_kcal", "fat_g", "carbohydrates_g")},
    }

    session = _make_session(ciqual_row)

    with (
        patch("pipeline.sync_variants.fetch_mappings", return_value=[mapping]),
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
    ):
        result = build_nutrient_lookup()

    assert "chicken_breast" in result
    assert result["chicken_breast"]["energy_kcal"] == pytest.approx(165.0)
    assert result["chicken_breast"]["fat_g"] == pytest.approx(3.6)
    # Session used only for reading
    session.commit.assert_not_called()
    session.close.assert_called_once()


def test_build_nutrient_lookup_skips_unknown_source():
    mapping = {
        "source": "unknown_db",
        "source_id": "99",
        "canonical_ingredient_id": "mystery_food",
    }
    session = _make_session(None)

    with (
        patch("pipeline.sync_variants.fetch_mappings", return_value=[mapping]),
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
    ):
        result = build_nutrient_lookup()

    assert result == {}
    session.execute.assert_not_called()
    session.commit.assert_not_called()


def test_build_nutrient_lookup_skips_when_source_row_missing():
    mapping = {
        "source": "ciqual",
        "source_id": "99999",
        "canonical_ingredient_id": "ghost_food",
    }
    session = _make_session(None)  # source table returns nothing

    with (
        patch("pipeline.sync_variants.fetch_mappings", return_value=[mapping]),
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
    ):
        result = build_nutrient_lookup()

    assert result == {}
    # One execute: SELECT from ciqual_foods (returns None → skip)
    assert session.execute.call_count == 1
    session.commit.assert_not_called()


def test_build_nutrient_lookup_returns_empty_when_no_mappings():
    session = _make_session(None)

    with (
        patch("pipeline.sync_variants.fetch_mappings", return_value=[]),
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
    ):
        result = build_nutrient_lookup()

    assert result == {}
    session.execute.assert_not_called()


def test_build_nutrient_lookup_closes_session_on_db_error():
    mapping = {
        "source": "ciqual",
        "source_id": "20001",
        "canonical_ingredient_id": "chicken_breast",
    }
    session = MagicMock()
    session.execute.side_effect = RuntimeError("DB down")

    with (
        patch("pipeline.sync_variants.fetch_mappings", return_value=[mapping]),
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
        pytest.raises(RuntimeError),
    ):
        build_nutrient_lookup()

    session.close.assert_called_once()
    session.commit.assert_not_called()


def test_sync_variants_is_noop():
    """sync_variants is a compatibility shim; it must not perform any DB or ES I/O."""
    session = MagicMock()
    with (
        patch("pipeline.sync_variants.fetch_mappings") as mock_fetch,
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
    ):
        sync_variants()

    mock_fetch.assert_not_called()
    session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_mappings (knowledge_store integration)
# ---------------------------------------------------------------------------

def test_fetch_mappings_filters_unlinked():
    from pipeline.knowledge_store import fetch_mappings

    linked = {"source": "ciqual", "source_id": "1", "canonical_ingredient_id": "apple"}
    unlinked = {"source": "ciqual", "source_id": "2", "canonical_ingredient_id": None}

    mock_r = MagicMock()
    mock_r.scan.side_effect = [(0, ["knowledge:mapping:ciqual:1", "knowledge:mapping:ciqual:2"])]
    mock_r.mget.return_value = [json.dumps(linked), json.dumps(unlinked)]

    with patch("pipeline.knowledge_store._r", return_value=mock_r):
        result = fetch_mappings()

    assert len(result) == 1
    assert result[0]["canonical_ingredient_id"] == "apple"


# ---------------------------------------------------------------------------
# _confidence_weighted_median
# ---------------------------------------------------------------------------

def test_confidence_weighted_median_single_candidate():
    nutrients = {"energy_kcal": 89.0, "protein_g": 1.1, "fat_g": 0.3}
    result = _confidence_weighted_median([(nutrients, 1.0)])
    assert result["energy_kcal"] == pytest.approx(89.0)
    assert result["protein_g"] == pytest.approx(1.1)


def test_confidence_weighted_median_omits_none_columns():
    nutrients = {"energy_kcal": 89.0, "protein_g": None}
    result = _confidence_weighted_median([(nutrients, 1.0)])
    assert "energy_kcal" in result
    assert "protein_g" not in result


def test_confidence_weighted_median_equal_weights_picks_median():
    # Three values: 80, 90, 100 — equal weights → median is 90
    candidates = [
        ({"energy_kcal": 80.0}, 1.0),
        ({"energy_kcal": 100.0}, 1.0),
        ({"energy_kcal": 90.0}, 1.0),
    ]
    result = _confidence_weighted_median(candidates)
    assert result["energy_kcal"] == pytest.approx(90.0)


def test_confidence_weighted_median_high_confidence_pulls_toward_high_value():
    # 80 weight=1, 200 weight=9 → cumulative tips at 200
    candidates = [
        ({"energy_kcal": 80.0}, 1.0),
        ({"energy_kcal": 200.0}, 9.0),
    ]
    result = _confidence_weighted_median(candidates)
    assert result["energy_kcal"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# build_nutrient_lookup — cooking_method filtering and median aggregation
# ---------------------------------------------------------------------------

def test_build_nutrient_lookup_excludes_non_raw_mappings():
    """A mapping with cooking_method='dried' must not contribute to the raw profile."""
    raw_mapping = {
        "source": "ciqual", "source_id": "13005",
        "canonical_ingredient_id": "banana",
        "cooking_method": "raw", "confidence": 1.0,
    }
    dried_mapping = {
        "source": "ciqual", "source_id": "13089",
        "canonical_ingredient_id": "banana",
        "cooking_method": "dried", "confidence": 1.0,
    }
    raw_row = {
        "proteins_g": 1.1, "energy_kcal": 89.0, "fat_g": 0.3,
        "carbohydrates_g": 23.0,
        **{col: None for col in ["fiber_g", "sodium_mg", "sugars_g", "calcium_mg",
                                 "iron_mg", "magnesium_mg", "phosphorus_mg",
                                 "potassium_mg", "zinc_mg", "vitamin_c_mg",
                                 "vitamin_d_ug", "vitamin_a_ug", "vitamin_b6_mg",
                                 "vitamin_b12_ug", "niacin_mg", "saturated_fat_g",
                                 "monounsaturated_fat_g", "polyunsaturated_fat_g",
                                 "cholesterol_mg", "water_g", "alcohol_g",
                                 "organic_acid_g", "ash_g"]},
    }

    session = _make_session(raw_row)

    with (
        patch("pipeline.sync_variants.fetch_mappings", return_value=[raw_mapping, dried_mapping]),
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
    ):
        result = build_nutrient_lookup()

    assert "banana" in result
    assert result["banana"]["energy_kcal"] == pytest.approx(89.0)
    # Session called once (only for raw mapping, not dried)
    assert session.execute.call_count == 1


def test_build_nutrient_lookup_falls_back_to_dry_when_no_raw_mapping():
    """Shelf-stable foods (rolled oats, pasta, lentils) are only mapped as 'dry' — the uncooked,
    as-sold state. Without a raw mapping, the dry one is their starting profile."""
    dry_mapping = {
        "source": "ciqual", "source_id": "9311",
        "canonical_ingredient_id": "rolled_oat",
        "cooking_method": "dry", "confidence": 0.95,
    }
    session = _make_session({"proteins_g": 13.5, "energy_kcal": 371.0, "fat_g": 7.0,
                             "carbohydrates_g": 58.7})

    with (
        patch("pipeline.sync_variants.fetch_mappings", return_value=[dry_mapping]),
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
    ):
        result = build_nutrient_lookup()

    assert result["rolled_oat"]["energy_kcal"] == pytest.approx(371.0)


def test_build_nutrient_lookup_ignores_dry_when_raw_mapping_exists():
    """A fresh ingredient with both raw and dry mappings keeps the raw profile only."""
    raw_mapping = {
        "source": "ciqual", "source_id": "13005",
        "canonical_ingredient_id": "banana",
        "cooking_method": "raw", "confidence": 1.0,
    }
    dry_mapping = {
        "source": "ciqual", "source_id": "13089",
        "canonical_ingredient_id": "banana",
        "cooking_method": "dry", "confidence": 1.0,
    }
    session = _make_session({"proteins_g": 1.1, "energy_kcal": 89.0, "fat_g": 0.3,
                             "carbohydrates_g": 23.0})

    with (
        patch("pipeline.sync_variants.fetch_mappings", return_value=[raw_mapping, dry_mapping]),
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
    ):
        result = build_nutrient_lookup()

    assert result["banana"]["energy_kcal"] == pytest.approx(89.0)
    assert session.execute.call_count == 1


def test_build_nutrient_lookup_aggregates_multiple_raw_sources():
    """Two raw mappings → values are merged via confidence-weighted median."""
    mapping_a = {
        "source": "ciqual", "source_id": "1",
        "canonical_ingredient_id": "apple",
        "cooking_method": "raw", "confidence": 1.0,
    }
    mapping_b = {
        "source": "mccance", "source_id": "2",
        "canonical_ingredient_id": "apple",
        "cooking_method": "raw", "confidence": 1.0,
    }

    def _mock_execute(stmt, params=None):
        mock_result = MagicMock()
        if "ciqual" in str(stmt):
            mock_result.mappings.return_value.first.return_value = {
                "proteins_g": 0.3, "energy_kcal": 52.0, "fat_g": 0.2,
                "carbohydrates_g": 14.0,
            }
        else:
            mock_result.mappings.return_value.first.return_value = {
                "protein_g": 0.3, "energy_kcal": 56.0, "fat_g": 0.1,
                "carbohydrates_g": 15.0,
            }
        return mock_result

    session = MagicMock()
    session.execute.side_effect = _mock_execute

    with (
        patch("pipeline.sync_variants.fetch_mappings", return_value=[mapping_a, mapping_b]),
        patch("pipeline.sync_variants.SessionLocal", return_value=session),
    ):
        result = build_nutrient_lookup()

    assert "apple" in result
    # Median of [52, 56] with equal weights → 52 (first value >= 50% cumulative)
    assert result["apple"]["energy_kcal"] == pytest.approx(52.0)


def test_fetch_mappings_empty_redis():
    from pipeline.knowledge_store import fetch_mappings

    mock_r = MagicMock()
    mock_r.scan.side_effect = [(0, [])]

    with patch("pipeline.knowledge_store._r", return_value=mock_r):
        result = fetch_mappings()

    assert result == []
