"""Tests for OFF product indexing helpers in pipeline.index_search — no DB, no network."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.index_search import (
    _build_off_document,
    _build_recipe_document,
    _first_brand,
    _select_curated,
    _stream_off_products_from_db,
)


# ---------------------------------------------------------------------------
# _select_curated
# ---------------------------------------------------------------------------

def test_select_curated_drops_unmapped_ingredient():
    ingredients = {
        "chicken_breast": {"id": "chicken_breast"},
        "lasagna_noodle": {"id": "lasagna_noodle"},  # no source mapping
    }
    mapped = {"chicken_breast"}
    result = _select_curated(ingredients, mapped)
    assert "chicken_breast" in result
    assert "lasagna_noodle" not in result


def test_select_curated_empty_mapped_drops_everything():
    ingredients = {"chicken_breast": {"id": "chicken_breast"}}
    assert _select_curated(ingredients, set()) == {}


def test_select_curated_preserves_ingredient_dict():
    ingredients = {"rice": {"id": "rice", "food_group": "cereal"}}
    result = _select_curated(ingredients, {"rice"})
    assert result["rice"] == {"id": "rice", "food_group": "cereal"}


# ---------------------------------------------------------------------------
# _build_off_document
# ---------------------------------------------------------------------------

def test_build_off_document_fr_only():
    row = {"code": "3033490004896", "product_name": None, "product_name_fr": "Heineken 25cl"}
    doc = _build_off_document(row)
    assert doc["product_id"] == "off:3033490004896"
    assert doc["product_type"] == "off_product"
    assert doc["boost"] == pytest.approx(0.5)
    assert doc["has_training_data"] is False
    labels = doc["labels"]
    assert len(labels) == 1
    assert labels[0]["lang"] == "fr"
    assert labels[0]["text"] == "Heineken 25cl"


def test_build_off_document_both_names():
    row = {"code": "123", "product_name": "Heineken Beer", "product_name_fr": "Heineken Bière"}
    doc = _build_off_document(row)
    langs = [lbl["lang"] for lbl in doc["labels"]]
    assert "fr" in langs
    assert "en" in langs
    assert len(doc["labels"]) == 2


def test_build_off_document_null_names_fallback():
    row = {"code": "9999", "product_name": None, "product_name_fr": None}
    doc = _build_off_document(row)
    assert len(doc["labels"]) == 1
    assert doc["labels"][0]["text"] == "9999"


def test_build_off_document_id_prefixed():
    row = {"code": "42", "product_name": "Test", "product_name_fr": None}
    doc = _build_off_document(row)
    assert doc["_id"] == "off:42"
    assert doc["product_id"] == "off:42"


def test_build_off_document_normalized_label():
    row = {"code": "1", "product_name": None, "product_name_fr": "Eau minérale Évian"}
    doc = _build_off_document(row)
    assert doc["labels"][0]["normalized"] == "eau minerale evian"


def test_build_off_document_embeds_nutrients():
    row = {
        "code": "123",
        "product_name": "Nutella",
        "product_name_fr": "Nutella",
        "energy_kcal_100g": 539.0,
        "proteins_100g": 6.3,
        "fat_100g": 30.9,
        "carbohydrates_100g": 57.5,
        "sugars_100g": 56.3,
        "fiber_100g": 3.0,
        "sodium_100g": 0.107,  # grams in PG
    }
    doc = _build_off_document(row)
    assert doc["energy_kcal"] == pytest.approx(539.0)
    assert doc["protein_g"] == pytest.approx(6.3)
    assert doc["fat_g"] == pytest.approx(30.9)
    assert doc["carbohydrates_g"] == pytest.approx(57.5)
    assert doc["sugars_g"] == pytest.approx(56.3)
    assert doc["fiber_g"] == pytest.approx(3.0)
    # sodium in PG is grams; doc should have mg
    assert doc["sodium_mg"] == pytest.approx(107.0)


def test_build_off_document_null_nutrients_when_missing():
    row = {"code": "42", "product_name": "Test", "product_name_fr": None}
    doc = _build_off_document(row)
    assert doc["energy_kcal"] is None
    assert doc["protein_g"] is None
    assert doc["sodium_mg"] is None


def test_build_off_document_package_size_g_for_solid_product():
    """A gram-quantified product (the common case) must populate package_size_g, not just
    package_size_ml. Regression test for a copy-paste bug where both fields checked
    `_display_unit == 'ml'`, leaving package_size_g permanently null for solid foods."""
    row = {"code": "1", "product_name": "Yaourt nature", "product_name_fr": None, "quantity": "125 g"}
    doc = _build_off_document(row)
    assert doc["package_size_g"] == pytest.approx(125.0)
    assert doc["package_size_ml"] is None


def test_build_off_document_package_size_ml_for_liquid_product():
    row = {"code": "2", "product_name": "Lait", "product_name_fr": None, "quantity": "1 L"}
    doc = _build_off_document(row)
    assert doc["package_size_ml"] == pytest.approx(1000.0)
    assert doc["package_size_g"] is None


def test_build_off_document_package_size_null_for_unparseable_quantity():
    row = {"code": "3", "product_name": "Mystère", "product_name_fr": None, "quantity": "lot de 3"}
    doc = _build_off_document(row)
    assert doc["package_size_g"] is None
    assert doc["package_size_ml"] is None


def test_build_off_document_serving_size_g_from_serving_quantity():
    """The manufacturer's declared per-serving weight — what gives a pizza or tart a real
    "1 part" shortcut instead of only a whole-package one."""
    row = {"code": "4", "product_name": "Pizza 4 fromages", "product_name_fr": None, "serving_quantity": 100.0}
    doc = _build_off_document(row)
    assert doc["serving_size_g"] == pytest.approx(100.0)


def test_build_off_document_serving_size_g_null_when_missing():
    row = {"code": "5", "product_name": "Test", "product_name_fr": None}
    doc = _build_off_document(row)
    assert doc["serving_size_g"] is None


def test_build_off_document_serving_size_g_rejects_non_positive():
    row = {"code": "6", "product_name": "Test", "product_name_fr": None, "serving_quantity": 0.0}
    doc = _build_off_document(row)
    assert doc["serving_size_g"] is None


# ---------------------------------------------------------------------------
# _build_recipe_document
# ---------------------------------------------------------------------------

_RACLETTE = {
    "id": "raclette",
    "name_en": "traditional raclette",
    "name_fr": "raclette traditionnelle",
    "aliases_fr": ["raclette savoyarde"],
    "components": [
        {"ingredient_id": "potato", "weight_g": 200, "cooking_method": "boiled"},
        {"ingredient_id": "raclette_cheese", "weight_g": 100, "cooking_method": "grilled"},
    ],
}


def test_build_recipe_document_basic():
    doc = _build_recipe_document("raclette", _RACLETTE, {"potato", "raclette_cheese"})
    assert doc["_id"] == "recipe:raclette"
    assert doc["product_id"] == "recipe:raclette"
    assert doc["product_type"] == "recipe"
    assert doc["boost"] == pytest.approx(1.5)
    assert doc["components"] == [
        {"ingredient_id": "potato", "weight_g": 200.0, "cooking_method": "boiled"},
        {"ingredient_id": "raclette_cheese", "weight_g": 100.0, "cooking_method": "grilled"},
    ]
    texts = {(lbl["lang"], lbl["text"]) for lbl in doc["labels"]}
    assert ("fr", "raclette traditionnelle") in texts
    assert ("en", "traditional raclette") in texts
    assert ("fr", "raclette savoyarde") in texts


def test_build_recipe_document_skips_non_curated_component():
    doc = _build_recipe_document("raclette", _RACLETTE, {"potato"})  # cheese not curated
    assert doc is None


def test_build_recipe_document_skips_empty_components():
    recipe = {**_RACLETTE, "components": []}
    assert _build_recipe_document("raclette", recipe, {"potato"}) is None


# ---------------------------------------------------------------------------
# _stream_off_products_from_db
# ---------------------------------------------------------------------------

def test_stream_off_products_empty_result():
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = []
    session = MagicMock()
    session.execute.return_value = mock_result

    with patch("pipeline.index_search.SessionLocal", return_value=session):
        result = list(_stream_off_products_from_db())

    assert result == []
    session.close.assert_called_once()


def test_stream_off_products_respects_limit():
    # 10 rows available, but limit is 3 — stream must stop after 3 yielded docs
    rows = [
        {"code": str(i), "product_name": f"Product {i}", "product_name_fr": None}
        for i in range(10)
    ]
    call_count = [0]

    def all_side_effect():
        call_count[0] += 1
        return rows if call_count[0] == 1 else []

    mock_result = MagicMock()
    mock_result.mappings.return_value.all.side_effect = all_side_effect
    session = MagicMock()
    session.execute.return_value = mock_result

    with (
        patch("pipeline.index_search.SessionLocal", return_value=session),
        patch("pipeline.index_search._OFF_MAX_PRODUCTS", 3),
        patch("pipeline.index_search._OFF_CHUNK_SIZE", 3),
    ):
        result = list(_stream_off_products_from_db())

    assert len(result) == 3
    session.close.assert_called_once()


def test_stream_off_products_closes_session_on_error():
    session = MagicMock()
    session.execute.side_effect = RuntimeError("DB error")

    with (
        patch("pipeline.index_search.SessionLocal", return_value=session),
        pytest.raises(RuntimeError),
    ):
        list(_stream_off_products_from_db())

    session.close.assert_called_once()


# ---------------------------------------------------------------------------
# _first_brand and the brand field on OFF documents
#
# Brand exists on the document for one reason: thousands of OFF products carry a product_name that
# is the bare food word, so a client that renders label alone shows several identical rows. Measured
# against the live index, `?q=poulet` returned 29 products all labelled "Poulet".
# ---------------------------------------------------------------------------

def test_first_brand_takes_the_leading_entry():
    # OFF stores brands comma-separated; only the first fits a suggestion row.
    assert _first_brand("Coca-Cola,Coca Cola,The Coca-Cola Company") == "Coca-Cola"


def test_first_brand_strips_surrounding_whitespace():
    assert _first_brand("  Le Gaulois , Loue ") == "Le Gaulois"


def test_first_brand_treats_blank_as_absent():
    # An empty string would force every client to guard against it; None is the honest answer.
    for blank in (None, "", "   ", ",", " , "):
        assert _first_brand(blank) is None, f"{blank!r} should yield None"


def test_build_off_document_carries_the_brand():
    row = {"code": "123", "product_name": "Poulet", "brand": "Le Gaulois"}
    assert _build_off_document(row)["brand"] == "Le Gaulois"


def test_build_off_document_brand_is_null_when_off_has_none():
    row = {"code": "123", "product_name": "Poulet"}
    assert _build_off_document(row)["brand"] is None


def test_curated_and_recipe_documents_carry_no_brand():
    """
    Only products have a brand. A curated ingredient or a recipe must leave the field absent rather
    than empty, so the API maps it to null and the client falls back to the food-group icon.
    """
    recipe = _build_recipe_document("raclette", _RACLETTE, {"potato", "raclette_cheese"})
    assert recipe.get("brand") is None
