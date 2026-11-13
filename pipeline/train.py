"""Train XGBoost model on training_pairs and export model.joblib artifact."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_OUTPUT_PATH = Path(os.getenv("MODEL_OUTPUT_PATH", "/models/model.joblib"))

# All 24 nutrients used as both input features and training targets.
# XGBoost handles NaN natively for missing feature values (e.g. USDA partial profiles).
FEATURE_COLUMNS = TARGET_COLUMNS = [
    "energy_kcal", "water_g", "protein_g", "fat_g", "carbohydrates_g",
    "fiber_g", "sugars_g", "calcium_mg", "iron_mg", "magnesium_mg",
    "sodium_mg", "vitamin_c_mg",
    "saturated_fat_g", "monounsaturated_fat_g", "polyunsaturated_fat_g",
    "cholesterol_mg", "potassium_mg", "phosphorus_mg", "zinc_mg",
    "vitamin_a_ug", "vitamin_d_ug", "vitamin_b6_mg", "vitamin_b12_ug",
    "niacin_mg",
]

COOKING_METHODS = ["boiled", "fried", "grilled", "roasted", "steamed"]
FOOD_GROUPS = ["cereal", "dairy", "fish", "meat", "shellfish", "vegetable"]
PLANT_FOOD_GROUPS = frozenset({"vegetable", "cereal"})


_KNOWN_GROUPS = frozenset({"meat", "fish", "shellfish", "vegetable", "cereal", "dairy"})


def _canonical_group(raw_group: str | None) -> str:
    if not raw_group:
        return "other"
    stripped = raw_group.strip()
    return stripped if stripped in _KNOWN_GROUPS else "other"


_MIN_GROUP_YIELDS = 3


def _group_yield_factors(pairs) -> dict[str, dict[str, float]]:
    """Median known yield per (food_group, cooking_method), for groups with enough data.

    ml falls back to it when a cooked weight must be converted to raw weight and the
    ingredient has no yield of its own for that method. Keyed by the knowledge food_group
    string, which is what the API sends as raw_profile.food_group. Theoretical pairs are
    excluded: their yield already is a group median.
    """
    buckets: dict[tuple[str, str], list[float]] = {}
    for p in pairs:
        group = (p.food_group or "").strip()
        if p.is_theoretical or not group or p.yield_f is None or p.yield_f <= 0:
            continue
        buckets.setdefault((group, p.cooking_method), []).append(float(p.yield_f))

    result: dict[str, dict[str, float]] = {}
    for (group, method), values in buckets.items():
        if len(values) >= _MIN_GROUP_YIELDS:
            result.setdefault(group, {})[method] = float(np.median(values))
    return result


def _masked_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-target MAE over the rows where the target was measured (NaN truth is skipped).

    A target with no measured row in ``y_true`` gets NaN.
    """
    observed = ~np.isnan(y_true)
    abs_err = np.where(observed, np.abs(np.nan_to_num(y_true) - y_pred), 0.0)
    counts = observed.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, abs_err.sum(axis=0) / counts, np.nan)


def _fit_masked(
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    n_jobs: int = 1,
) -> MultiOutputRegressor:
    """Fit one XGBoost regressor per target, each on the rows where that target is measured.

    Imputing missing targets (e.g. with the column median) would teach the model a constant
    for sparse nutrients. Masking keeps only real observations per target. The result is a
    plain ``MultiOutputRegressor`` whose ``estimators_`` are set directly, so ml can load
    and call ``predict()`` on it without any custom class.
    """
    estimators = []
    for j in range(y.shape[1]):
        observed = ~np.isnan(y[:, j])
        if not observed.any():
            raise ValueError(f"target {j} has no measured value — cannot train it")
        est = XGBRegressor(**params, random_state=42, n_jobs=n_jobs, verbosity=0)
        est.fit(X[observed], y[observed, j], sample_weight=weights[observed])
        estimators.append(est)

    model = MultiOutputRegressor(XGBRegressor(**params, random_state=42, n_jobs=n_jobs, verbosity=0))
    model.estimators_ = estimators
    model.n_features_in_ = X.shape[1]
    return model


def _optuna_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
) -> float:
    n_estimators = trial.suggest_int("n_estimators", 100, 600)
    max_depth = trial.suggest_int("max_depth", 3, 7)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True)
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample,
        "reg_alpha": reg_alpha,
    }

    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    maes: list[float] = []
    for train_idx, val_idx in cv.split(X, y, groups):
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        model = _fit_masked(params, X_t, y_t, weights[train_idx])
        maes.append(float(np.nanmean(_masked_mae(y_v, model.predict(X_v)))))
    return float(np.nanmean(maes))


def train_and_export() -> None:
    from pipeline.db import TrainingPair, get_session
    from pipeline.knowledge_store import fetch_ingredients

    session = get_session()
    pairs = session.query(TrainingPair).all()
    session.close()

    ingredients = fetch_ingredients()

    if not pairs:
        raise RuntimeError("No training pairs found — run transform step first")

    log.info("Training on %d pairs", len(pairs))

    sources = np.array([p.source for p in pairs])
    food_names = np.array([p.food_name for p in pairs])
    cooking_methods_raw = np.array([[p.cooking_method] for p in pairs])
    food_groups_raw = np.array([[_canonical_group(p.food_group)] for p in pairs])
    theoretical_flags = np.array([p.is_theoretical for p in pairs])

    raw_profiles = np.array(
        [[json.loads(p.raw_profile).get(c) for c in FEATURE_COLUMNS] for p in pairs],
        dtype=float,
    )
    cooked_raw = np.array(
        [[json.loads(p.cooked_profile).get(c) for c in TARGET_COLUMNS] for p in pairs],
        dtype=float,
    )

    # Drop rows where all core macros are NaN (completely unusable).
    core_idx = [FEATURE_COLUMNS.index(c) for c in ["energy_kcal", "protein_g", "fat_g", "carbohydrates_g"]]
    valid = ~np.isnan(raw_profiles[:, core_idx]).all(axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        log.warning("Dropped %d pairs with all-NaN core features", n_dropped)
    raw_profiles = raw_profiles[valid]
    cooked_raw = cooked_raw[valid]
    cooking_methods_raw = cooking_methods_raw[valid]
    food_groups_raw = food_groups_raw[valid]
    sources = sources[valid]
    food_names = food_names[valid]
    theoretical_flags = theoretical_flags[valid]

    log.info("Training on %d pairs after filter", len(raw_profiles))

    # Theoretical pairs (group-median yields) get half weight so known exact yields
    # drive the model signal while theoretical pairs still improve generalisation.
    weights = np.where(theoretical_flags, 0.5, 1.0)

    # Missing targets stay NaN: each nutrient is trained only on rows where it was measured.
    target_counts = {col: int((~np.isnan(cooked_raw[:, j])).sum()) for j, col in enumerate(TARGET_COLUMNS)}
    for col, n in target_counts.items():
        log.info("Training rows with measured %s: %d", col, n)

    ohe_cooking = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[COOKING_METHODS])
    cooking_encoded = ohe_cooking.fit_transform(cooking_methods_raw)

    ohe_foodgroup = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[FOOD_GROUPS])
    foodgroup_encoded = ohe_foodgroup.fit_transform(food_groups_raw)

    # Pipeline handles USDA partial profiles: impute missing nutrients with median, then scale.
    scaler = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    raw_scaled = scaler.fit_transform(raw_profiles)

    X = np.hstack([raw_scaled, cooking_encoded, foodgroup_encoded])
    y = cooked_raw

    n_trials = int(os.getenv("OPTUNA_TRIALS", "20"))
    log.info("Running Optuna hyperparameter search (%d trials)...", n_trials)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _optuna_objective(trial, X, y, food_names, weights),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params
    log.info("Best params: %s  MAE=%.4f", best_params, study.best_value)

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, weights, test_size=0.1, random_state=42,
    )
    final_model = _fit_masked(best_params, X_train, y_train, w_train, n_jobs=-1)

    y_pred = final_model.predict(X_test)
    mae_per_nutrient = _masked_mae(y_test, y_pred)
    cv_mae = {col: float(err) for col, err in zip(TARGET_COLUMNS, mae_per_nutrient)}
    for col, err in cv_mae.items():
        log.info("MAE %s: %.4f", col, err)

    # Build food_index and raw_index from training pairs (food_name = knowledge slug).
    food_index: dict[str, dict] = {}
    raw_index: dict[str, dict] = {}
    for p in pairs:
        slug = p.food_name  # already a canonical knowledge slug
        raw_p: dict = json.loads(p.raw_profile)
        if slug not in food_index:
            entry = {col: raw_p.get(col) for col in FEATURE_COLUMNS if raw_p.get(col) is not None}
            entry["food_group"] = _canonical_group(p.food_group)
            absorption = ingredients.get(slug, {}).get("absorption_factors") or {}
            if absorption:
                entry["absorption_factors"] = absorption
            food_index[slug] = entry
        if slug not in raw_index:
            raw_index[slug] = {col: raw_p.get(col) for col in TARGET_COLUMNS if raw_p.get(col) is not None}

    yield_factors: dict[str, dict[str, float]] = {}
    for p in pairs:
        if p.yield_f is None:
            continue
        yield_factors.setdefault(p.food_name, {})[p.cooking_method] = float(p.yield_f)

    known_ingredients = {p.food_name for p in pairs}
    plant_ingredient_ids = {
        iid for iid, entry in food_index.items()
        if entry.get("food_group") in PLANT_FOOD_GROUPS
    }

    artifact = {
        "model_version": "2.0",
        "model": final_model,
        "scaler": scaler,
        "cooking_encoder": ohe_cooking,
        "food_group_encoder": ohe_foodgroup,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "food_index": food_index,
        "raw_index": raw_index,
        "yield_factors": yield_factors,
        "group_yield_factors": _group_yield_factors(pairs),
        "known_ingredients": sorted(known_ingredients),
        "plant_ingredient_ids": sorted(plant_ingredient_ids),
        "cv_mae": cv_mae,
        "target_counts": target_counts,
    }
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    log.info("Model artifact saved to %s", MODEL_OUTPUT_PATH)
    log.info(
        "Food index: %d entries, %d plant, %d with yield factors",
        len(food_index), len(plant_ingredient_ids), len(yield_factors),
    )
