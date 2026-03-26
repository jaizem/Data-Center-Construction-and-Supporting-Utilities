# rf baseline on openings vs cascade dc model in models.py
# extra cols for elec water levels not only dc lags

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.features import DC_FEATS
from src.models import score

warnings.filterwarnings("ignore")


# feature list for rf, DC_FEATS is in features.py
def openings_rf_feature_columns() -> list[str]:
    # DC_FEATS already has elec lags and growth
    # water_lag1 same idea as WATER_FEATS in features.py
    # raw mwh mgal from panel.py merge
    extra = ["water_lag1", "electricity_usage_mwh", "water_usage_mgal"]
    out: list[str] = []
    seen: set[str] = set()
    for c in list(DC_FEATS) + extra:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


RF_DC_DEFAULTS = dict(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
)


def _year_weights_for_training(years: np.ndarray) -> np.ndarray:
    # same weighting as _sample_weights in models.py cascade
    w = np.ones(len(years), dtype=float)
    w[(years >= 2010) & (years < 2015)] = 1.5
    w[years >= 2015] = 3.0
    w[years >= 2018] = 4.0
    return w


def _openings_train_test_frames(
    featured: pd.DataFrame,
    rf_feats: list[str],
    train_cutoff: int,
    dc_cutoff: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # stricter dropna than train_all_models in models.py, that one only needs DC_FEATS + openings
    cols = rf_feats + ["openings"]
    train = featured[
        (featured["year"] < train_cutoff) & (featured["year"] <= dc_cutoff)
    ].dropna(subset=cols)
    test = featured[
        (featured["year"] >= train_cutoff) & (featured["year"] <= dc_cutoff)
    ].dropna(subset=cols)
    return train, test


def fit_openings_random_forest(
    featured: pd.DataFrame,
    rf_feats: list[str] | None = None,
    train_cutoff: int = 2016,
    dc_cutoff: int = 2020,
    rf_params: dict | None = None,
) -> tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame]:
    # returns model and the train test frames actually used
    feats = rf_feats if rf_feats is not None else openings_rf_feature_columns()
    train_df, test_df = _openings_train_test_frames(
        featured, feats, train_cutoff, dc_cutoff
    )

    x_train = train_df[feats].values
    y_train = train_df["openings"].values
    weights = _year_weights_for_training(train_df["year"].values)

    params = {**RF_DC_DEFAULTS, **(rf_params or {})}
    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train, sample_weight=weights)

    return model, train_df, test_df


def predict_openings_random_forest(
    model: RandomForestRegressor,
    rf_feats: list[str],
    df: pd.DataFrame,
) -> np.ndarray:
    # clip after predict openings shouldnt go negative
    raw = model.predict(df[rf_feats].values)
    return np.clip(raw, 0.0, None)


def score_dc_comparison(
    y_true: np.ndarray,
    y_pred_rf: np.ndarray,
    y_pred_cascade: np.ndarray,
) -> pd.DataFrame:
    # score() from models.py so same format as rest of main.ipynb
    score(y_true, y_pred_rf, label="RF (openings, elec+water-augmented)")
    score(y_true, y_pred_cascade, label="Cascade (DC openings)")

    rows = []
    for name, pred in [("RF", y_pred_rf), ("Cascade", y_pred_cascade)]:
        mae = mean_absolute_error(y_true, pred)
        rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
        r2 = r2_score(y_true, pred)
        rows.append({"model": name, "mae": mae, "rmse": rmse, "r2": r2})
    return pd.DataFrame(rows)


def build_aligned_test_predictions(
    featured: pd.DataFrame,
    models,
    rf_model: RandomForestRegressor,
    rf_feats: list[str] | None = None,
    train_cutoff: int = 2016,
    dc_cutoff: int = 2020,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # aligned test rows, cascade still uses DC_FEATS from features.py only
    from src.features import DC_FEATS as _DC

    feats = rf_feats if rf_feats is not None else openings_rf_feature_columns()
    _, test_df = _openings_train_test_frames(
        featured, feats, train_cutoff, dc_cutoff
    )
    y_true = test_df["openings"].values
    pred_rf = predict_openings_random_forest(rf_model, feats, test_df)
    pred_cascade = models["dc"].predict(test_df[_DC].values)
    return test_df, y_true, pred_rf, pred_cascade


def build_history_prediction_frame(
    featured: pd.DataFrame,
    models,
    rf_model: RandomForestRegressor,
    rf_feats: list[str] | None = None,
    dc_cutoff: int = 2020,
    train_cutoff: int = 2016,
) -> pd.DataFrame:
    # history through dc_cutoff for model_compare_plots, needs full rf cols
    from src.features import DC_FEATS as _DC

    feats = rf_feats if rf_feats is not None else openings_rf_feature_columns()
    hist = featured[featured["year"] <= dc_cutoff].dropna(subset=feats + ["openings"]).copy()

    hist["pred_cascade"] = models["dc"].predict(hist[_DC].values)
    hist["pred_rf"] = predict_openings_random_forest(rf_model, feats, hist)
    hist["period"] = np.where(hist["year"] < train_cutoff, "train", "test")
    return hist[
        ["state_abbrev", "year", "openings", "pred_cascade", "pred_rf", "period"]
    ].copy()


def reload_pipeline_for_section8(
    datacenters_path: str = "../data/datacenters_clean.csv",
    elec_path: str = "../data/electricity_clean.csv",
    water_path: str = "../data/water_clean.csv",
):
    # section 8 shortcut reruns pipeline same as top of main.ipynb
    from src.panel import build_panel
    from src.features import add_features, DC_FEATS, ELEC_FEATS, WATER_FEATS
    from src.models import train_all_models

    panel = build_panel(datacenters_path, elec_path, water_path)
    featured = add_features(panel)
    models, test_dfs = train_all_models(featured, DC_FEATS, ELEC_FEATS, WATER_FEATS)
    return panel, featured, models, test_dfs


# backwards compat old names
fit_dc_random_forest = fit_openings_random_forest
predict_dc_random_forest = predict_openings_random_forest
