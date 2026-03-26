# rf baselines vs cascade in models.py: dc openings, elec, water
# dc rf uses extra cols; elec water rf use same feature lists as cascade heads

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import DC_FEATS, ELEC_FEATS, WATER_FEATS
from src.models import score

warnings.filterwarnings("ignore")


def openings_rf_feature_columns() -> list[str]:
    # DC_FEATS plus water lag and raw usage from panel
    extra = ["water_lag1", "electricity_usage_mwh", "water_usage_mgal"]
    out: list[str] = []
    seen: set[str] = set()
    for c in list(DC_FEATS) + extra:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


RF_DEFAULTS = dict(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
)


def _year_weights_for_training(years: np.ndarray) -> np.ndarray:
    # same as _sample_weights in models.py for dc cascade
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
    cols = rf_feats + ["openings"]
    train = featured[
        (featured["year"] < train_cutoff) & (featured["year"] <= dc_cutoff)
    ].dropna(subset=cols)
    test = featured[
        (featured["year"] >= train_cutoff) & (featured["year"] <= dc_cutoff)
    ].dropna(subset=cols)
    return train, test


def _elec_train_test_frames(
    featured: pd.DataFrame,
    elec_feats: list[str],
    train_cutoff: int,
    dc_cutoff: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # same row logic as train_all_models elec_data in models.py
    cols = elec_feats + ["electricity_usage_mwh"]
    elec_data = featured[featured["year"] <= dc_cutoff].dropna(subset=cols)
    train = elec_data[elec_data["year"] < train_cutoff]
    test = elec_data[elec_data["year"] >= train_cutoff]
    return train, test


def _water_train_test_frames(
    featured: pd.DataFrame,
    water_feats: list[str],
    train_cutoff: int,
    dc_cutoff: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # same as models.py water_data year 2000 to dc_cutoff
    cols = water_feats + ["water_usage_mgal"]
    water_data = featured[
        (featured["year"] >= 2000) & (featured["year"] <= dc_cutoff)
    ].dropna(subset=cols)
    train = water_data[water_data["year"] < train_cutoff]
    test = water_data[water_data["year"] >= train_cutoff]
    return train, test


def fit_openings_random_forest(
    featured: pd.DataFrame,
    rf_feats: list[str] | None = None,
    train_cutoff: int = 2016,
    dc_cutoff: int = 2020,
    rf_params: dict | None = None,
) -> tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame]:
    feats = rf_feats if rf_feats is not None else openings_rf_feature_columns()
    train_df, test_df = _openings_train_test_frames(
        featured, feats, train_cutoff, dc_cutoff
    )
    x_train = train_df[feats].values
    y_train = train_df["openings"].values
    weights = _year_weights_for_training(train_df["year"].values)
    params = {**RF_DEFAULTS, **(rf_params or {})}
    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train, sample_weight=weights)
    return model, train_df, test_df


def fit_elec_random_forest(
    featured: pd.DataFrame,
    elec_feats: list[str] | None = None,
    train_cutoff: int = 2016,
    dc_cutoff: int = 2020,
    rf_params: dict | None = None,
) -> tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame]:
    # cascade elec in models.py has no sample weights on fit
    feats = elec_feats if elec_feats is not None else list(ELEC_FEATS)
    train_df, test_df = _elec_train_test_frames(
        featured, feats, train_cutoff, dc_cutoff
    )
    x_train = train_df[feats].values
    y_train = train_df["electricity_usage_mwh"].values
    params = {**RF_DEFAULTS, **(rf_params or {})}
    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)
    return model, train_df, test_df


def fit_water_random_forest(
    featured: pd.DataFrame,
    water_feats: list[str] | None = None,
    train_cutoff: int = 2016,
    dc_cutoff: int = 2020,
    rf_params: dict | None = None,
) -> tuple[RandomForestRegressor, pd.DataFrame, pd.DataFrame]:
    feats = water_feats if water_feats is not None else list(WATER_FEATS)
    train_df, test_df = _water_train_test_frames(
        featured, feats, train_cutoff, dc_cutoff
    )
    x_train = train_df[feats].values
    y_train = train_df["water_usage_mgal"].values
    params = {**RF_DEFAULTS, **(rf_params or {})}
    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)
    return model, train_df, test_df


def fit_all_random_forests(
    featured: pd.DataFrame,
    train_cutoff: int = 2016,
    dc_cutoff: int = 2020,
    openings_rf_feats: list[str] | None = None,
    rf_params: dict | None = None,
) -> dict[str, RandomForestRegressor]:
    # one rf per head, same cutoffs as train_all_models
    of = openings_rf_feats if openings_rf_feats is not None else openings_rf_feature_columns()
    dc_m, _, _ = fit_openings_random_forest(
        featured, of, train_cutoff, dc_cutoff, rf_params
    )
    el_m, _, _ = fit_elec_random_forest(
        featured, None, train_cutoff, dc_cutoff, rf_params
    )
    wa_m, _, _ = fit_water_random_forest(
        featured, None, train_cutoff, dc_cutoff, rf_params
    )
    return {"dc": dc_m, "elec": el_m, "water": wa_m}


def predict_openings_random_forest(
    model: RandomForestRegressor,
    rf_feats: list[str],
    df: pd.DataFrame,
) -> np.ndarray:
    raw = model.predict(df[rf_feats].values)
    return np.clip(raw, 0.0, None)


def predict_elec_random_forest(
    model: RandomForestRegressor,
    elec_feats: list[str],
    df: pd.DataFrame,
) -> np.ndarray:
    raw = model.predict(df[elec_feats].values)
    return np.clip(raw, 0.0, None)


def predict_water_random_forest(
    model: RandomForestRegressor,
    water_feats: list[str],
    df: pd.DataFrame,
) -> np.ndarray:
    raw = model.predict(df[water_feats].values)
    return np.clip(raw, 0.0, None)


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def build_aligned_test_predictions(
    featured: pd.DataFrame,
    models,
    rf_model: RandomForestRegressor,
    rf_feats: list[str] | None = None,
    train_cutoff: int = 2016,
    dc_cutoff: int = 2020,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    from src.features import DC_FEATS as _DC

    feats = rf_feats if rf_feats is not None else openings_rf_feature_columns()
    _, test_df = _openings_train_test_frames(
        featured, feats, train_cutoff, dc_cutoff
    )
    y_true = test_df["openings"].values
    pred_rf = predict_openings_random_forest(rf_model, feats, test_df)
    pred_cascade = models["dc"].predict(test_df[_DC].values)
    return test_df, y_true, pred_rf, pred_cascade


def build_full_rf_vs_cascade_metrics(
    models,
    rf_models: dict[str, RandomForestRegressor],
    test_dfs: dict,
    featured: pd.DataFrame,
    openings_rf_feats: list[str] | None = None,
    train_cutoff: int = 2016,
    dc_cutoff: int = 2020,
) -> pd.DataFrame:
    # one table: target x model with mae mse rmse r2
    rows: list[dict] = []
    of = openings_rf_feats if openings_rf_feats is not None else openings_rf_feature_columns()

    _, y_dc, pr_dc, pc_dc = build_aligned_test_predictions(
        featured, models, rf_models["dc"], of, train_cutoff, dc_cutoff
    )
    for name, pred in [("RF", pr_dc), ("Cascade", pc_dc)]:
        m = _metrics_dict(y_dc, pred)
        rows.append({"target": "openings", "model": name, **m})

    e_te = test_dfs["elec"]
    ye = e_te["electricity_usage_mwh"].values
    pe_rf = predict_elec_random_forest(rf_models["elec"], list(ELEC_FEATS), e_te)
    pe_c = e_te["pred_elec"].values
    for name, pred in [("RF", pe_rf), ("Cascade", pe_c)]:
        rows.append({"target": "electricity_mwh", "model": name, **_metrics_dict(ye, pred)})

    w_te = test_dfs["water"]
    yw = w_te["water_usage_mgal"].values
    pw_rf = predict_water_random_forest(rf_models["water"], list(WATER_FEATS), w_te)
    pw_c = w_te["pred_water"].values
    for name, pred in [("RF", pw_rf), ("Cascade", pw_c)]:
        rows.append({"target": "water_mgal", "model": name, **_metrics_dict(yw, pred)})

    return pd.DataFrame(rows)


def print_rf_vs_cascade_scores(
    models,
    rf_models: dict[str, RandomForestRegressor],
    test_dfs: dict,
    featured: pd.DataFrame,
    openings_rf_feats: list[str] | None = None,
    train_cutoff: int = 2016,
    dc_cutoff: int = 2020,
) -> None:
    # score() from models.py same print format as main notebook
    of = openings_rf_feats if openings_rf_feats is not None else openings_rf_feature_columns()
    _, y_dc, pr_dc, pc_dc = build_aligned_test_predictions(
        featured, models, rf_models["dc"], of, train_cutoff, dc_cutoff
    )
    score(y_dc, pr_dc, label="RF openings")
    score(y_dc, pc_dc, label="Cascade openings")

    e_te = test_dfs["elec"]
    ye = e_te["electricity_usage_mwh"].values
    pe_rf = predict_elec_random_forest(rf_models["elec"], list(ELEC_FEATS), e_te)
    pe_c = e_te["pred_elec"].values
    score(ye, pe_rf, label="RF electricity")
    score(ye, pe_c, label="Cascade electricity")

    w_te = test_dfs["water"]
    yw = w_te["water_usage_mgal"].values
    pw_rf = predict_water_random_forest(rf_models["water"], list(WATER_FEATS), w_te)
    pw_c = w_te["pred_water"].values
    score(yw, pw_rf, label="RF water")
    score(yw, pw_c, label="Cascade water")


def build_history_prediction_frame(
    featured: pd.DataFrame,
    models,
    rf_model: RandomForestRegressor,
    rf_feats: list[str] | None = None,
    dc_cutoff: int = 2020,
    train_cutoff: int = 2016,
) -> pd.DataFrame:
    from src.features import DC_FEATS as _DC

    feats = rf_feats if rf_feats is not None else openings_rf_feature_columns()
    hist = featured[featured["year"] <= dc_cutoff].dropna(subset=feats + ["openings"]).copy()
    hist["pred_cascade"] = models["dc"].predict(hist[_DC].values)
    hist["pred_rf"] = predict_openings_random_forest(rf_model, feats, hist)
    hist["period"] = np.where(hist["year"] < train_cutoff, "train", "test")
    return hist[
        ["state_abbrev", "year", "openings", "pred_cascade", "pred_rf", "period"]
    ].copy()


def score_dc_comparison(
    y_true: np.ndarray,
    y_pred_rf: np.ndarray,
    y_pred_cascade: np.ndarray,
) -> pd.DataFrame:
    score(y_true, y_pred_rf, label="RF (openings)")
    score(y_true, y_pred_cascade, label="Cascade (openings)")
    rows = []
    for name, pred in [("RF", y_pred_rf), ("Cascade", y_pred_cascade)]:
        rows.append({"model": name, **_metrics_dict(y_true, pred)})
    return pd.DataFrame(rows)


def reload_pipeline_for_section8(
    datacenters_path: str = "../data/datacenters_clean.csv",
    elec_path: str = "../data/electricity_clean.csv",
    water_path: str = "../data/water_clean.csv",
):
    from src.panel import build_panel
    from src.features import add_features, DC_FEATS, ELEC_FEATS, WATER_FEATS
    from src.models import train_all_models

    panel = build_panel(datacenters_path, elec_path, water_path)
    featured = add_features(panel)
    models, test_dfs = train_all_models(
        featured, DC_FEATS, ELEC_FEATS, WATER_FEATS
    )
    return panel, featured, models, test_dfs


RF_DC_DEFAULTS = RF_DEFAULTS
fit_dc_random_forest = fit_openings_random_forest
predict_dc_random_forest = predict_openings_random_forest
