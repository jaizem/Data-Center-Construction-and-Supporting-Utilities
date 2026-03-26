# cascade vs rf charts for section 8, mirrors forecast_plots model_fit style for dc elec water
# pngs under viz/model_compare
# matplotlib lazy until first plot

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

COMPARE_DIR = Path("viz/model_compare")

BLUE = "#1D4ED8"
ORANGE = "#EA580C"
GREY = "#9CA3AF"
RED = "#DC2626"

_plt = None


def _pyplot():
    global _plt
    if _plt is None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for model comparison plots. "
                "Install with: pip install matplotlib"
            ) from e

        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "figure.dpi": 130,
            }
        )
        _plt = plt
    return _plt


def _save(fig, name: str) -> None:
    plt = _pyplot()
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARE_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def model_fit_panel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
    quantity: str = "openings",
) -> None:
    # like model_fit in forecast_plots.py, scatter plus residual hist, mae mse r2 in box
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.35, s=22, color=BLUE)
    lim = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred))) * 1.05
    if lim <= 0:
        lim = 1.0
    ax.plot([0, lim], [0, lim], "--", color=ORANGE, lw=1.5, label="perfect fit")
    r2 = 1 - np.nansum((y_true - y_pred) ** 2) / np.nansum(
        (y_true - np.nanmean(y_true)) ** 2
    )
    mae = np.nanmean(np.abs(y_true - y_pred))
    mse = np.nanmean((y_true - y_pred) ** 2)
    ax.text(
        0.05,
        0.90,
        f"R2 = {r2:.3f}\nMAE = {mae:.2f}\nMSE = {mse:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax.set_xlabel(f"Actual ({quantity})")
    ax.set_ylabel(f"Predicted ({quantity})")
    ax.set_title("Actual vs Predicted", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    ax.hist(residuals, bins=35, color=BLUE, alpha=0.7, edgecolor="white")
    ax.axvline(0, color=ORANGE, lw=2, ls="--")
    ax.axvline(
        np.median(residuals),
        color=GREY,
        lw=1.5,
        ls=":",
        label=f"Median residual: {np.median(residuals):.2f}",
    )
    ax.set_xlabel("Residual (Pred minus Actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residuals", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, filename)


def feature_importance_rf(
    model: RandomForestRegressor,
    feat_names: list[str],
    title: str,
    filename: str,
) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    plt = _pyplot()
    imp = pd.Series(model.feature_importances_, index=feat_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, len(feat_names) * 0.42)))
    colors = [ORANGE if imp[f] > imp.quantile(0.75) else BLUE for f in imp.index]
    ax.barh(imp.index, imp.values, color=colors, alpha=0.85)
    ax.axvline(imp.mean(), color=GREY, lw=1, ls="--", label="Mean importance")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Importance score")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, filename)


def national_historical_comparison(
    hist: pd.DataFrame,
    train_cutoff: int = 2016,
    filename: str = "national_history_openings_rf_vs_cascade",
) -> None:
    plt = _pyplot()
    nat = (
        hist.groupby("year", as_index=False)
        .agg(
            openings=("openings", "sum"),
            pred_cascade=("pred_cascade", "sum"),
            pred_rf=("pred_rf", "sum"),
        )
        .sort_values("year")
    )

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(
        nat["year"],
        nat["openings"],
        color=GREY,
        lw=2.4,
        label="Actual (national sum)",
    )
    ax.plot(
        nat["year"],
        nat["pred_cascade"],
        color=ORANGE,
        lw=2,
        ls="--",
        alpha=0.9,
        label="Cascade (sum of preds)",
    )
    ax.plot(
        nat["year"],
        nat["pred_rf"],
        color=BLUE,
        lw=2,
        ls="-.",
        alpha=0.9,
        label="Random Forest (sum of preds)",
    )
    ax.axvline(
        train_cutoff - 0.5,
        color=RED,
        lw=1.2,
        ls=":",
        alpha=0.8,
        label=f"Train / test split ({train_cutoff})",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Openings (national total)")
    ax.set_title(
        "Historical U.S. openings: actual vs models (aligned rows, state-year level)",
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, filename)


def test_set_side_by_side_scatter(
    y_true: np.ndarray,
    pred_cascade: np.ndarray,
    pred_rf: np.ndarray,
    filename: str,
    quantity: str = "openings",
) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    y_max = max(float(np.nanmax(y_true)), float(np.nanmax(pred_cascade)), float(np.nanmax(pred_rf)))

    for ax, pred, title, color in zip(
        axes,
        [pred_cascade, pred_rf],
        ["Cascade", "Random Forest"],
        [ORANGE, BLUE],
    ):
        ax.scatter(y_true, pred, alpha=0.35, s=22, color=color)
        lim = max(y_max * 1.05, 1e-9)
        ax.plot([0, lim], [0, lim], "--", color=GREY, lw=1.2)
        ax.set_xlabel(f"Actual ({quantity})")
        ax.set_ylabel(f"Predicted ({quantity})")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    fig.suptitle(
        "Test set: actual vs predicted (same aligned rows)",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, filename)


def run_openings_model_comparison_plots(
    y_true: np.ndarray,
    pred_cascade: np.ndarray,
    pred_rf: np.ndarray,
    rf_model: RandomForestRegressor,
    rf_feat_names: list[str],
    hist_frame: pd.DataFrame,
    train_cutoff: int = 2016,
) -> None:
    # backwards compat wrapper, openings only
    run_all_rf_cascade_comparison_plots(
        y_openings=y_true,
        pred_cascade_openings=pred_cascade,
        pred_rf_openings=pred_rf,
        rf_dc_model=rf_model,
        openings_rf_feats=rf_feat_names,
        y_elec=np.array([]),
        pred_cascade_elec=np.array([]),
        pred_rf_elec=np.array([]),
        rf_elec_model=None,
        y_water=np.array([]),
        pred_cascade_water=np.array([]),
        pred_rf_water=np.array([]),
        rf_water_model=None,
        hist_openings=hist_frame,
        train_cutoff=train_cutoff,
    )


def run_all_rf_cascade_comparison_plots(
    y_openings: np.ndarray,
    pred_cascade_openings: np.ndarray,
    pred_rf_openings: np.ndarray,
    rf_dc_model: RandomForestRegressor | None,
    openings_rf_feats: list[str],
    y_elec: np.ndarray,
    pred_cascade_elec: np.ndarray,
    pred_rf_elec: np.ndarray,
    rf_elec_model: RandomForestRegressor | None,
    y_water: np.ndarray,
    pred_cascade_water: np.ndarray,
    pred_rf_water: np.ndarray,
    rf_water_model: RandomForestRegressor | None,
    hist_openings: pd.DataFrame | None,
    train_cutoff: int = 2016,
) -> None:
    from src.features import ELEC_FEATS as _EF
    from src.features import WATER_FEATS as _WF

    # dc openings
    if len(y_openings) > 0 and rf_dc_model is not None:
        model_fit_panel(
            y_openings,
            pred_cascade_openings,
            "Cascade - DC openings (test)",
            "test_fit_cascade_dc",
            quantity="count",
        )
        model_fit_panel(
            y_openings,
            pred_rf_openings,
            "Random Forest - DC openings (test)",
            "test_fit_rf_dc",
            quantity="count",
        )
        test_set_side_by_side_scatter(
            y_openings,
            pred_cascade_openings,
            pred_rf_openings,
            "test_scatter_dc_openings",
            quantity="count",
        )
        feature_importance_rf(
            rf_dc_model,
            openings_rf_feats,
            "Random Forest - DC openings feature importance",
            "rf_feature_importance_dc",
        )
        if hist_openings is not None and len(hist_openings) > 0:
            national_historical_comparison(hist_openings, train_cutoff=train_cutoff)

    # electricity
    if len(y_elec) > 0 and rf_elec_model is not None:
        model_fit_panel(
            y_elec,
            pred_cascade_elec,
            "Cascade - electricity (test)",
            "test_fit_cascade_elec",
            quantity="MWh",
        )
        model_fit_panel(
            y_elec,
            pred_rf_elec,
            "Random Forest - electricity (test)",
            "test_fit_rf_elec",
            quantity="MWh",
        )
        test_set_side_by_side_scatter(
            y_elec,
            pred_cascade_elec,
            pred_rf_elec,
            "test_scatter_elec",
            quantity="MWh",
        )
        feature_importance_rf(
            rf_elec_model,
            list(_EF),
            "Random Forest - electricity feature importance",
            "rf_feature_importance_elec",
        )

    # water
    if len(y_water) > 0 and rf_water_model is not None:
        model_fit_panel(
            y_water,
            pred_cascade_water,
            "Cascade - water (test)",
            "test_fit_cascade_water",
            quantity="Mgal",
        )
        model_fit_panel(
            y_water,
            pred_rf_water,
            "Random Forest - water (test)",
            "test_fit_rf_water",
            quantity="Mgal",
        )
        test_set_side_by_side_scatter(
            y_water,
            pred_cascade_water,
            pred_rf_water,
            "test_scatter_water",
            quantity="Mgal",
        )
        feature_importance_rf(
            rf_water_model,
            list(_WF),
            "Random Forest - water feature importance",
            "rf_feature_importance_water",
        )
