# U.S. Data Center Growth and Resource Demand Forecast

Predicts how many data centers will be built across U.S. states through 2030,
and what that means for commercial electricity and water demand.

## What it does

Trains three ML models on 1990-2020 data and forecasts through 2030:

- **DC openings** -- two-stage cascade model in `models.py` (will any open? then how many?)
- **Electricity** -- gradient boosting regression driven by projected DC count
- **Water** -- same approach as electricity

**Random Forest baselines** live in `treemodel.py`: three regressors trained on the same train/test year split as the cascade (2016-2020 test by default). The **openings** RF uses `DC_FEATS` plus `water_lag1`, `electricity_usage_mwh`, and `water_usage_mgal`, with **year-based sample weights** matching the cascade DC head. The **electricity** and **water** RFs use `ELEC_FEATS` and `WATER_FEATS` from `features.py` (same inputs as the HistGradientBoosting heads) and are fit **without** those sample weights, like the cascade resource models. **Section 8** of `main.ipynb` fits all three, prints **MAE, MSE, RMSE, and R2** for each target side by side (RF vs cascade), writes comparison figures under `viz/model_compare/`, and can reload the pipeline if you run only that section in a fresh kernel.

Forecasts run at both state and national level. The rollout is autoregressive,
meaning each year's predictions feed into the next year's features. For
2021-2024, actual electricity data is injected directly since we have it --
that captures the real AI-driven demand surge those years.

## Setup

```bash
pip install -r requirements.txt
```

Data files go in `data/` at the **repository root**:
- `datacenters_clean.csv`
- `electricity_clean.csv`
- `water_clean.csv`

From the repo root, use the notebooks folder as the working directory so imports and `viz/` paths resolve:

```bash
cd notebooks
```

Then open `main.ipynb` in Jupyter or VS Code and run top to bottom.

## Project structure

```
data/                      input CSVs (repo root)
notebooks/
  main.ipynb               full pipeline; includes EDA, training, forecast, and RF vs cascade comparison
  eda/                     exploratory notebooks (optional; paths assume cwd = notebooks/)
    eda.ipynb
    national_analysis.ipynb
    electricity_visualizations.ipynb
  src/
    panel.py               loads and merges the three datasets into one panel
    features.py            builds lag features, rolling averages, momentum signals
    models.py              cascade + HistGradientBoosting, training loop, hyperparameter tuning
    treemodel.py           Random Forest baselines: DC openings, electricity, water; metrics vs cascade
    model_compare_plots.py cascade vs RF panels (model-fit style), scatters, importances (section 8)
    forecast.py            autoregressive rollout for 2021-2030
    eda_plots.py           pipeline historical charts (1990-2020)
    forecast_plots.py      prediction charts (2021-2030)
  viz/
    eda/                   PNGs from pipeline EDA (`run_all_eda` in main.ipynb)
      national/
      state/
    forecast/              PNGs from forecast plots
      national/
      state/
    model_compare/         cascade vs RF: test_fit panels, scatters, importances, national history (section 8)
```

## EDA layout

- **Pipeline EDA (canonical):** Section 2 of `notebooks/main.ipynb` calls `run_all_eda` from `eda_plots.py` and writes figures under `notebooks/viz/eda/`.
- **Exploratory notebooks:** Additional ad hoc analysis lives under `notebooks/eda/`. Those notebooks may reference other CSVs or pickles you generate locally (for example combined electricity files or merged panels); they are optional and not required to run `main.ipynb`.

## Data notes

- DC openings capped at 2020 -- that is the last year with confirmed data
- Electricity data runs to 2024 and is used as a signal feature in the rollout
- Water data covers 2000-2020 and is missing Alaska, Hawaii, and D.C.
- Commercial sector electricity is the right column to use -- that is where
  data centers show up in the EIA data, not industrial

## Model notes

The DC openings model in **`models.py`** uses a cascade because roughly 60% of state-years have
zero openings. A single regressor trained on all that data would learn to
predict near zero for everything. The cascade separates the binary question
(any DCs?) from the count question (how many?) which fixes that.

**`treemodel.py`** exposes `fit_all_random_forests()` (returns models for `dc`, `elec`, `water`), `build_full_rf_vs_cascade_metrics()` (one table with rows per target and model: MAE, MSE, RMSE, R2), and helpers for aligned test predictions. **Openings** scores for the RF use rows where all opening RF features are non-null; **electricity** and **water** scores use the same test frames as `models.py` (electricity: years through the DC cutoff; water: 2000 through cutoff). That means DC row counts can differ slightly from the cascade test slice in section 4, while resource heads align with the cascade test splits.

**`model_compare_plots.py`** mirrors the `forecast_plots` model-fit layout: for each target it saves cascade and RF panels (`test_fit_cascade_dc`, `test_fit_rf_dc`, and the `_elec` / `_water` pairs), side-by-side test scatters (`test_scatter_dc_openings`, `test_scatter_elec`, `test_scatter_water`), RF importances (`rf_feature_importance_dc`, `_elec`, `_water`), and for openings a national historical overlay (`national_history_openings_rf_vs_cascade`). Plots annotate **R2, MAE, and MSE** on the scatter panels. Matplotlib is imported only when plotting, so importing `treemodel` without matplotlib still works.

Recent years are weighted 3-4x more in training since pre-2010 data looks
nothing like the current cloud/AI market. A model fit equally on 1993 and
2019 would be pulled in the wrong direction.

Approximate R2 from the main pipeline (cascade) on the test period 2016-2020:
- DC openings: ~0.34 (count data is noisy, national totals are more reliable)
- Electricity: ~0.99
- Water: ~0.99

RF vs cascade numbers for all three heads are printed and plotted in section 8; they will not match the figures above exactly, especially for openings, because the RF uses different model families and (for DC) a different feature row filter.
