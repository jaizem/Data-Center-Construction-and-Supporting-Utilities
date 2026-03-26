# U.S. Data Center Growth and Resource Demand Forecast

Predicts how many data centers will be built across U.S. states through 2030,
and what that means for commercial electricity and water demand.

## What it does

Trains three ML models on 1990-2020 data and forecasts through 2030:

- **DC openings** -- two-stage cascade model in `models.py` (will any open? then how many?)
- **Electricity** -- gradient boosting regression driven by projected DC count
- **Water** -- same approach as electricity

A separate **Random Forest** baseline for openings lives in `treemodel.py`: one regressor that uses **DC features plus explicit commercial electricity and water usage** (levels and `water_lag1`, in addition to the elec signals already inside `DC_FEATS`). **Section 8** of `main.ipynb` compares it to the cascade, saves diagnostic PNGs under `viz/model_compare/`, and can reload the pipeline if you run only that section in a fresh kernel.

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
    treemodel.py           Random Forest openings baseline (elec + water augmented features)
    model_compare_plots.py figures for cascade vs RF (section 8)
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
    model_compare/         cascade vs RF: test fits, national history, importances (section 8)
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

**`treemodel.py`** implements a **Random Forest** on an expanded feature set (see `openings_rf_feature_columns()`): everything in `DC_FEATS` plus `water_lag1`, `electricity_usage_mwh`, and `water_usage_mgal`. Test metrics use **aligned** rows where all of those inputs are non-null, so counts can differ slightly from the cascade-only test slice in section 4. Metrics, scatter/residual panels (like `forecast_plots.model_fit`), national time-series overlays, and RF feature importances are written to **`viz/model_compare/`** when you run section 8.

Recent years are weighted 3-4x more in training since pre-2010 data looks
nothing like the current cloud/AI market. A model fit equally on 1993 and
2019 would be pulled in the wrong direction.

R2 on the test period 2016-2020:
- DC openings: ~0.34 (count data is noisy, national totals are more reliable)
- Electricity: ~0.99
- Water: ~0.99
