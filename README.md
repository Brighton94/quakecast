# quakecast

Earthquake physics + ML: prospective earthquake forecasting for California (and Italy, Japan, New Zealand), scored the way CSEP scores models. The goal is a physics-informed ML model that beats ETAS on forecasts committed before the outcome is known.

## Why ETAS is the bar

The Epidemic-Type Aftershock Sequence (ETAS) model is a statistical model built on two empirical laws of earthquake physics: the Omori–Utsu decay of aftershock rates in time, and Gutenberg–Richter magnitude scaling, with every event able to trigger its own aftershocks. It has a handful of parameters and has been the standard short-term forecasting model for decades.

**Almost no ML model beats it.** Deep-learning and neural point-process forecasters regularly look good in papers, but when they are compared with ETAS on the same catalogs, windows and likelihood scores, they mostly match it or fall short. The EarthquakeNPP benchmark (Stockman, Lawson and Werner, 2024) tested five neural point processes against ETAS on its benchmark catalogs and found none outperformed ETAS. The few reported wins are retrospective, on a single region, or not scored with CSEP's tests.

That is why quakecast keeps ETAS as the baseline, puts the physics into the ML model rather than replacing it, and only counts a result if it shows positive information gain over ETAS on forecasts committed before the outcome.

## Quick start

Open the folder in VS Code and choose **Reopen in Container**. The container installs everything with `uv sync`. Then:

```bash
uv run quakecast fetch --start 2018-01-01 --min-mag 3.0
uv run quakecast forecast data/california_comcat.parquet --days 7
uv run quakecast verify commitments/<file>.json
uv run quakecast score commitments/<file>.json data/california_comcat.parquet
```

For exploratory analysis, fetch every ComCat column (ids, magnitude type, location quality, event type) and print the headline statistics:

```bash
uv run quakecast fetch --full --start 1985-01-01 --min-mag 2.5
uv run quakecast eda data/california_comcat_full.parquet --m0 3.0
```

`notebooks/02_eda.ipynb` is the full analysis: column descriptions, missing values, outliers, skewness, completeness and b-value, overdispersion, heteroskedasticity, seasonality, Omori decay and feature-target correlations. Its main finding is that declustered seismicity is Poisson with no seasonality, so the forecastable signal is the triggered (aftershock) part that ETAS already models.

`score` refuses to run until the catalog covers the whole window, so refetch after the window closes.

## Workflow for a verifiable result

1. Fit and commit a forecast before the window opens. The JSON in `commitments/` holds the forecast and its SHA-256 hash.
2. Commit that file to git and push, and post the hash publicly (LinkedIn, a tweet). Either gives you an independent timestamp.
3. After the window closes, refetch ComCat and run `score`. Anyone can rerun `verify` to confirm the forecast was not edited.
4. Report the N-test, Poisson log score and Brier score, and the information gain of your model over ETAS on the same windows.

## Layout

| Path | Purpose |
| --- | --- |
| `src/quakecast/catalog.py` | ComCat download (pyCSEP, or full FDSN columns), column descriptions, regions, parquet I/O |
| `src/quakecast/etas.py` | Temporal ETAS likelihood, MLE fit, b-value, forecasts |
| `src/quakecast/evaluate.py` | N-test, Poisson log score, Brier score, information gain |
| `src/quakecast/commit.py` | Hash commitments and verification |
| `src/quakecast/eda.py` | EDA helpers: data quality, Mc and b-value, declustering, Omori fits, stationarity, heteroskedasticity, seasonality |
| `src/quakecast/plots.py` | Shared plot style and EDA plots |
| `scripts/build_eda_notebook.py` | Generates `notebooks/02_eda.ipynb` |
| `src/quakecast/cli.py` | `quakecast` command |
| `notebooks/` | Executed exploration notebooks |
| `commitments/` | Tracked in git; `data/` and `forecasts/` are not |

## Baseline notes

ETAS v0 is temporal only and its forecast counts triggering from events before the window, not cascades inside it, so it underestimates rates right after a large shock. Next steps are a spatio-temporal ETAS with gridded output in pyCSEP format, then the ML model on top.

The CSEP ten-year California next-day forecast benchmark is on Zenodo; score against those models with pyCSEP's gridded tests once forecasts are gridded.
