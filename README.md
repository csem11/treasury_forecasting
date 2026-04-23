# Treasury Rate Forecasting

Machine learning pipeline for forecasting U.S. Treasury yields and benchmark rates (SOFR, 1M, 1Y, 5Y, 10Y) one day and one week ahead.

Data is pulled daily from the U.S. Treasury and FRED APIs. Models are evaluated through a walk-forward backtest — no future data ever leaks into training.

See [ANALYSIS.md](ANALYSIS.md) for results and charts.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Update data

Add your FRED API key to a `.env` file:

```
export FRED_API_KEY=your_key_here
```

Then run:

```bash
python -m treasury_forecasting
```

This fetches the latest Treasury yield curve and FRED macro data and saves them to `data/`.

## Notebooks

| Notebook | Contents |
|---|---|
| `01_rate_environment.ipynb` | Rate cycle overview, yield curve inversion |
| `02_feature_engineering.ipynb` | Feature construction and correlation analysis |
| `03_model_exploration.ipynb` | 7-model comparison, hyperparameter tuning |
| `04_ensemble_and_results.ipynb` | Ensemble construction, full validation, regime analysis |

## Project layout

```
treasury_forecasting/
├── src/treasury_forecasting/
│   ├── data_ingest.py       # Treasury yield curve ETL
│   ├── fred_data.py         # FRED macro data ETL
│   └── validation.py        # Data quality checks
├── notebooks/
├── data/                    # Raw and engineered data CSVs
├── images/                  # Charts exported from notebooks
└── ANALYSIS.md
```
