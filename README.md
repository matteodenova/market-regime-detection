# Market Regime Detection & Volatility Analytics Platform

An end-to-end quantitative pipeline that identifies hidden market regimes (calm vs.
stressed) and models volatility dynamics across a multi-asset basket — equity, energy,
FX, and volatility indices — from 2004 to 2025.

## Motivation

Markets move through distinct regimes — periods of calm and periods of acute stress —
and recognizing these regimes early matters for risk management, pricing, and capital
allocation, both in equity and commodity/energy markets. This project builds, end to
end and on real data, a system that identifies these regimes and translates them into
actionable risk metrics: from raw data ingestion to model output.

It combines two parts of my background: a data engineering foundation (scalable,
production-style pipelines) with a statistics-driven approach to time series and
quantitative modeling — the direction I'm moving toward professionally.

## Project status

 **Work in progress.** Ingestion, feature engineering, EDA, and regime detection
(HMM) are complete. Volatility forecasting, risk metrics, and the dashboard layer are
in development. 

## Data

| Asset class | Tickers | Source |
|---|---|---|
| Equity | `^GSPC` (S&P 500), `FEZ` (Euro Stoxx 50) | Yahoo Finance |
| Energy | `CL=F` (WTI Crude), `NG=F` (Natural Gas), Brent | Yahoo Finance, EIA |
| FX | `EURUSD=X`, `USDCHF=X` | Yahoo Finance |
| Volatility | `^VIX` | Yahoo Finance |

Daily data from 2004-01-05 to 2025-12-31 (~5,450 trading days).

## Architecture

The pipeline follows this pattern:

```
data/
├── raw/          # bronze — raw prices from yfinance + EIA, as downloaded
└── processed/     # gold — engineered features, ready for modeling

source/
├── 01_ingest.py       # AssetDataIngestion: downloads & combines raw price data
├── 02_features.py     # FeatureEngineer: log returns, rolling volatility/momentum,
│                       # drawdown, SPX/VIX rolling correlation, rule-based regime flags
└── 03_regime_detection.py   # (in progress) HMM + change point detection, packaged
                              # as a reusable RegimeDetector class

notebooks/
├── 01_EDA.md                    # exploratory data analysis (static preview)
├── 02_hmm_exploration.md        # HMM regime detection — model exploration
└── *_files/                     # exported chart images for GitHub rendering
```

Each `.md` file is a GitHub-renderable export of the corresponding `.ipynb` notebook
(kept locally, not versioned, to avoid bloating the repo with large interactive Plotly
outputs). Static chart images are extracted alongside.

## Methods

**Feature engineering** — daily log returns, rolling volatility (21d / 63d) and
realized volatility, rolling momentum, drawdown from peak, 63-day rolling SPX/VIX
correlation, and rule-based regime flags (`high_vol_regime`, `bear_market_flag`) as an
interpretable baseline.

**Exploratory analysis** — distributional diagnostics on log returns (Jarque-Bera test,
QQ-plots), stationarity checks (Augmented Dickey-Fuller on price levels vs. returns),
and formal tests for volatility clustering (ACF on squared returns, Ljung-Box, Engle's
ARCH test) — the empirical justification for moving to GARCH-family models in the next
phase.

**Regime detection** — a Gaussian Hidden Markov Model (`hmmlearn`) fit on S&P 500 log
returns, learning both the emission distribution of each hidden state and the
transition dynamics between them, without arbitrary thresholds. Model order (2 vs. 3
vs. 4+ states) is selected via AIC/BIC, weighed against interpretability. Decoded
regimes are validated against known stress periods (2008 GFC, COVID-19 2020, the 2022
energy crisis) and cross-checked against the rule-based flags from feature engineering.

*(Change point detection as a complementary method is in progress.)*

## Tech stack

Python · pandas · PySpark · NumPy · SciPy · statsmodels · hmmlearn · `arch` (GARCH,
planned) · scikit-learn / XGBoost (planned) · Plotly · DuckDB (planned, local SQL
prototyping) · MLflow (planned, experiment tracking) 



## Reproducing this project

```
git clone https://github.com/matteodenova/market-regime-detection.git
cd market-regime-detection
pip install -r requirements.txt
python source/01_ingest.py      # downloads raw price data
python source/02_features.py    # builds the feature set
```

Notebooks under `notebooks/` (local `.ipynb` files, not tracked in git) can be run
against the generated parquet file for exploratory analysis and model fitting.

## About

Built by [Matteo Denova](https://www.linkedin.com/in/matteo-denova) — Data Scientist
with a background in Statistics, currently focused on quantitative and energy
analytics. This project is part of a broader effort to apply statistical modeling to
real market data, bridging production-grade data engineering with quantitative
analysis.
