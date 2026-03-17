# Spirit Airlines FLL Hub — OTP Prediction & Capacity Scenario Engine

> **Freelance Data Science Contract** — US Ultra-Low-Cost Carrier, Hub Operations Analytics
> Period: 2023 Q3 – 2024 Q1 · Data: 2022–2024 (synthetic, BTS T-100 statistical properties)

---

## Business Problem

The client's Fort Lauderdale hub operated 10 routes with degrading on-time performance and no
quantitative framework for evaluating schedule change decisions. Three specific gaps:

1. **No delay prediction** — operations team couldn't anticipate high-risk flights before departure
2. **No capacity scenario tooling** — schedule additions were evaluated by intuition, not simulation
3. **No demand visibility** — forward passenger volumes were unavailable for network planning

---

## Solution Architecture

```text
Raw Data (CSV)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  ETL Pipeline  (pandas → DuckDB)                            │
│  5 analytical views · schema validation · rotation chains   │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
  ┌──────────────┐  ┌────────────┐  ┌─────────────────┐
  │ XGBoost OTP  │  │  SARIMAX   │  │  Monte Carlo    │
  │  Predictor   │  │  Demand    │  │  Scenario Sim   │
  │ AUC 0.817    │  │ Forecaster │  │  10k runs       │
  │ 43 features  │  │ 12-month   │  │  3 strategies   │
  └──────┬───────┘  └─────┬──────┘  └───────┬─────────┘
         │                │                  │
         └────────────────┼──────────────────┘
                          ▼
              ┌───────────────────────┐
              │   Streamlit Dashboard │
              │   5 pages · dark UI   │
              │   GPT-4o-mini NL→SQL  │
              │   Live OWM weather    │
              └───────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              │     MLOps Layer       │
              │  MLflow · GitHub CI   │
              └───────────────────────┘
```

---

## Modules

### `src/pipeline/`

- **`etl.py`** — ingest CSVs, clean, merge, load into DuckDB; creates 5 analytical views
  (`route_performance`, `capacity_utilization`, `delay_cause_decomposition`, `weather_impact`, `monthly_otp_summary`)

- **`features.py`** — 43-feature engineering pipeline including 6 rotation-chain features:
  `legs_today_before`, `cumulative_tail_delay_today`, `prev_tail_arr_delay`,
  `hours_since_last_flight`, `quick_turn_flag`, `tail_delay_streak`

### `src/models/`

- **`otp_predictor.py`** — XGBoost binary classifier (on-time vs delayed 15+ min)
  - Platt-scaling calibration · stratified k-fold CV · gain-based feature importance
  - AUC-ROC **0.817** · F1 **0.552** · Average Precision **0.682**

- **`scenario_simulator.py`** — Monte Carlo capacity simulator
  - Frequency-demand elasticity model (ULCC economics)
  - RASK/CASM revenue P&L · 3 schedule strategies: `all_day`, `peak_only`, `off_peak`
  - Outputs: LF distribution, OTP distribution, annual revenue delta with P10/P90 bands

- **`demand_forecaster.py`** — SARIMAX per-route monthly forecaster
  - 12-month forward forecasts · 95% confidence intervals · stored as Parquet

### `src/dashboard/`

- **`app.py`** — 5-page Streamlit dashboard (dark theme, production-grade UI)
  - Overview: network KPIs + OTP trend + model status strip
  - Route Performance: OTP heatmap · delay cause decomposition · comparison table
  - OTP Predictor: real-time prediction form with live weather fetch
  - Scenario Simulator: Monte Carlo histograms + comparison table + recommendations
  - Ask Analytics: GPT-4o-mini NL→SQL chat with pattern-matching fallback

- **`llm_query.py`** — OpenAI GPT-4o-mini conversational analytics (NL → SQL → DuckDB)

- **`powerbi_export.py`** — 7-sheet Power BI-ready Excel workbook generator

### `src/mlops/`

- **`tracking.py`** — MLflow experiment tracker (SQLite backend, no server required)
  - Auto-logs: hyperparameters · AUC-ROC · F1 · feature count · model artifact
  - `get_best_run()` queries registry for champion model

### `src/integrations/`

- **`weather_api.py`** — OpenWeatherMap client for 11 airport locations
  - `WeatherSnapshot` dataclass · severity scoring (0–10) · graceful fallback

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data storage | DuckDB (analytical), Parquet (processed) |
| ML — classification | XGBoost 2.x · scikit-learn · Platt calibration |
| ML — forecasting | statsmodels SARIMAX |
| Simulation | NumPy Monte Carlo (10,000 runs) |
| Dashboard | Streamlit · Plotly |
| LLM analytics | OpenAI GPT-4o-mini |
| Live weather | OpenWeatherMap API |
| Experiment tracking | MLflow 3.x (SQLite) |
| CI/CD | GitHub Actions (weekly scheduled retrain) |
| Export | openpyxl (Power BI Excel) |
| Language | Python 3.11 |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# → Add OPENAI_API_KEY and OPENWEATHERMAP_API_KEY

# 3. Run full pipeline
python main.py generate-data
python main.py run-etl
python main.py train-models

# 4. Launch dashboard
python main.py launch-dashboard
# → http://localhost:8501

# 5. MLOps status
python main.py mlops-status

# 6. View experiment runs
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
# → http://localhost:5000
```

---

## CLI Reference

```text
python main.py generate-data       Generate synthetic flight / capacity / weather data
python main.py run-etl             ETL → DuckDB (5 analytical views)
python main.py train-models        XGBoost + SARIMAX + MLflow logging
python main.py run-simulator       Monte Carlo scenario (see --help for options)
python main.py mlops-status        Print model registry, MLflow runs, data freshness
python main.py launch-dashboard    Streamlit dashboard
```

Simulator options:

```bash
python main.py run-simulator --route FLL-ATL --flights 1 --schedule peak_only --runs 10000
```

---

## MLOps Pipeline

```text
GitHub Actions (every Monday 03:00 UTC)
  │
  ├── validate-data     schema check · null rate · route coverage
  ├── train-models      XGBoost + MLflow logging
  ├── quality-gate      AUC-ROC ≥ 0.75 · F1 ≥ 0.40  (fail → pipeline stops)
  └── regression-test   storm flight P(delay) > clear-day P(delay)
```

Artifacts retained: trained model (30 days) · MLflow DB (90 days)

---

## Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.817 |
| F1 Score | 0.552 |
| Average Precision | 0.682 |
| Features | 43 (incl. 6 rotation-chain) |
| Training data | 2022–2024 · 10 FLL routes |
| Positive rate | ~27% (delayed 15+ min) |

Top rotation-chain features:

- `prev_tail_arr_delay` — previous leg arrival delay on the same aircraft
- `cumulative_tail_delay_today` — total delay accrued by tail number within the day
- `quick_turn_flag` — binary, turnaround < 60 minutes
- `tail_delay_streak` — consecutive delayed arrivals on same tail

---

## Scenario Simulator — Sample Output (FLL-ATL +1 daily flight)

| Strategy | LF Change | OTP Change | Revenue Delta |
|---|---|---|---|
| All-Day | −1.7 pp | −0.3 pp | −$0.6M |
| Peak Only | +0.6 pp | −0.1 pp | +$0.1M |
| Off-Peak | −4.8 pp | −0.5 pp | −$1.4M |

**Recommendation:** Peak-only addition is the only revenue-positive strategy; off-peak creates significant demand dilution relative to incremental ASMs.

---

## Project Structure

```text
.
├── main.py                         CLI entry point
├── requirements.txt
├── .env                            API keys (not committed)
├── .github/
│   └── workflows/
│       └── retrain_pipeline.yml    Weekly CI/CD pipeline
├── src/
│   ├── pipeline/
│   │   ├── etl.py
│   │   └── features.py
│   ├── models/
│   │   ├── otp_predictor.py
│   │   ├── scenario_simulator.py
│   │   └── demand_forecaster.py
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── llm_query.py
│   │   └── powerbi_export.py
│   ├── mlops/
│   │   └── tracking.py
│   ├── integrations/
│   │   └── weather_api.py
│   └── sql/
│       └── create_tables.sql
├── data/
│   ├── raw/                        Generated synthetic data
│   ├── processed/                  Parquet files
│   └── models/                     Trained model + forecasts
├── mlruns/
│   └── mlflow.db                   MLflow experiment registry
├── exports/
│   └── spirit_fll_powerbi.xlsx     Power BI workbook
├── tests/
└── notebooks/
```

---

## Environment Variables

```env
OPENAI_API_KEY=sk-...              GPT-4o-mini NL→SQL analytics
OPENWEATHERMAP_API_KEY=...         Live FLL weather in OTP Predictor
```

Both integrations degrade gracefully if keys are absent — dashboard remains fully functional
with manual inputs and pattern-matching SQL fallback.

---

*Data is synthetic, generated to match BTS T-100 statistical properties for Spirit Airlines FLL hub routes (2022–2024).*
