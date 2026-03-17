# Technical Methodology

## Spirit Airlines FLL Hub: OTP Prediction & Demand-Capacity Scenario Engine

---

## 1. Data Sources and Processing

### 1.1 Data Sources

This project uses **synthetic data** generated to mimic the structure and statistical properties of real-world airline operations data:

| Dataset | Real Equivalent | Rows | Key Fields |
|---------|----------------|------|-----------|
| Flights | BTS Form 41 / On-Time Performance | ~500,000 | FlightDate, Route, DepDelay, ArrDelay, ArrDel15, delay causes |
| Capacity | BTS T-100 Domestic Segment | ~360 | Monthly Passengers, Seats, LoadFactor per route |
| Weather | ASOS / METAR (Iowa State Mesonet) | ~26,000 | Hourly temperature, wind, precipitation, visibility, sky cover |

### 1.2 Synthetic Data Realism

The synthetic generator (`data/scripts/generate_synthetic_data.py`) produces statistically realistic data by:

**Seasonal patterns:**
- Delay multipliers calibrated to known seasonal patterns: summer thunderstorm season (June–September) adds 30% more delay probability; December holiday congestion adds 25%.
- Load factor seasonal adjustments: +6pp in summer peak, +5pp in December, -4pp in September.

**Time-of-day delay propagation:**
- Departures after 15:00 face 35–45% higher delay rates due to accumulated operational disruptions throughout the day (the "delay snowball" effect).
- Early morning departures (05:00–07:00) have 30–40% lower delay rates as they begin fresh rotations.

**Route-specific characteristics:**
- OTP baselines reflect published Spirit performance: FLL-JFK (68%) is the most constrained due to NYC airspace; FLL-MIA (80%) benefits from short sector and high rotation frequency.
- Load factors range from 78% (FLL-MIA) to 91% (FLL-LAS), reflecting leisure route demand elasticity.

**Delay propagation modeling:**
- Carrier delays are correlated with route OTP baseline.
- Late aircraft delays are correlated with time-of-day and seasonal factors.
- Cancellations triggered when total delay > 180 min or weather delay > 60 min, with probabilities 3–8%.

### 1.3 ETL Pipeline

The ETL pipeline (`src/pipeline/etl.py`) performs:

1. **Loading:** Read raw CSVs with appropriate dtypes and parse dates.
2. **Cleaning:**
   - Remove records with missing FlightDate or Route.
   - Clamp delay values to [-30, 600] minutes to remove outliers.
   - Coerce numeric types and fill logical nulls.
3. **Feature derivation:** Year, Month, DayOfWeek, Quarter, DepHour from FlightDate and CRSDepTime.
4. **Weather join:** Left join flights with hourly FLL weather by matching on (FlightDate, DepHour), resulting in a matched weather observation for each departure.
5. **Output:** Processed Parquet files and DuckDB registration.

---

## 2. Feature Engineering Decisions

### 2.1 Time Features

| Feature | Rationale |
|---------|-----------|
| `hour_of_day` | Captures delay propagation: later hours = more accumulated delays |
| `hour_sin`, `hour_cos` | Cyclical encoding preserves 23:00–00:00 continuity |
| `is_holiday`, `is_holiday_window` | Holiday windows (±1 day) drive demand spikes and congestion |
| `is_summer`, `is_hurricane_season`, `is_winter` | Regime indicators for weather and demand seasonality |

### 2.2 Route Features

| Feature | Rationale |
|---------|-----------|
| `route_otp_baseline` | Encodes structural route difficulty (airspace, connecting hub congestion) |
| `route_distance` | Longer routes have more recovery opportunity but also more exposure |
| `is_high_congestion_dest` | ATL, ORD, JFK, DFW, LAX add systemic arrival delay risk |

### 2.3 Weather Features

The composite `weather_severity` score combines:
- Thunderstorm presence (weight: 4.0) — most operationally disruptive
- Low visibility (<3 miles, weight: 2.5) — directly affects landing
- Wind gusts (>25 kt, weight: 2.0) — crosswind constraints
- Precipitation (weight: 1.5) — de-icing, slower operations

### 2.4 Propagation Features

`rolling_avg_dep_delay`: 2-hour rolling average of departure delays at FLL captures real-time operational state. When the airport is already running behind, future departures inherit that disruption. This is the most operationally intuitive feature and frequently appears as top-3 in importance.

`prev_tail_dep_delay`: The previous departure delay for the same aircraft tail number captures the inbound delay of the last rotation. This is particularly important for Spirit's dense hub operations where a single late aircraft can cascade through 3–4 rotations.

---

## 3. Model Selection Rationale

### 3.1 Classification Model: XGBoost

**Why XGBoost over alternatives:**

| Model | Considered | Decision |
|-------|-----------|----------|
| Logistic Regression | Yes | Baseline — too linear for interaction effects |
| Random Forest | Yes | Good AUC but slower inference, less calibrated |
| LightGBM | Yes | Similar performance to XGBoost; XGBoost chosen for wider production adoption |
| XGBoost | **Selected** | Best AUC-ROC, fast inference, built-in L1/L2 regularization |
| Neural Network | No | Overkill for tabular data; less interpretable |

**Key XGBoost hyperparameters:**
- `max_depth=6`: Allows capture of higher-order interactions (weather × route × time) without overfitting.
- `scale_pos_weight=1.5`: Adjusts for class imbalance (delayed flights ~27% of operated flights).
- `subsample=0.8`, `colsample_bytree=0.8`: Reduces variance through stochastic sampling.

**Probability calibration:** Platt scaling (logistic regression on model outputs) is applied to ensure predicted probabilities are well-calibrated, which is critical for threshold-based operational decisions (e.g., "alert if >40% delay probability").

### 3.2 Demand Forecasting: SARIMAX

SARIMAX(1,1,1)(1,1,1)₁₂ was chosen for monthly passenger demand forecasting because:
- Captures both short-term autocorrelation (AR/MA terms) and seasonal cycles (12-month period).
- External regressors (fuel price, economic index) allow scenario conditioning.
- Provides uncertainty intervals natively, enabling risk-adjusted planning.

**Fallback:** Holt-Winters ETS is used if SARIMAX does not converge (typically for routes with fewer than 24 months of data).

---

## 4. Monte Carlo Simulation Design

### 4.1 Simulation Architecture

The Monte Carlo simulator (`src/models/scenario_simulator.py`) models capacity scenario uncertainty through five stochastic input layers:

1. **Demand capture rate** (Beta distribution): How many passengers fill the new capacity vs. flying empty. Peak-only scheduling achieves higher capture (mean 1.22) due to concentrated demand, off-peak achieves lower (mean 0.72).

2. **Frequency elasticity** (deterministic + noise): Based on published elasticity research, a 10% frequency increase stimulates ~3% incremental demand. Uncertainty modelled as ±3% normal noise.

3. **Load factor simulation**: Derived from (original_pax × demand_stimulus × capture) / new_seats, with ±1.5% normal noise for operational variance.

4. **OTP degradation model**:
   - Each percentage point of LF above 85% reduces OTP by 0.15pp (congestion sensitivity).
   - Each additional daily departure reduces OTP by ~0.8pp (operational complexity).
   - Peak-only scheduling adds ~0.75pp extra OTP drag (peak hour congestion multiplier).

5. **Yield uncertainty** (Normal distribution): ±6% yield multiplier accounts for pricing response to new capacity.

### 4.2 Output Metrics

Each simulation run produces:
- Load factor (range: 30%–98%)
- OTP rate (range: 40%–99%)
- Annual revenue delta vs. baseline (includes both revenue gain and incremental operating cost)

### 4.3 Confidence Intervals

The simulation returns P10/P90 ranges representing the 10th and 90th percentile outcomes across all Monte Carlo runs. A scenario is considered **commercially attractive** if:
- P10 revenue delta > 0 (profitable even in adverse scenarios)
- Mean OTP delta > -3pp (manageable operational degradation)

---

## 5. Limitations and Assumptions

### 5.1 Data Limitations

- **Synthetic data**: All analysis is based on generated data. Real BTS/ASQP data would incorporate actual mechanical delay patterns, actual weather events (named storms), and precise flight-level operational data.
- **Single origin**: Only FLL-origin routes are modelled. In reality, Spirit operates FLL as an inbound hub too, and delays on feeder routes affect FLL operations.
- **No connecting passenger modelling**: Demand model assumes point-to-point passengers only; connecting itineraries through Spirit's network are not captured.

### 5.2 Model Assumptions

- OTP degradation with load factor is assumed **linear** above 85%. In practice, the relationship may be non-linear (step changes at operational tipping points).
- Demand elasticity is assumed constant across seasons. Summer demand may be significantly less elastic than shoulder season.
- The simulator assumes no competitive response (other carriers do not adjust schedules in response to Spirit's capacity changes).
- Aircraft operating costs are modelled as CASM × ASM (constant per seat-mile). Variable costs (fuel hedging, crew overtime) are not separately modelled.

### 5.3 Forecast Limitations

- SARIMAX assumes additive seasonality and linear trends. Structural breaks (e.g., post-COVID demand normalization, fare wars) are not captured.
- External regressors (fuel price, economic index) are synthetic proxies generated as random walks and do not reflect actual market conditions.
- Forecast accuracy degrades beyond 6 months due to cumulative uncertainty in external regressors.
