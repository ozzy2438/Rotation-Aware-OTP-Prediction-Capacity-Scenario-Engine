"""
Power BI Export Module — Spirit Airlines FLL Hub Analytics
==========================================================

Generates a multi-sheet Excel workbook ready for import into Power BI Desktop.

Each sheet is a clean, flat fact or dimension table with descriptive headers
and no merged cells — matching Power BI's preferred data model structure.

Sheets produced
---------------
1. otp_monthly        — Monthly OTP, cancel rate, avg delay by route
2. delay_causes       — Delay-cause breakdown (carrier / weather / NAS / late-ac)
3. load_factor        — Monthly load factor and seat utilisation by route
4. weather_impact     — Weather severity vs OTP correlation by route / month
5. scenario_summary   — Pre-computed Monte Carlo scenario comparison (FLL-ATL)
6. demand_forecast    — 12-month demand forecast for all FLL hub routes
7. metadata           — Data dictionary, date of export, model performance

Usage
-----
    python src/dashboard/powerbi_export.py
    # → exports/spirit_fll_powerbi.xlsx

From Streamlit dashboard the export button calls ``generate_workbook()``.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
import sys
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH      = PROJECT_ROOT / "data" / "spirit_otp.duckdb"
EXPORT_DIR   = PROJECT_ROOT / "exports"

# Ensure project root is on sys.path when running as a standalone script
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _query(sql: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Execute a read-only DuckDB query and return a DataFrame."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Individual sheet builders
# ---------------------------------------------------------------------------

def _sheet_otp_monthly() -> pd.DataFrame:
    return _query("""
        SELECT
            Route                                            AS "Route",
            Year                                             AS "Year",
            Month                                            AS "Month",
            PRINTF('%d-%02d', Year, Month)                   AS "Year-Month",
            total_flights                                    AS "Total Flights",
            operated_flights                                 AS "Operated Flights",
            cancelled_flights                                AS "Cancelled Flights",
            ROUND(cancel_rate * 100, 2)                      AS "Cancel Rate %",
            delayed_flights                                  AS "Delayed Flights (15+ min)",
            ROUND(otp_rate * 100, 2)                         AS "OTP %",
            ROUND(avg_arr_delay_min, 1)                      AS "Avg Arrival Delay (min)",
            ROUND(avg_dep_delay_min, 1)                      AS "Avg Departure Delay (min)"
        FROM route_performance
        ORDER BY Route, Year, Month
    """)


def _sheet_delay_causes() -> pd.DataFrame:
    return _query("""
        SELECT
            Route                                            AS "Route",
            Year                                             AS "Year",
            Month                                            AS "Month",
            PRINTF('%d-%02d', Year, Month)                   AS "Year-Month",
            delayed_count                                    AS "Delayed Flights",
            ROUND(avg_carrier_delay, 1)                      AS "Avg Carrier Delay (min)",
            ROUND(avg_weather_delay, 1)                      AS "Avg Weather Delay (min)",
            ROUND(avg_nas_delay, 1)                          AS "Avg NAS Delay (min)",
            ROUND(avg_late_ac_delay, 1)                      AS "Avg Late Aircraft Delay (min)",
            ROUND(avg_security_delay, 1)                     AS "Avg Security Delay (min)",
            ROUND(carrier_delay_pct * 100, 1)                AS "Carrier Delay %",
            ROUND(weather_delay_pct * 100, 1)                AS "Weather Delay %",
            ROUND(nas_delay_pct * 100, 1)                    AS "NAS Delay %",
            ROUND(late_ac_delay_pct * 100, 1)                AS "Late Aircraft Delay %"
        FROM delay_cause_decomposition
        ORDER BY Route, Year, Month
    """)


def _sheet_load_factor() -> pd.DataFrame:
    return _query("""
        SELECT
            cu.Route                                         AS "Route",
            cu.Year                                          AS "Year",
            cu.Month                                         AS "Month",
            cu.YearMonth                                     AS "Year-Month",
            cu.DepScheduled                                  AS "Scheduled Departures",
            cu.DepPerformed                                  AS "Performed Departures",
            cu.Seats                                         AS "Available Seats",
            cu.Passengers                                    AS "Passengers",
            ROUND(cu.LoadFactor * 100, 2)                    AS "Load Factor %",
            ROUND(cu.otp_rate * 100, 2)                      AS "OTP %",
            ROUND(cu.avg_arr_delay_min, 1)                   AS "Avg Arrival Delay (min)"
        FROM capacity_utilization cu
        WHERE cu.LoadFactor IS NOT NULL
        ORDER BY cu.Route, cu.Year, cu.Month
    """)


def _sheet_weather_impact() -> pd.DataFrame:
    return _query("""
        SELECT
            Route                                            AS "Route",
            Year                                             AS "Year",
            Month                                            AS "Month",
            PRINTF('%d-%02d', Year, Month)                   AS "Year-Month",
            total_flights                                    AS "Total Flights",
            ROUND(avg_weather_severity, 2)                   AS "Avg Weather Severity (0-10)",
            thunderstorm_count                               AS "Thunderstorm Days",
            precipitation_count                              AS "Precipitation Days",
            low_vis_count                                    AS "Low Visibility Days",
            wind_gust_count                                  AS "High Wind Days",
            ROUND(otp_rate * 100, 2)                         AS "OTP %",
            ROUND(avg_arr_delay, 1)                          AS "Avg Arrival Delay (min)"
        FROM weather_impact
        ORDER BY Route, Year, Month
    """)


def _sheet_scenario_summary() -> pd.DataFrame:
    """Run the Monte Carlo simulator for key routes and return comparison table."""
    try:
        from src.models.scenario_simulator import ScenarioSimulator, ROUTE_CONFIG
        simulator = ScenarioSimulator(rng_seed=42)

        rows = []
        for route in list(ROUTE_CONFIG.keys())[:5]:   # top-5 routes by pax volume
            comparison, _ = simulator.compare_scenarios(
                route=route,
                additional_daily_flights=1,
                simulation_runs=5_000,
            )
            comparison.insert(0, "Route", route)
            rows.append(comparison)

        df = pd.concat(rows, ignore_index=True)
        # Rename to Power BI-friendly headers
        df = df.rename(columns={
            "Scenario":                  "Schedule Strategy",
            "Daily_Flights_Before":      "Flights Before",
            "Daily_Flights_After":       "Flights After",
            "Baseline_LF_%":             "Baseline LF %",
            "Projected_LF_%":            "Projected LF %",
            "LF_Change_pp":              "LF Change (pp)",
            "LF_P10_%":                  "LF P10 %",
            "LF_P90_%":                  "LF P90 %",
            "Baseline_OTP_%":            "Baseline OTP %",
            "Projected_OTP_%":           "Projected OTP %",
            "OTP_Change_pp":             "OTP Change (pp)",
            "Annual_Revenue_Delta_M$":   "Annual Revenue Delta ($M)",
            "Rev_P10_M$":                "Revenue P10 ($M)",
            "Rev_P90_M$":                "Revenue P90 ($M)",
            "Recommendation":            "Recommendation",
        })
        return df
    except Exception as exc:
        logger.warning("Scenario sheet skipped: %s", exc)
        return pd.DataFrame({"Note": ["Run ETL + train-models first to enable scenario sheet."]})


def _sheet_demand_forecast() -> pd.DataFrame:
    """Load pre-computed demand forecasts if available, else run lightweight forecast."""
    forecast_path = PROJECT_ROOT / "data" / "models" / "demand_forecasts.parquet"
    if forecast_path.exists():
        df = pd.read_parquet(forecast_path)
        return df.rename(columns={
            "Route":                "Route",
            "forecast_month":       "Forecast Month",
            "passengers_forecast":  "Forecast Passengers",
            "passengers_lower":     "Lower Bound (95% CI)",
            "passengers_upper":     "Upper Bound (95% CI)",
            "seats_forecast":       "Forecast Seats",
            "load_factor_forecast": "Forecast Load Factor",
        })
    return pd.DataFrame({"Note": ["Run 'python main.py train-models' to generate demand forecasts."]})


def _sheet_metadata() -> pd.DataFrame:
    rows = [
        ("Export Date",        str(date.today())),
        ("Data Range",         "2022-01-01 to 2024-12-31"),
        ("Airline",            "Spirit Airlines (NK)"),
        ("Hub",                "Fort Lauderdale (FLL)"),
        ("Routes",             "FLL-ATL, FLL-LAS, FLL-LAX, FLL-ORD, FLL-DFW, FLL-MCO, FLL-JFK, FLL-BOS, FLL-DTW, FLL-MIA"),
        ("OTP Model",          "XGBoost binary classifier — AUC-ROC 0.817 (43 rotation-aware features)"),
        ("Demand Model",       "SARIMAX per-route monthly forecaster"),
        ("Scenario Model",     "Monte Carlo simulation (10,000 runs per scenario)"),
        ("LLM Layer",          "GPT-4o-mini NL→SQL analytics"),
        ("Source",             "Synthetic data generated to match BTS T-100 statistical properties"),
        ("otp_monthly",        "Monthly OTP, delay, and cancellation rates by route"),
        ("delay_causes",       "Delay-cause decomposition: carrier / weather / NAS / late-aircraft / security"),
        ("load_factor",        "Monthly load factor and seat utilisation by route"),
        ("weather_impact",     "Weather severity correlation with OTP and average delay"),
        ("scenario_summary",   "+1 daily flight Monte Carlo results across schedule strategies"),
        ("demand_forecast",    "12-month forward passenger demand forecasts with confidence intervals"),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def generate_workbook(
    output_path: Optional[Path] = None,
    db_path: Path = DB_PATH,
) -> Path:
    """Generate the Power BI-ready Excel workbook.

    Args:
        output_path: Destination .xlsx file. Defaults to exports/spirit_fll_powerbi.xlsx.
        db_path: Path to the DuckDB database.

    Returns:
        Path to the written workbook.
    """
    global DB_PATH
    DB_PATH = db_path

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or (EXPORT_DIR / "spirit_fll_powerbi.xlsx")

    logger.info("Building Power BI workbook → %s", output_path)

    sheets: dict[str, pd.DataFrame] = {
        "otp_monthly":      _sheet_otp_monthly(),
        "delay_causes":     _sheet_delay_causes(),
        "load_factor":      _sheet_load_factor(),
        "weather_impact":   _sheet_weather_impact(),
        "scenario_summary": _sheet_scenario_summary(),
        "demand_forecast":  _sheet_demand_forecast(),
        "metadata":         _sheet_metadata(),
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Auto-fit column widths
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 3, 50)

            logger.info("  %-20s %d rows × %d cols", sheet_name, *df.shape)

    logger.info("Workbook saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    path = generate_workbook()
    print(f"\nExported: {path}")
    sys.exit(0)
