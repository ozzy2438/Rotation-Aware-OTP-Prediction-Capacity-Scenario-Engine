"""
ETL pipeline for Spirit Airlines FLL Hub OTP & Scenario Engine.

Loads raw CSVs, cleans/validates, merges with weather data,
writes processed Parquet files, and registers everything into DuckDB.

Usage:
    python src/pipeline/etl.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PROJECT_ROOT / "data" / "spirit_otp.duckdb"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_flights(path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw flight CSV and apply basic dtype coercions.

    Args:
        path: Path to CSV file. Defaults to data/raw/flights.csv.

    Returns:
        Raw flight DataFrame with corrected dtypes.
    """
    path = path or RAW_DIR / "flights.csv"
    logger.info("Loading flights from %s", path)
    df = pd.read_csv(
        path,
        dtype={
            "Reporting_Airline": "category",
            "Origin": "category",
            "Dest": "category",
            "AircraftType": "category",
            "Route": "category",
            "CancellationCode": "category",
            "Tail_Number": "category",
        },
        parse_dates=["FlightDate"],
        low_memory=False,
    )
    logger.info("Loaded %d flight records", len(df))
    return df


def load_capacity(path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw T-100 capacity CSV.

    Args:
        path: Path to CSV file. Defaults to data/raw/capacity.csv.

    Returns:
        Raw capacity DataFrame.
    """
    path = path or RAW_DIR / "capacity.csv"
    logger.info("Loading capacity from %s", path)
    df = pd.read_csv(
        path,
        dtype={
            "Origin": "category",
            "Dest": "category",
            "Route": "category",
            "Reporting_Airline": "category",
            "AircraftType": "category",
        },
        low_memory=False,
    )
    logger.info("Loaded %d capacity records", len(df))
    return df


def load_weather(path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw hourly weather CSV for FLL.

    Args:
        path: Path to CSV file. Defaults to data/raw/weather_fll.csv.

    Returns:
        Weather DataFrame with parsed timestamps.
    """
    path = path or RAW_DIR / "weather_fll.csv"
    logger.info("Loading weather from %s", path)
    df = pd.read_csv(path, parse_dates=["valid"], low_memory=False)
    df = df.sort_values("valid").reset_index(drop=True)
    logger.info("Loaded %d weather records", len(df))
    return df


# ---------------------------------------------------------------------------
# Cleaning & validation
# ---------------------------------------------------------------------------

def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate flight records.

    - Remove records with invalid dates or missing core fields.
    - Clamp delay values to [-30, 600] minute window.
    - Derive helper columns: Year, Month, DayOfWeek, Hour.

    Args:
        df: Raw flight DataFrame.

    Returns:
        Cleaned flight DataFrame.
    """
    logger.info("Cleaning flights (%d rows) …", len(df))
    initial = len(df)

    # Drop rows missing FlightDate or Route
    df = df.dropna(subset=["FlightDate", "Route", "Origin", "Dest"])

    # Coerce numeric columns
    numeric_cols = [
        "DepDelay", "ArrDelay", "CarrierDelay", "WeatherDelay",
        "NASDelay", "SecurityDelay", "LateAircraftDelay", "AirTime",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clamp delays
    for col in ["DepDelay", "ArrDelay"]:
        df[col] = df[col].clip(-30, 600)

    # Ensure ArrDel15 is integer
    df["ArrDel15"] = df["ArrDel15"].fillna(0).astype(int)
    df["Cancelled"] = df["Cancelled"].fillna(0).astype(int)

    # Derive time features from FlightDate
    df["Year"] = df["FlightDate"].dt.year
    df["Month"] = df["FlightDate"].dt.month
    df["DayOfWeek"] = df["FlightDate"].dt.dayofweek  # 0=Mon
    df["DayOfYear"] = df["FlightDate"].dt.dayofyear
    df["Quarter"] = df["FlightDate"].dt.quarter

    # Departure hour from CRSDepTime (HHMM integer)
    df["DepHour"] = (df["CRSDepTime"].fillna(0).astype(int) // 100).clip(0, 23)

    dropped = initial - len(df)
    logger.info("Cleaned flights: %d rows dropped, %d remaining", dropped, len(df))
    return df.reset_index(drop=True)


def clean_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean capacity records.

    Args:
        df: Raw capacity DataFrame.

    Returns:
        Cleaned capacity DataFrame.
    """
    logger.info("Cleaning capacity (%d rows) …", len(df))
    df = df.dropna(subset=["Route", "Year", "Month"])
    df["LoadFactor"] = pd.to_numeric(df["LoadFactor"], errors="coerce")
    df["LoadFactor"] = df["LoadFactor"].clip(0.0, 1.0)
    logger.info("Capacity cleaning complete: %d rows", len(df))
    return df.reset_index(drop=True)


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean hourly weather records.

    Args:
        df: Raw weather DataFrame.

    Returns:
        Cleaned weather DataFrame with derived severity metrics.
    """
    logger.info("Cleaning weather (%d rows) …", len(df))
    df = df.dropna(subset=["valid"])
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce").clip(-10, 120)
    df["sknt"] = pd.to_numeric(df["sknt"], errors="coerce").abs().fillna(0)
    df["p01i"] = pd.to_numeric(df["p01i"], errors="coerce").fillna(0).clip(0, 10)
    df["vsby"] = pd.to_numeric(df["vsby"], errors="coerce").clip(0, 10).fillna(10)
    df["gust"] = pd.to_numeric(df["gust"], errors="coerce")

    # Derived flags
    df["thunderstorm_flag"] = df["wxcodes"].fillna("").str.contains("TS").astype(int)
    df["precipitation_flag"] = (df["p01i"] > 0.01).astype(int)
    df["low_visibility_flag"] = (df["vsby"] < 3.0).astype(int)
    df["wind_gust_flag"] = (df["gust"].fillna(0) > 25).astype(int)

    # Composite weather severity score [0-10]
    df["weather_severity"] = (
        df["thunderstorm_flag"] * 4.0
        + df["low_visibility_flag"] * 2.5
        + df["precipitation_flag"] * 1.5
        + df["wind_gust_flag"] * 2.0
        + (df["p01i"] / 0.5).clip(0, 1) * 1.0
    ).clip(0, 10)

    logger.info("Weather cleaning complete: %d rows", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_flights_weather(
    flights: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge flight records with nearest-hour FLL weather at scheduled departure.

    Strategy: round FlightDate + DepHour to the nearest weather observation,
    then left-join on that rounded timestamp.

    Args:
        flights: Cleaned flight DataFrame.
        weather: Cleaned weather DataFrame.

    Returns:
        Merged DataFrame with weather columns appended.
    """
    logger.info("Merging flights with weather …")

    # Build a lookup keyed on (date, hour)
    weather_cols = [
        "valid", "tmpf", "dwpf", "drct", "sknt", "p01i", "vsby",
        "gust", "skyc1", "wxcodes", "thunderstorm_flag", "precipitation_flag",
        "low_visibility_flag", "wind_gust_flag", "weather_severity",
    ]
    wx = weather[weather_cols].copy()
    wx["wx_date"] = wx["valid"].dt.date
    wx["wx_hour"] = wx["valid"].dt.hour

    # Merge key on flights side
    flights["wx_date"] = flights["FlightDate"].dt.date
    flights["wx_hour"] = flights["DepHour"]

    merged = flights.merge(
        wx.drop(columns=["valid"]),
        on=["wx_date", "wx_hour"],
        how="left",
    )

    # Drop helper keys
    merged = merged.drop(columns=["wx_date", "wx_hour"])

    # Fill missing weather with safe defaults
    merged["weather_severity"] = merged["weather_severity"].fillna(0)
    merged["thunderstorm_flag"] = merged["thunderstorm_flag"].fillna(0).astype(int)
    merged["precipitation_flag"] = merged["precipitation_flag"].fillna(0).astype(int)
    merged["low_visibility_flag"] = merged["low_visibility_flag"].fillna(0).astype(int)
    merged["wind_gust_flag"] = merged["wind_gust_flag"].fillna(0).astype(int)
    merged["p01i"] = merged["p01i"].fillna(0)
    merged["vsby"] = merged["vsby"].fillna(10)

    logger.info("Merged flights+weather: %d rows", len(merged))
    return merged


# ---------------------------------------------------------------------------
# DuckDB registration
# ---------------------------------------------------------------------------

def register_duckdb(
    flights: pd.DataFrame,
    capacity: pd.DataFrame,
    weather: pd.DataFrame,
    db_path: Optional[Path] = None,
) -> duckdb.DuckDBPyConnection:
    """Register all processed DataFrames in DuckDB and run schema SQL.

    Args:
        flights: Processed flight DataFrame.
        capacity: Processed capacity DataFrame.
        weather: Processed weather DataFrame.
        db_path: Path for persistent DuckDB file (None = in-memory).

    Returns:
        Open DuckDB connection.
    """
    db_path = db_path or DB_PATH
    logger.info("Registering tables in DuckDB at %s …", db_path)

    con = duckdb.connect(str(db_path))

    # Create tables from DataFrames
    con.execute("DROP TABLE IF EXISTS flights")
    con.execute("DROP TABLE IF EXISTS capacity")
    con.execute("DROP TABLE IF EXISTS weather")
    con.register("flights_df", flights)
    con.register("capacity_df", capacity)
    con.register("weather_df", weather)
    con.execute("CREATE TABLE flights AS SELECT * FROM flights_df")
    con.execute("CREATE TABLE capacity AS SELECT * FROM capacity_df")
    con.execute("CREATE TABLE weather AS SELECT * FROM weather_df")

    # Load and execute SQL schema / views
    sql_path = PROJECT_ROOT / "src" / "sql" / "create_tables.sql"
    if sql_path.exists():
        with open(sql_path) as fh:
            sql = fh.read()
        # Execute statements individually, stripping comment-only lines first
        for stmt in sql.split(";"):
            # Remove comment-only lines so we detect real SQL correctly
            sql_lines = [
                line for line in stmt.splitlines()
                if line.strip() and not line.strip().startswith("--")
            ]
            stmt_clean = "\n".join(sql_lines).strip()
            if stmt_clean:
                try:
                    con.execute(stmt_clean)
                except Exception as exc:
                    logger.warning("SQL stmt failed (skipping): %s — %s", stmt_clean[:60], exc)

    logger.info("DuckDB registration complete")
    return con


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_etl() -> duckdb.DuckDBPyConnection:
    """Full ETL pipeline: load → clean → merge → save → register DuckDB.

    Returns:
        Open DuckDB connection with all tables loaded.
    """
    logger.info("===== ETL Pipeline Start =====")

    # Load
    flights_raw = load_flights()
    capacity_raw = load_capacity()
    weather_raw = load_weather()

    # Clean
    flights = clean_flights(flights_raw)
    capacity = clean_capacity(capacity_raw)
    weather = clean_weather(weather_raw)

    # Merge
    flights_merged = merge_flights_weather(flights, weather)

    # Save processed
    flights_out = PROCESSED_DIR / "flights_processed.parquet"
    capacity_out = PROCESSED_DIR / "capacity_processed.parquet"
    weather_out = PROCESSED_DIR / "weather_processed.parquet"

    flights_merged.to_parquet(flights_out, index=False)
    capacity.to_parquet(capacity_out, index=False)
    weather.to_parquet(weather_out, index=False)
    logger.info("Saved processed files to %s", PROCESSED_DIR)

    # DuckDB
    con = register_duckdb(flights_merged, capacity, weather)

    logger.info("===== ETL Pipeline Complete =====")
    return con


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    con = run_etl()
    result = con.execute("SELECT Route, COUNT(*) as cnt FROM flights GROUP BY Route ORDER BY cnt DESC").fetchdf()
    print(result.to_string(index=False))
    con.close()
