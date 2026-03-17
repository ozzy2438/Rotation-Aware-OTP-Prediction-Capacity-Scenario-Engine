"""
pytest tests for ETL pipeline: loading, cleaning, merging, and validation.

Run:
    pytest tests/test_etl.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.etl import (
    clean_capacity,
    clean_flights,
    clean_weather,
    merge_flights_weather,
)
from src.pipeline.features import (
    add_route_features,
    add_time_features,
    add_weather_features,
    build_features,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_flights() -> pd.DataFrame:
    """Minimal synthetic flight DataFrame for testing."""
    return pd.DataFrame({
        "FlightDate": pd.to_datetime(["2023-06-15", "2023-06-15", "2023-12-20", "2023-01-01"]),
        "Reporting_Airline": ["NK", "NK", "NK", "NK"],
        "Origin": ["FLL", "FLL", "FLL", "FLL"],
        "Dest": ["ATL", "LAS", "ORD", "JFK"],
        "Route": ["FLL-ATL", "FLL-LAS", "FLL-ORD", "FLL-JFK"],
        "CRSDepTime": [800, 1300, 600, 1700],
        "DepTime": [815.0, 1310.0, 602.0, np.nan],
        "DepDelay": [15.0, 10.0, 2.0, np.nan],
        "ArrDelay": [12.0, 8.0, -3.0, np.nan],
        "ArrDel15": [0, 0, 0, 0],
        "Cancelled": [0, 0, 0, 1],
        "CancellationCode": ["", "", "", "B"],
        "CarrierDelay": [0.0, 0.0, 0.0, np.nan],
        "WeatherDelay": [15.0, 0.0, 0.0, np.nan],
        "NASDelay": [0.0, 0.0, 0.0, np.nan],
        "SecurityDelay": [0.0, 0.0, 0.0, np.nan],
        "LateAircraftDelay": [0.0, 10.0, 0.0, np.nan],
        "Distance": [581, 2316, 1182, 1069],
        "AirTime": [85.0, 280.0, 155.0, np.nan],
        "AircraftType": ["A320", "A321", "A320", "A319"],
        "Tail_Number": ["NK001", "NK002", "NK003", "NK004"],
    })


@pytest.fixture
def sample_weather() -> pd.DataFrame:
    """Minimal synthetic weather DataFrame."""
    hours = pd.date_range("2023-06-15 00:00", periods=48, freq="h")
    n = len(hours)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "valid": hours,
        "station": "FLL",
        "tmpf": rng.uniform(75, 90, n),
        "dwpf": rng.uniform(65, 80, n),
        "drct": rng.integers(0, 360, n).astype(float),
        "sknt": rng.uniform(5, 20, n),
        "p01i": np.where(rng.random(n) < 0.15, rng.exponential(0.1, n), 0.0),
        "vsby": rng.uniform(5, 10, n),
        "gust": np.where(rng.random(n) < 0.05, rng.uniform(25, 40, n), np.nan),
        "skyc1": rng.choice(["CLR", "FEW", "SCT", "BKN"], size=n),
        "wxcodes": np.where(rng.random(n) < 0.05, "TSRA", ""),
    })


@pytest.fixture
def sample_capacity() -> pd.DataFrame:
    """Minimal synthetic capacity DataFrame."""
    return pd.DataFrame({
        "Year": [2023, 2023, 2023],
        "Month": [6, 7, 8],
        "YearMonth": ["2023-06", "2023-07", "2023-08"],
        "Origin": ["FLL", "FLL", "FLL"],
        "Dest": ["ATL", "ATL", "ATL"],
        "Route": ["FLL-ATL", "FLL-ATL", "FLL-ATL"],
        "Reporting_Airline": ["NK", "NK", "NK"],
        "AircraftType": ["A320", "A320", "A321"],
        "DepScheduled": [180, 186, 186],
        "DepPerformed": [177, 183, 182],
        "Seats": [31_560, 32_574, 41_496],
        "Passengers": [27_469, 29_316, 36_110],
        "LoadFactor": [0.87, 0.90, 0.87],
        "AvgSeatsPerDep": [178.3, 178.0, 228.0],
    })


# ---------------------------------------------------------------------------
# Tests: clean_flights
# ---------------------------------------------------------------------------

class TestCleanFlights:
    def test_returns_dataframe(self, sample_flights):
        result = clean_flights(sample_flights)
        assert isinstance(result, pd.DataFrame)

    def test_derives_year_month(self, sample_flights):
        result = clean_flights(sample_flights)
        assert "Year" in result.columns
        assert "Month" in result.columns
        assert result["Year"].iloc[0] == 2023
        assert result["Month"].iloc[0] == 6

    def test_derives_dep_hour(self, sample_flights):
        result = clean_flights(sample_flights)
        assert "DepHour" in result.columns
        assert result["DepHour"].iloc[0] == 8  # CRSDepTime=800 → hour 8

    def test_arr_del15_is_int(self, sample_flights):
        result = clean_flights(sample_flights)
        assert result["ArrDel15"].dtype in (int, np.int64, np.int32)

    def test_cancelled_is_int(self, sample_flights):
        result = clean_flights(sample_flights)
        assert result["Cancelled"].dtype in (int, np.int64, np.int32)

    def test_delay_clamping(self, sample_flights):
        df = sample_flights.copy()
        df.loc[0, "DepDelay"] = 700.0  # above 600 cap
        df.loc[1, "ArrDelay"] = -50.0  # below -30 cap
        result = clean_flights(df)
        assert result.loc[result.index[0], "DepDelay"] <= 600
        assert result.loc[result.index[1], "ArrDelay"] >= -30

    def test_drops_rows_missing_route(self, sample_flights):
        df = sample_flights.copy()
        df.loc[0, "Route"] = np.nan
        result = clean_flights(df)
        assert len(result) == len(sample_flights) - 1

    def test_quarter_derived(self, sample_flights):
        result = clean_flights(sample_flights)
        assert "Quarter" in result.columns
        assert result.loc[result["Month"] == 6, "Quarter"].iloc[0] == 2


# ---------------------------------------------------------------------------
# Tests: clean_weather
# ---------------------------------------------------------------------------

class TestCleanWeather:
    def test_returns_dataframe(self, sample_weather):
        result = clean_weather(sample_weather)
        assert isinstance(result, pd.DataFrame)

    def test_thunderstorm_flag(self, sample_weather):
        df = sample_weather.copy()
        df.loc[0, "wxcodes"] = "TSRA"
        result = clean_weather(df)
        assert result.loc[0, "thunderstorm_flag"] == 1

    def test_precipitation_flag(self, sample_weather):
        df = sample_weather.copy()
        df.loc[0, "p01i"] = 0.5
        result = clean_weather(df)
        assert result.loc[0, "precipitation_flag"] == 1

    def test_low_visibility_flag(self, sample_weather):
        df = sample_weather.copy()
        df.loc[0, "vsby"] = 1.5
        result = clean_weather(df)
        assert result.loc[0, "low_visibility_flag"] == 1

    def test_weather_severity_range(self, sample_weather):
        result = clean_weather(sample_weather)
        assert result["weather_severity"].between(0, 10).all()

    def test_no_negative_precipitation(self, sample_weather):
        result = clean_weather(sample_weather)
        assert (result["p01i"] >= 0).all()


# ---------------------------------------------------------------------------
# Tests: clean_capacity
# ---------------------------------------------------------------------------

class TestCleanCapacity:
    def test_returns_dataframe(self, sample_capacity):
        result = clean_capacity(sample_capacity)
        assert isinstance(result, pd.DataFrame)

    def test_load_factor_range(self, sample_capacity):
        result = clean_capacity(sample_capacity)
        valid = result["LoadFactor"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_drops_missing_route(self, sample_capacity):
        df = sample_capacity.copy()
        df.loc[0, "Route"] = np.nan
        result = clean_capacity(df)
        assert len(result) == len(sample_capacity) - 1


# ---------------------------------------------------------------------------
# Tests: merge_flights_weather
# ---------------------------------------------------------------------------

class TestMergeFlightsWeather:
    def test_merge_produces_weather_cols(self, sample_flights, sample_weather):
        flights_clean = clean_flights(sample_flights)
        weather_clean = clean_weather(sample_weather)
        result = merge_flights_weather(flights_clean, weather_clean)
        assert "weather_severity" in result.columns
        assert "thunderstorm_flag" in result.columns

    def test_row_count_preserved(self, sample_flights, sample_weather):
        flights_clean = clean_flights(sample_flights)
        weather_clean = clean_weather(sample_weather)
        result = merge_flights_weather(flights_clean, weather_clean)
        assert len(result) == len(flights_clean)

    def test_no_nan_in_weather_severity(self, sample_flights, sample_weather):
        flights_clean = clean_flights(sample_flights)
        weather_clean = clean_weather(sample_weather)
        result = merge_flights_weather(flights_clean, weather_clean)
        assert result["weather_severity"].notna().all()


# ---------------------------------------------------------------------------
# Tests: feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_add_time_features(self, sample_flights):
        df = clean_flights(sample_flights)
        result = add_time_features(df)
        assert "hour_sin" in result.columns
        assert "is_weekend" in result.columns
        assert "is_summer" in result.columns
        assert result["is_summer"].iloc[0] == 1  # June 15

    def test_add_route_features(self, sample_flights):
        df = clean_flights(sample_flights)
        result = add_route_features(df)
        assert "route_distance" in result.columns
        assert "route_otp_baseline" in result.columns
        assert result.loc[result["Route"] == "FLL-ATL", "route_distance"].iloc[0] == 581

    def test_add_weather_features_defaults(self, sample_flights):
        df = clean_flights(sample_flights)
        result = add_weather_features(df)
        assert "weather_severity" in result.columns
        assert "precipitation_flag" in result.columns

    def test_build_features_all_columns(self, sample_flights, sample_weather):
        flights_clean = clean_flights(sample_flights)
        weather_clean = clean_weather(sample_weather)
        merged = merge_flights_weather(flights_clean, weather_clean)
        result = build_features(merged, include_propagation=False, include_tail_lag=False)
        feature_cols = get_feature_columns()
        for col in feature_cols:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_cyclical_encoding_range(self, sample_flights):
        df = clean_flights(sample_flights)
        result = add_time_features(df)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()

    def test_is_holiday_detection(self):
        df = pd.DataFrame({
            "FlightDate": pd.to_datetime(["2023-07-04", "2023-06-15"]),
            "Route": ["FLL-ATL", "FLL-ATL"],
            "Origin": ["FLL", "FLL"],
            "Dest": ["ATL", "ATL"],
            "CRSDepTime": [800, 800],
            "DepHour": [8, 8],
            "Cancelled": [0, 0],
            "ArrDel15": [0, 0],
            "DepDelay": [0.0, 0.0],
            "ArrDelay": [0.0, 0.0],
            "Year": [2023, 2023],
            "Month": [7, 6],
            "DayOfWeek": [1, 3],
            "DayOfYear": [185, 166],
            "Quarter": [3, 2],
        })
        result = add_time_features(df)
        assert result.loc[result["FlightDate"] == pd.Timestamp("2023-07-04"), "is_holiday"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Tests: get_feature_columns
# ---------------------------------------------------------------------------

class TestGetFeatureColumns:
    def test_returns_list(self):
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 10

    def test_no_duplicates(self):
        cols = get_feature_columns()
        assert len(cols) == len(set(cols))

    def test_includes_key_features(self):
        cols = get_feature_columns()
        required = [
            "weather_severity", "rolling_avg_dep_delay", "hour_of_day",
            "route_otp_baseline", "thunderstorm_flag",
        ]
        for f in required:
            assert f in cols, f"Feature '{f}' missing from feature list"
