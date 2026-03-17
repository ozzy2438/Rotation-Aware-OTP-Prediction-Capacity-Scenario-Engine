"""
Feature engineering module for Spirit Airlines OTP Prediction Engine.

Transforms raw/merged flight data into model-ready features including:
- Time-based features
- Route-level aggregate features
- Weather severity features
- Delay propagation (rolling airport delays)
- Tail-number lag features (previous flight delay)

Usage:
    python src/pipeline/features.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# US federal holidays (approximate, covering 2022-2024)
US_HOLIDAYS: set[str] = {
    # 2022
    "2022-01-17", "2022-02-21", "2022-05-30", "2022-06-20", "2022-07-04",
    "2022-09-05", "2022-10-10", "2022-11-11", "2022-11-24", "2022-12-26",
    # 2023
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-05-29", "2023-06-19",
    "2023-07-04", "2023-09-04", "2023-10-09", "2023-11-10", "2023-11-23",
    "2023-12-25",
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28",
    "2024-12-25",
}

# Route-level baseline OTP (used as a feature)
ROUTE_OTP_BASELINE: dict[str, float] = {
    "FLL-ATL": 0.74, "FLL-LAS": 0.78, "FLL-LAX": 0.76, "FLL-ORD": 0.70,
    "FLL-DFW": 0.73, "FLL-MCO": 0.77, "FLL-JFK": 0.68, "FLL-BOS": 0.71,
    "FLL-DTW": 0.72, "FLL-MIA": 0.80,
}

ROUTE_DISTANCE: dict[str, int] = {
    "FLL-ATL": 581,  "FLL-LAS": 2316, "FLL-LAX": 2342, "FLL-ORD": 1182,
    "FLL-DFW": 1235, "FLL-MCO": 170,  "FLL-JFK": 1069, "FLL-BOS": 1240,
    "FLL-DTW": 1154, "FLL-MIA": 21,
}

# High-congestion destination airports (extra delay risk)
HIGH_CONGESTION_AIRPORTS: set[str] = {"ATL", "ORD", "JFK", "DFW", "LAX"}


# ---------------------------------------------------------------------------
# Individual feature builders
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and time-of-day features.

    Args:
        df: DataFrame with FlightDate (datetime) and DepHour columns.

    Returns:
        DataFrame with added time feature columns.
    """
    logger.debug("Adding time features …")
    df = df.copy()

    if "FlightDate" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["FlightDate"]):
        df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    df["hour_of_day"] = df["DepHour"].clip(0, 23)
    df["day_of_week"] = df["FlightDate"].dt.dayofweek
    df["month"] = df["FlightDate"].dt.month
    df["quarter"] = df["FlightDate"].dt.quarter
    df["day_of_year"] = df["FlightDate"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    date_str = df["FlightDate"].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = date_str.isin(US_HOLIDAYS).astype(int)

    # Peak holiday windows (day before / after major holidays)
    holiday_dates = pd.to_datetime(sorted(US_HOLIDAYS))
    expanded = set()
    for h in holiday_dates:
        for delta in range(-1, 3):
            expanded.add((h + pd.Timedelta(days=delta)).strftime("%Y-%m-%d"))
    df["is_holiday_window"] = date_str.isin(expanded).astype(int)

    # Time-of-day buckets
    df["is_early_morning"] = (df["hour_of_day"] < 7).astype(int)
    df["is_peak_morning"] = ((df["hour_of_day"] >= 7) & (df["hour_of_day"] < 10)).astype(int)
    df["is_afternoon"] = ((df["hour_of_day"] >= 13) & (df["hour_of_day"] < 18)).astype(int)
    df["is_evening"] = (df["hour_of_day"] >= 18).astype(int)

    # Summer season (Jun-Aug)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    # Hurricane season (Aug-Oct)
    df["is_hurricane_season"] = df["month"].isin([8, 9, 10]).astype(int)
    # Winter (Dec-Feb)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)

    # Cyclical encoding for periodic features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add route-level static and computed features.

    Args:
        df: DataFrame with Route and Dest columns.

    Returns:
        DataFrame with added route feature columns.
    """
    logger.debug("Adding route features …")
    df = df.copy()

    df["route_distance"] = df["Route"].map(ROUTE_DISTANCE).fillna(df.get("Distance", 0))
    df["route_otp_baseline"] = df["Route"].map(ROUTE_OTP_BASELINE).fillna(0.75)
    df["is_high_congestion_dest"] = df["Dest"].isin(HIGH_CONGESTION_AIRPORTS).astype(int)
    df["is_long_haul"] = (df["route_distance"] >= 1500).astype(int)
    df["is_short_haul"] = (df["route_distance"] < 400).astype(int)

    # Normalised distance
    max_dist = max(ROUTE_DISTANCE.values())
    df["distance_normalised"] = df["route_distance"] / max_dist

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure weather severity features are present and complete.

    Computes or fills: weather_severity, thunderstorm_flag, precipitation_flag,
    low_visibility_flag, wind_gust_flag.

    Args:
        df: DataFrame potentially containing raw weather columns.

    Returns:
        DataFrame with complete weather feature columns.
    """
    logger.debug("Adding weather features …")
    df = df.copy()

    # If raw weather columns exist, re-derive flags
    if "p01i" in df.columns:
        df["precipitation_flag"] = (df["p01i"].fillna(0) > 0.01).astype(int)
    else:
        df["precipitation_flag"] = df.get("precipitation_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)

    if "vsby" in df.columns:
        df["low_visibility_flag"] = (df["vsby"].fillna(10) < 3.0).astype(int)
    else:
        df["low_visibility_flag"] = df.get("low_visibility_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)

    if "gust" in df.columns:
        df["wind_gust_flag"] = (df["gust"].fillna(0) > 25).astype(int)
    else:
        df["wind_gust_flag"] = df.get("wind_gust_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)

    if "wxcodes" in df.columns:
        df["thunderstorm_flag"] = df["wxcodes"].fillna("").str.contains("TS").astype(int)
    else:
        df["thunderstorm_flag"] = df.get("thunderstorm_flag", pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Composite severity if not present
    if "weather_severity" not in df.columns:
        df["weather_severity"] = (
            df["thunderstorm_flag"] * 4.0
            + df["low_visibility_flag"] * 2.5
            + df["precipitation_flag"] * 1.5
            + df["wind_gust_flag"] * 2.0
        ).clip(0, 10)
    else:
        df["weather_severity"] = df["weather_severity"].fillna(0)

    # Wind speed bins
    if "sknt" in df.columns:
        df["wind_speed"] = df["sknt"].fillna(0)
        df["high_wind_flag"] = (df["wind_speed"] > 20).astype(int)
    else:
        df["wind_speed"] = 0
        df["high_wind_flag"] = 0

    return df


def add_propagation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add delay propagation features: rolling average delay at FLL.

    Computes a 2-hour rolling average of DepDelay across all flights departing
    from FLL, acting as a proxy for airport congestion at that time.

    Args:
        df: DataFrame sorted by FlightDate + DepHour.

    Returns:
        DataFrame with rolling_avg_dep_delay column.
    """
    logger.debug("Adding propagation features …")
    df = df.copy()

    # Build a datetime column for sorting
    if "FlightDate" in df.columns:
        base_dt = pd.to_datetime(df["FlightDate"])
    else:
        logger.warning("FlightDate not found; skipping propagation features")
        df["rolling_avg_dep_delay"] = 0.0
        return df

    dep_hour = df.get("DepHour", df.get("hour_of_day", pd.Series(12, index=df.index))).fillna(12).astype(int)
    df["_dep_dt"] = base_dt + pd.to_timedelta(dep_hour, unit="h")

    df = df.sort_values("_dep_dt").reset_index(drop=True)

    # Rolling 2-hour window mean (only for FLL origin)
    origin_mask = df["Origin"] == "FLL"
    delay_series = df["DepDelay"].where(origin_mask)

    # Time-indexed rolling
    temp = df[["_dep_dt", "DepDelay"]].copy()
    temp = temp.set_index("_dep_dt")
    rolling_mean = (
        temp["DepDelay"]
        .rolling("2h", min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
    df["rolling_avg_dep_delay"] = rolling_mean.fillna(0).values

    df = df.drop(columns=["_dep_dt"])
    return df


def add_tail_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-flight delay lag for each tail number.

    For each aircraft tail, look up the most recent prior DepDelay.
    This captures inbound delay propagation from the previous leg.

    Args:
        df: DataFrame with Tail_Number, FlightDate, DepHour, DepDelay.

    Returns:
        DataFrame with prev_tail_dep_delay column.
    """
    logger.debug("Adding tail-number lag features …")
    df = df.copy()

    if "Tail_Number" not in df.columns or "DepDelay" not in df.columns:
        df["prev_tail_dep_delay"] = 0.0
        return df

    # Sort chronologically
    if "FlightDate" in df.columns:
        base_dt = pd.to_datetime(df["FlightDate"])
        dep_hour = df.get("DepHour", pd.Series(12, index=df.index)).fillna(12).astype(int)
        df["_sort_dt"] = base_dt + pd.to_timedelta(dep_hour, unit="h")
        df = df.sort_values(["Tail_Number", "_sort_dt"]).reset_index(drop=True)
    else:
        df["_sort_dt"] = pd.Series(range(len(df)), index=df.index)

    # Shift DepDelay within each tail group
    df["prev_tail_dep_delay"] = (
        df.groupby("Tail_Number")["DepDelay"]
        .shift(1)
        .fillna(0)
    )

    df = df.drop(columns=["_sort_dt"])
    return df


def add_rotation_chain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rotation-chain awareness features for each aircraft tail number.

    Airline delay does not originate flight-by-flight — it propagates through
    an aircraft's daily rotation chain.  A tail that ran 3 late legs before noon
    will almost certainly depart late in the afternoon.  These features capture
    that chain-level signal:

    - ``legs_today_before``: How many legs has this tail already flown today
      before this departure?  High values → end-of-day congestion risk.
    - ``cumulative_tail_delay_today``: Sum of DepDelay minutes on prior legs
      today.  A proxy for accumulated schedule debt.
    - ``prev_tail_arr_delay``: Arrival delay of the inbound leg (not just
      departure delay).  The aircraft that arrives 30 min late will almost
      always depart 30 min late — this is the strongest single propagation
      signal.
    - ``hours_since_last_flight``: Hours between the previous arrival and
      this scheduled departure.  < 1 h = quick-turn risk; > 4 h = buffer.
    - ``quick_turn_flag``: Binary flag: gap < 60 minutes.
    - ``tail_delay_streak``: Count of consecutive delayed legs (ArrDel15=1)
      immediately before this flight.  A streak of 3+ signals systemic
      disruption, not random noise.

    Args:
        df: DataFrame with Tail_Number, FlightDate, DepHour, DepDelay,
            ArrDelay, AirTime columns.

    Returns:
        DataFrame with rotation chain feature columns added.
    """
    logger.debug("Adding rotation chain features …")
    df = df.copy()

    required = {"Tail_Number", "FlightDate", "DepDelay"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("Skipping rotation chain features — missing columns: %s", missing)
        for col in [
            "legs_today_before", "cumulative_tail_delay_today",
            "prev_tail_arr_delay", "hours_since_last_flight",
            "quick_turn_flag", "tail_delay_streak",
        ]:
            df[col] = 0.0
        return df

    # Build precise departure datetime for sorting
    base_dt = pd.to_datetime(df["FlightDate"])
    dep_hour = df.get("DepHour", pd.Series(12, index=df.index)).fillna(12).astype(int)
    df["_dep_dt"] = base_dt + pd.to_timedelta(dep_hour, unit="h")

    # Estimated arrival datetime (departure + airtime)
    air_min = df.get("AirTime", pd.Series(90, index=df.index)).fillna(90)
    dep_delay_min = df["DepDelay"].fillna(0)
    df["_arr_dt"] = df["_dep_dt"] + pd.to_timedelta(dep_delay_min + air_min, unit="m")

    df = df.sort_values(["Tail_Number", "_dep_dt"]).reset_index(drop=True)

    grp = df.groupby("Tail_Number")

    # 1. Legs flown today before this departure (within same calendar day)
    df["_date"] = df["_dep_dt"].dt.date
    df["legs_today_before"] = (
        df.groupby(["Tail_Number", "_date"]).cumcount()  # 0-indexed within day
    )

    # 2. Cumulative DepDelay minutes for this tail on today's prior legs
    df["_dep_delay_filled"] = dep_delay_min.values
    df["cumulative_tail_delay_today"] = (
        df.groupby(["Tail_Number", "_date"])["_dep_delay_filled"]
        .transform(lambda s: s.shift(1).fillna(0).cumsum())
    )

    # 3. Previous-leg arrival delay (strongest propagation signal)
    arr_delay = df.get("ArrDelay", pd.Series(0.0, index=df.index)).fillna(0)
    df["prev_tail_arr_delay"] = grp["ArrDelay"].shift(1).fillna(0) if "ArrDelay" in df.columns else 0.0

    # 4. Hours between previous arrival and this scheduled departure
    prev_arr_dt = grp["_arr_dt"].shift(1)
    gap_hours = (df["_dep_dt"] - prev_arr_dt).dt.total_seconds() / 3600
    # Gap only meaningful within same tail rotation (same or next day, < 16 h)
    df["hours_since_last_flight"] = gap_hours.where(
        (gap_hours >= 0) & (gap_hours <= 16), other=np.nan
    ).fillna(4.0)  # default: assume comfortable 4-h buffer

    # 5. Quick-turn flag: gap < 1 hour → very high delay propagation risk
    df["quick_turn_flag"] = (df["hours_since_last_flight"] < 1.0).astype(int)

    # 6. Tail delay streak: consecutive ArrDel15=1 legs ending at this flight
    if "ArrDel15" in df.columns:
        def _streak(series: pd.Series) -> pd.Series:
            """Count consecutive 1s ending at each position (look-back only)."""
            result = []
            count = 0
            for val in series:
                if val == 1:
                    count += 1
                else:
                    count = 0
                result.append(count)
            # Shift so current flight sees the streak from prior legs
            return pd.Series(result, index=series.index).shift(1).fillna(0)

        df["tail_delay_streak"] = (
            df.groupby("Tail_Number")["ArrDel15"].transform(_streak)
        )
    else:
        df["tail_delay_streak"] = 0.0

    # Clip to sensible ranges
    df["legs_today_before"] = df["legs_today_before"].clip(0, 10)
    df["cumulative_tail_delay_today"] = df["cumulative_tail_delay_today"].clip(-30, 300)
    df["prev_tail_arr_delay"] = df["prev_tail_arr_delay"].clip(-30, 300)
    df["hours_since_last_flight"] = df["hours_since_last_flight"].clip(0, 16)
    df["tail_delay_streak"] = df["tail_delay_streak"].clip(0, 10)

    df = df.drop(columns=["_dep_dt", "_arr_dt", "_date", "_dep_delay_filled"], errors="ignore")
    logger.debug("Rotation chain features added")
    return df


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    include_propagation: bool = True,
    include_tail_lag: bool = True,
    include_rotation_chain: bool = True,
) -> pd.DataFrame:
    """Apply all feature engineering steps to a flight DataFrame.

    Args:
        df: Merged/cleaned flight DataFrame.
        include_propagation: Whether to compute rolling airport delay features.
        include_tail_lag: Whether to compute tail-number lag features.
        include_rotation_chain: Whether to compute full rotation-chain features.

    Returns:
        Feature-enriched DataFrame ready for model training.
    """
    logger.info("Building features for %d records …", len(df))

    df = add_time_features(df)
    df = add_route_features(df)
    df = add_weather_features(df)

    if include_propagation:
        df = add_propagation_features(df)
    else:
        df["rolling_avg_dep_delay"] = 0.0

    if include_tail_lag:
        df = add_tail_lag_features(df)
    else:
        df["prev_tail_dep_delay"] = 0.0

    if include_rotation_chain:
        df = add_rotation_chain_features(df)
    else:
        for col in [
            "legs_today_before", "cumulative_tail_delay_today",
            "prev_tail_arr_delay", "hours_since_last_flight",
            "quick_turn_flag", "tail_delay_streak",
        ]:
            df[col] = 0.0

    logger.info("Feature engineering complete: %d columns", df.shape[1])
    return df


def get_feature_columns() -> list[str]:
    """Return the canonical list of model feature column names.

    Returns:
        List of feature names used by OTP predictor.
    """
    return [
        # Time
        "hour_of_day", "day_of_week", "month", "quarter",
        "is_weekend", "is_monday", "is_friday",
        "is_holiday", "is_holiday_window",
        "is_early_morning", "is_peak_morning", "is_afternoon", "is_evening",
        "is_summer", "is_hurricane_season", "is_winter",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        # Route
        "route_distance", "route_otp_baseline",
        "is_high_congestion_dest", "is_long_haul", "is_short_haul",
        "distance_normalised",
        # Weather
        "weather_severity", "thunderstorm_flag", "precipitation_flag",
        "low_visibility_flag", "wind_gust_flag", "wind_speed", "high_wind_flag",
        # Propagation
        "rolling_avg_dep_delay",
        # Tail lag
        "prev_tail_dep_delay",
        # Rotation chain
        "legs_today_before", "cumulative_tail_delay_today",
        "prev_tail_arr_delay", "hours_since_last_flight",
        "quick_turn_flag", "tail_delay_streak",
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    proc_path = PROCESSED_DIR / "flights_processed.parquet"
    if not proc_path.exists():
        logger.error("Processed flight data not found at %s. Run etl.py first.", proc_path)
        sys.exit(1)

    df = pd.read_parquet(proc_path)
    df = build_features(df)

    out = PROCESSED_DIR / "flights_features.parquet"
    df.to_parquet(out, index=False)
    logger.info("Saved feature-enriched data → %s (%d rows, %d cols)", out, *df.shape)

    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    logger.info("Feature columns available: %d / %d", len(available), len(feature_cols))
