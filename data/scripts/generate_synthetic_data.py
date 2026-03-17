"""
Synthetic data generator for Spirit Airlines FLL Hub OTP & Scenario Engine.

Generates ~500,000 realistic flight records mimicking BTS / T-100 structure for
Spirit Airlines (NK) FLL-hub routes between 2022-01-01 and 2024-12-31, plus
matching weather and capacity data.

Run directly:
    python data/scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Route definitions  (Origin always FLL)
# ---------------------------------------------------------------------------
ROUTES: dict[str, dict] = {
    # FLL outbound (each also generates a return leg, so x2 records)
    "FLL-ATL": {"dest": "ATL", "distance": 581,  "airtime": 85,  "base_lf": 0.87, "base_otp": 0.74, "daily_flights": 8},
    "FLL-LAS": {"dest": "LAS", "distance": 2316, "airtime": 280, "base_lf": 0.91, "base_otp": 0.78, "daily_flights": 4},
    "FLL-LAX": {"dest": "LAX", "distance": 2342, "airtime": 300, "base_lf": 0.88, "base_otp": 0.76, "daily_flights": 3},
    "FLL-ORD": {"dest": "ORD", "distance": 1182, "airtime": 155, "base_lf": 0.85, "base_otp": 0.70, "daily_flights": 6},
    "FLL-DFW": {"dest": "DFW", "distance": 1235, "airtime": 165, "base_lf": 0.83, "base_otp": 0.73, "daily_flights": 6},
    "FLL-MCO": {"dest": "MCO", "distance": 170,  "airtime": 45,  "base_lf": 0.89, "base_otp": 0.77, "daily_flights": 8},
    "FLL-JFK": {"dest": "JFK", "distance": 1069, "airtime": 150, "base_lf": 0.86, "base_otp": 0.68, "daily_flights": 5},
    "FLL-BOS": {"dest": "BOS", "distance": 1240, "airtime": 165, "base_lf": 0.84, "base_otp": 0.71, "daily_flights": 4},
    "FLL-DTW": {"dest": "DTW", "distance": 1154, "airtime": 155, "base_lf": 0.82, "base_otp": 0.72, "daily_flights": 5},
    "FLL-MIA": {"dest": "MIA", "distance": 21,   "airtime": 25,  "base_lf": 0.78, "base_otp": 0.80, "daily_flights": 4},
}

# Spirit A320-family fleet — seats per aircraft type
AIRCRAFT_CONFIG: dict[str, int] = {
    "A319": 145,
    "A320": 178,
    "A321": 228,
}

# Typical scheduled departure blocks (HHMM) per route group
DEPARTURE_BLOCKS: dict[str, list[int]] = {
    "short":  [600, 730, 900, 1030, 1200, 1330, 1500, 1630, 1800, 1930],
    "medium": [600, 730, 900, 1100, 1300, 1500, 1700, 1900],
    "long":   [600, 900, 1200, 1500, 1800],
}

def _get_block(distance: int) -> str:
    if distance < 400:
        return "short"
    if distance < 1500:
        return "medium"
    return "long"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _month_seasonality(month: int) -> float:
    """Return a delay multiplier based on month (1-12)."""
    seasonality = {
        1: 1.25, 2: 1.15, 3: 1.05, 4: 0.90, 5: 0.95,
        6: 1.10, 7: 1.20, 8: 1.30, 9: 1.20, 10: 1.00,
        11: 0.95, 12: 1.30,
    }
    return seasonality[month]


def _hour_effect(hour: int) -> float:
    """Departure hour effect on delay (early = lower, afternoon = higher)."""
    # Late afternoon / evening highest due to propagation
    effects = {
        5: 0.6, 6: 0.7, 7: 0.8, 8: 0.85, 9: 0.90,
        10: 0.95, 11: 1.00, 12: 1.05, 13: 1.10, 14: 1.15,
        15: 1.20, 16: 1.30, 17: 1.35, 18: 1.40, 19: 1.45,
        20: 1.40, 21: 1.30, 22: 1.20, 23: 1.10,
    }
    return effects.get(hour, 1.0)


def _weather_delay(month: int, rng: np.random.Generator) -> tuple[float, float]:
    """Return (weather_delay_minutes, nas_delay_minutes) based on month."""
    # Florida summer thunderstorm season (Jun-Sep) + winter storms (Dec-Feb for connecting airports)
    if month in (6, 7, 8, 9):
        weather_prob = 0.18
        weather_mean = 28
    elif month in (12, 1, 2):
        weather_prob = 0.12
        weather_mean = 22
    else:
        weather_prob = 0.05
        weather_mean = 15

    if rng.random() < weather_prob:
        w = float(rng.exponential(weather_mean))
        n = float(rng.exponential(10)) if rng.random() < 0.5 else 0.0
    else:
        w, n = 0.0, 0.0
    return w, n


def _carrier_delay(base_otp: float, rng: np.random.Generator) -> float:
    """Carrier maintenance / crew delay."""
    prob = (1 - base_otp) * 0.35
    if rng.random() < prob:
        return float(rng.exponential(20))
    return 0.0


def _late_aircraft_delay(prop_factor: float, rng: np.random.Generator) -> float:
    """Delay from previous leg (propagation)."""
    if rng.random() < prop_factor * 0.4:
        return float(rng.exponential(25))
    return 0.0


def _cancellation(total_delay: float, weather_delay: float, rng: np.random.Generator) -> tuple[bool, str]:
    """Return (cancelled, code)."""
    cancel_prob = 0.0
    code = ""
    if weather_delay > 60 or total_delay > 180:
        cancel_prob = 0.08
        code = "B"
    elif total_delay > 90:
        cancel_prob = 0.03
        code = "A"
    elif rng.random() < 0.005:
        cancel_prob = 0.005
        code = "A"
    if cancel_prob and rng.random() < cancel_prob:
        return True, code
    return False, ""


# ---------------------------------------------------------------------------
# Flight data generation
# ---------------------------------------------------------------------------

def generate_flights() -> pd.DataFrame:
    """Generate ~500k synthetic flight records for Spirit FLL routes 2022-2024."""
    logger.info("Generating synthetic flight data …")
    rng = np.random.default_rng(RANDOM_SEED)

    date_range = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    tail_numbers = [f"NK{str(i).zfill(3)}" for i in range(1, 61)]  # 60 tails

    records: list[dict] = []

    for route_key, cfg in ROUTES.items():
        dest = cfg["dest"]
        distance = cfg["distance"]
        airtime_base = cfg["airtime"]
        base_lf = cfg["base_lf"]
        base_otp = cfg["base_otp"]
        n_daily = cfg["daily_flights"]
        block = _get_block(distance)
        dep_blocks_pool = DEPARTURE_BLOCKS[block]
        # Use all available blocks up to n_daily; cycle if needed
        dep_blocks = [dep_blocks_pool[i % len(dep_blocks_pool)] for i in range(n_daily)]

        for date in tqdm(date_range, desc=route_key, leave=False):
            month = date.month
            dow = date.dayofweek  # 0=Mon
            season_mult = _month_seasonality(month)
            is_weekend = dow >= 5
            # Extra flights on weekends for leisure routes
            extra = 1 if is_weekend and rng.random() < 0.3 else 0

            for i, dep_block in enumerate(dep_blocks + dep_blocks[:extra]):
                # Scheduled departure time (minutes from midnight)
                sched_dep_min = dep_block + int(rng.normal(0, 5))
                sched_dep_min = max(300, min(1380, sched_dep_min))
                hour = sched_dep_min // 60

                hour_eff = _hour_effect(hour)

                # Delay components
                weather_del, nas_del = _weather_delay(month, rng)
                carrier_del = _carrier_delay(base_otp, rng)
                prop_factor = season_mult * hour_eff * 0.5
                late_ac_del = _late_aircraft_delay(prop_factor, rng)
                security_del = float(rng.exponential(2)) if rng.random() < 0.01 else 0.0

                # Taxi / gate delays absorbed into departure
                total_delay = (
                    weather_del + nas_del + carrier_del + late_ac_del + security_del
                ) * season_mult * hour_eff

                # Apply random noise
                total_delay = max(0.0, total_delay + rng.normal(0, 3))

                cancelled, cancel_code = _cancellation(total_delay, weather_del, rng)

                if cancelled:
                    dep_delay = np.nan
                    arr_delay = np.nan
                    dep_time = np.nan
                    arr_del15 = 0
                    air_time = np.nan
                else:
                    dep_delay = round(total_delay, 1)
                    dep_time_min = sched_dep_min + dep_delay
                    dep_time = int(dep_time_min) % (24 * 60)
                    # Arrival delay ≈ dep_delay with small airborne recovery
                    recovery = float(rng.uniform(0, min(5, dep_delay))) if dep_delay > 0 else 0
                    arr_delay = round(dep_delay - recovery + rng.normal(0, 2), 1)
                    arr_del15 = int(arr_delay >= 15)
                    air_time = airtime_base + int(rng.normal(0, 8))
                    dep_time = _mins_to_hhmm(dep_time_min)

                sched_dep_hhmm = _mins_to_hhmm(sched_dep_min)
                tail = random.choice(tail_numbers)
                aircraft = random.choices(
                    list(AIRCRAFT_CONFIG.keys()),
                    weights=[0.25, 0.50, 0.25],
                )[0]

                outbound = {
                    "FlightDate": date.strftime("%Y-%m-%d"),
                    "Reporting_Airline": "NK",
                    "Tail_Number": tail,
                    "Flight_Number_Reporting_Airline": f"NK{rng.integers(100, 999)}",
                    "Origin": "FLL",
                    "Dest": dest,
                    "CRSDepTime": sched_dep_hhmm,
                    "DepTime": dep_time,
                    "DepDelay": round(dep_delay, 1) if not np.isnan(dep_delay) else np.nan,
                    "ArrDelay": round(arr_delay, 1) if not np.isnan(arr_delay) else np.nan,
                    "ArrDel15": arr_del15,
                    "Cancelled": int(cancelled),
                    "CancellationCode": cancel_code,
                    "CarrierDelay": round(carrier_del * season_mult * hour_eff, 1) if not cancelled else np.nan,
                    "WeatherDelay": round(weather_del * season_mult, 1) if not cancelled else np.nan,
                    "NASDelay": round(nas_del * season_mult, 1) if not cancelled else np.nan,
                    "SecurityDelay": round(security_del, 1) if not cancelled else np.nan,
                    "LateAircraftDelay": round(late_ac_del * season_mult * hour_eff, 1) if not cancelled else np.nan,
                    "Distance": distance,
                    "AirTime": air_time if not cancelled else np.nan,
                    "AircraftType": aircraft,
                    "Route": route_key,
                }
                records.append(outbound)

                # Generate the return leg (dest → FLL) using same tail
                # Return departs ~45 min after arrival at dest
                turnaround = 45  # minutes ground time
                if not cancelled and not np.isnan(dep_delay):
                    ret_sched_dep_min = sched_dep_min + airtime_base + turnaround
                    ret_sched_dep_min = ret_sched_dep_min % (24 * 60)
                    ret_hour = ret_sched_dep_min // 60
                    ret_hour_eff = _hour_effect(ret_hour)
                    ret_weather_del, ret_nas_del = _weather_delay(month, rng)
                    ret_carrier_del = _carrier_delay(base_otp, rng)
                    # Propagate outbound delay to return
                    ret_prop = max(0.0, dep_delay * 0.6)
                    ret_total_delay = max(0.0, (
                        ret_weather_del + ret_nas_del + ret_carrier_del + ret_prop
                    ) * season_mult * ret_hour_eff + rng.normal(0, 3))
                    ret_cancelled, ret_cancel_code = _cancellation(ret_total_delay, ret_weather_del, rng)
                    if ret_cancelled:
                        ret_dep_delay = np.nan
                        ret_arr_delay = np.nan
                        ret_arr_del15 = 0
                        ret_air_time = np.nan
                        ret_dep_time_hhmm = np.nan
                    else:
                        ret_dep_delay = round(ret_total_delay, 1)
                        ret_recovery = float(rng.uniform(0, min(5, ret_dep_delay))) if ret_dep_delay > 0 else 0
                        ret_arr_delay = round(ret_dep_delay - ret_recovery + rng.normal(0, 2), 1)
                        ret_arr_del15 = int(ret_arr_delay >= 15)
                        ret_air_time = airtime_base + int(rng.normal(0, 8))
                        ret_dep_time_hhmm = _mins_to_hhmm(ret_sched_dep_min + ret_total_delay)
                    ret_route = f"{dest}-FLL"
                    records.append({
                        "FlightDate": date.strftime("%Y-%m-%d"),
                        "Reporting_Airline": "NK",
                        "Tail_Number": tail,
                        "Flight_Number_Reporting_Airline": f"NK{rng.integers(100, 999)}",
                        "Origin": dest,
                        "Dest": "FLL",
                        "CRSDepTime": _mins_to_hhmm(ret_sched_dep_min),
                        "DepTime": ret_dep_time_hhmm,
                        "DepDelay": ret_dep_delay,
                        "ArrDelay": ret_arr_delay,
                        "ArrDel15": ret_arr_del15,
                        "Cancelled": int(ret_cancelled),
                        "CancellationCode": ret_cancel_code,
                        "CarrierDelay": round(ret_carrier_del * season_mult * ret_hour_eff, 1) if not ret_cancelled else np.nan,
                        "WeatherDelay": round(ret_weather_del * season_mult, 1) if not ret_cancelled else np.nan,
                        "NASDelay": round(ret_nas_del * season_mult, 1) if not ret_cancelled else np.nan,
                        "SecurityDelay": 0.0,
                        "LateAircraftDelay": round(ret_prop * season_mult * ret_hour_eff, 1) if not ret_cancelled else np.nan,
                        "Distance": distance,
                        "AirTime": ret_air_time,
                        "AircraftType": aircraft,
                        "Route": ret_route,
                    })

    df = pd.DataFrame(records)
    logger.info("Generated %d flight records across %d routes", len(df), len(ROUTES))
    return df


def _mins_to_hhmm(mins: float) -> int:
    """Convert minutes-from-midnight to HHMM integer."""
    mins_int = int(mins) % (24 * 60)
    h = mins_int // 60
    m = mins_int % 60
    return h * 100 + m


# ---------------------------------------------------------------------------
# T-100 Capacity data
# ---------------------------------------------------------------------------

def generate_capacity() -> pd.DataFrame:
    """Generate monthly T-100 style capacity data per route."""
    logger.info("Generating synthetic T-100 capacity data …")
    rng = np.random.default_rng(RANDOM_SEED + 1)

    months = pd.date_range("2022-01-01", "2024-12-31", freq="MS")
    records = []

    for route_key, cfg in ROUTES.items():
        dest = cfg["dest"]
        base_lf = cfg["base_lf"]
        n_daily = cfg["daily_flights"]
        aircraft_mix = {"A319": 0.25, "A320": 0.50, "A321": 0.25}

        for month_start in months:
            month = month_start.month
            days_in_month = month_start.days_in_month
            season_lf = base_lf + rng.normal(0, 0.03) + _lf_seasonality(month)
            season_lf = np.clip(season_lf, 0.55, 0.98)

            # Departures (some cancellations)
            sched_dep = n_daily * days_in_month
            cancel_rate = 0.015 + (_month_seasonality(month) - 1) * 0.01
            performed = int(sched_dep * (1 - cancel_rate))
            performed = max(performed, 0)

            # Aircraft mix weighted by route distance
            dist = cfg["distance"]
            if dist < 400:
                weights = [0.40, 0.45, 0.15]
            elif dist < 1500:
                weights = [0.20, 0.55, 0.25]
            else:
                weights = [0.10, 0.45, 0.45]

            aircraft_types = list(aircraft_mix.keys())
            seats_per_dep = sum(
                w * AIRCRAFT_CONFIG[a] for w, a in zip(weights, aircraft_types)
            )
            total_seats = int(performed * seats_per_dep)
            passengers = int(total_seats * season_lf)
            # Dominant aircraft this month
            dominant_ac = random.choices(aircraft_types, weights=weights)[0]

            records.append({
                "Year": month_start.year,
                "Month": month,
                "YearMonth": month_start.strftime("%Y-%m"),
                "Origin": "FLL",
                "Dest": dest,
                "Route": route_key,
                "Reporting_Airline": "NK",
                "AircraftType": dominant_ac,
                "DepScheduled": sched_dep,
                "DepPerformed": performed,
                "Seats": total_seats,
                "Passengers": passengers,
                "LoadFactor": round(passengers / total_seats, 4) if total_seats > 0 else np.nan,
                "AvgSeatsPerDep": round(seats_per_dep, 1),
            })

    df = pd.DataFrame(records)
    logger.info("Generated %d capacity records", len(df))
    return df


def _lf_seasonality(month: int) -> float:
    """Load factor seasonal adjustment."""
    adj = {
        1: 0.02, 2: 0.04, 3: 0.06, 4: 0.01, 5: 0.00,
        6: 0.04, 7: 0.06, 8: 0.03, 9: -0.04, 10: -0.02,
        11: -0.01, 12: 0.05,
    }
    return adj[month]


# ---------------------------------------------------------------------------
# Weather data (hourly METAR-style for FLL)
# ---------------------------------------------------------------------------

def generate_weather() -> pd.DataFrame:
    """Generate hourly weather data for FLL (2022-2024)."""
    logger.info("Generating synthetic hourly weather data for FLL …")
    rng = np.random.default_rng(RANDOM_SEED + 2)

    hours = pd.date_range("2022-01-01", "2024-12-31 23:00", freq="h")
    n = len(hours)

    # Base temperature curve (Florida: hot summers ~89°F, mild winters ~65°F)
    day_of_year = hours.dayofyear
    hour_of_day = hours.hour
    month = hours.month

    base_temp = 76 + 13 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    diurnal_temp = 6 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    tmpf = base_temp + diurnal_temp + rng.normal(0, 3, n)

    dwpf = tmpf - rng.uniform(8, 20, n)  # dewpoint typically 8-20°F below

    # Wind
    drct = rng.integers(0, 360, n)
    sknt = np.abs(rng.normal(8, 5, n))  # avg 8 knots
    # Occasional gusty days (thunderstorm season)
    gust_mask = rng.random(n) < 0.05
    gust = np.where(gust_mask, sknt + rng.uniform(10, 25, n), np.nan)

    # Precipitation (Jun-Sep: afternoon thunderstorms; winter: occasional fronts)
    month_arr = np.array(month)
    precip_prob = np.where(
        (month_arr >= 6) & (month_arr <= 9), 0.20,
        np.where((month_arr <= 2) | (month_arr == 12), 0.07, 0.04),
    )
    # Stronger in afternoon
    hour_of_day_arr = np.array(hour_of_day)
    afternoon_mult = 1 + 0.5 * np.clip(np.sin(2 * np.pi * (hour_of_day_arr - 14) / 24), 0, None)
    precip_mask = rng.random(n) < (precip_prob * afternoon_mult)
    p01i = np.where(precip_mask, rng.exponential(0.15, n), 0.0)

    # Visibility (usually 10, degrades with weather)
    vsby = np.where(precip_mask, rng.uniform(2, 7, n), 10.0)
    vsby = np.clip(vsby + rng.normal(0, 0.3, n), 0.25, 10.0)

    # Sky cover
    skyc1_clear = rng.choice(["CLR", "FEW", "SCT", "BKN", "OVC"], size=n,
                              p=[0.30, 0.30, 0.20, 0.15, 0.05])
    skyc1_precip = rng.choice(["BKN", "OVC", "SCT"], size=n, p=[0.40, 0.40, 0.20])
    skyc1 = np.where(precip_mask, skyc1_precip, skyc1_clear)

    # Weather codes (thunderstorms etc.)
    ts_mask = precip_mask & (rng.random(n) < 0.30) & (month_arr >= 6) & (month_arr <= 9)
    wxcodes = np.where(ts_mask, "TSRA", np.where(precip_mask, "RA", ""))

    df = pd.DataFrame({
        "valid": hours,
        "station": "FLL",
        "tmpf": np.round(tmpf, 1),
        "dwpf": np.round(dwpf, 1),
        "drct": drct.astype(float),
        "sknt": np.round(sknt, 1),
        "p01i": np.round(p01i, 2),
        "vsby": np.round(vsby, 1),
        "gust": np.round(gust, 1),
        "skyc1": skyc1,
        "wxcodes": wxcodes,
    })

    logger.info("Generated %d hourly weather records for FLL", len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all synthetic datasets and save to data/raw/."""
    logger.info("=== Spirit Airlines Synthetic Data Generator ===")

    # Flights
    flights_df = generate_flights()
    out = RAW_DIR / "flights.csv"
    flights_df.to_csv(out, index=False)
    logger.info("Saved flights → %s (%d rows)", out, len(flights_df))

    # Capacity
    capacity_df = generate_capacity()
    out = RAW_DIR / "capacity.csv"
    capacity_df.to_csv(out, index=False)
    logger.info("Saved capacity → %s (%d rows)", out, len(capacity_df))

    # Weather
    weather_df = generate_weather()
    out = RAW_DIR / "weather_fll.csv"
    weather_df.to_csv(out, index=False)
    logger.info("Saved weather → %s (%d rows)", out, len(weather_df))

    logger.info("=== Data generation complete ===")


if __name__ == "__main__":
    main()
