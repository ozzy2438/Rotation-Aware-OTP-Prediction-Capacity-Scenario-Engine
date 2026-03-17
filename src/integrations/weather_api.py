"""OpenWeatherMap integration for real-time airport weather conditions.

Provides a WeatherClient that fetches current weather for Spirit Airlines
hub airports and converts the response into a severity-scored WeatherSnapshot.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Airport coordinates (lat, lon)
# ---------------------------------------------------------------------------

AIRPORT_COORDS: dict[str, tuple[float, float]] = {
    "FLL": (26.0726, -80.1527),
    "ATL": (33.6407, -84.4277),
    "LAS": (36.0840, -115.1537),
    "LAX": (33.9425, -118.4081),
    "ORD": (41.9742, -87.9073),
    "DFW": (32.8998, -97.0403),
    "MCO": (28.4312, -81.3081),
    "JFK": (40.6413, -73.7781),
    "BOS": (42.3656, -71.0096),
    "DTW": (42.2162, -83.3554),
    "MIA": (25.7959, -80.2870),
}

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WeatherAPIError(Exception):
    """Raised when a weather API request fails."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class WeatherSnapshot:
    """Structured weather observation for a single airport."""

    airport: str
    description: str
    temp_c: float
    wind_speed_ms: float
    visibility_m: int
    thunderstorm: bool
    precipitation: bool
    low_visibility: bool
    wind_gust: bool
    severity: float
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Severity computation
# ---------------------------------------------------------------------------


def _compute_severity(
    thunderstorm: bool,
    precipitation: bool,
    low_visibility: bool,
    wind_gust: bool,
) -> float:
    """Compute a 0–10 weather severity score.

    Args:
        thunderstorm: Thunderstorm is present.
        precipitation: Non-thunderstorm precipitation present.
        low_visibility: Visibility is below 4800 m.
        wind_gust: Wind speed exceeds 12 m/s.

    Returns:
        Clamped severity score in [0.0, 10.0].
    """
    base = 0.0
    if thunderstorm:
        base += 4.5
    elif precipitation:
        base += 2.0
    if low_visibility:
        base += 2.5
    if wind_gust:
        base += 1.5
    return min(max(base, 0.0), 10.0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_CLIENT_INSTANCE: "WeatherClient | None" = None

_OWM_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


class WeatherClient:
    """Thin wrapper around the OpenWeatherMap current-weather API."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialise the client.

        Args:
            api_key: OpenWeatherMap API key. Falls back to the
                ``OPENWEATHERMAP_API_KEY`` environment variable if not provided.
        """
        self.api_key: str | None = api_key or os.environ.get("OPENWEATHERMAP_API_KEY") or None
        self._configured: bool = bool(self.api_key)

    def is_configured(self) -> bool:
        """Return True when a valid API key is available.

        Returns:
            Configuration status.
        """
        return self._configured

    def get_airport_weather(self, airport_code: str) -> WeatherSnapshot:
        """Fetch current weather for the given airport.

        Args:
            airport_code: IATA airport code, e.g. ``"FLL"``.

        Returns:
            WeatherSnapshot populated from the API response.

        Raises:
            WeatherAPIError: On network failure, bad status code, or
                unknown airport code.
        """
        code = airport_code.upper()
        if code not in AIRPORT_COORDS:
            raise WeatherAPIError(f"Unknown airport code: {airport_code!r}")

        if not self._configured:
            raise WeatherAPIError("OpenWeatherMap API key is not configured.")

        lat, lon = AIRPORT_COORDS[code]
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }

        try:
            response = requests.get(_OWM_BASE_URL, params=params, timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WeatherAPIError(f"Weather API request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise WeatherAPIError(f"Invalid JSON response: {exc}") from exc

        weather_block = data.get("weather", [{}])[0]
        main_block = data.get("main", {})
        wind_block = data.get("wind", {})
        visibility_raw = data.get("visibility", 10000)

        description: str = weather_block.get("description", "unknown")
        weather_id: int = int(weather_block.get("id", 800))
        temp_c: float = float(main_block.get("temp", 20.0))
        wind_speed_ms: float = float(wind_block.get("speed", 0.0))
        visibility_m: int = int(visibility_raw)

        # Derive boolean condition flags
        thunderstorm: bool = 200 <= weather_id < 300
        precipitation: bool = (300 <= weather_id < 700) and not thunderstorm
        low_visibility: bool = visibility_m < 4800
        wind_gust: bool = wind_speed_ms > 12.0

        severity = _compute_severity(thunderstorm, precipitation, low_visibility, wind_gust)

        return WeatherSnapshot(
            airport=code,
            description=description,
            temp_c=temp_c,
            wind_speed_ms=wind_speed_ms,
            visibility_m=visibility_m,
            thunderstorm=thunderstorm,
            precipitation=precipitation,
            low_visibility=low_visibility,
            wind_gust=wind_gust,
            severity=severity,
            raw=data,
        )

    def get_fll_weather(self) -> WeatherSnapshot:
        """Fetch current weather for Fort Lauderdale-Hollywood International (FLL).

        Returns:
            WeatherSnapshot for FLL.

        Raises:
            WeatherAPIError: If the API request fails.
        """
        return self.get_airport_weather("FLL")


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------


def get_client() -> WeatherClient:
    """Return a cached WeatherClient, loading .env first.

    Returns:
        Shared WeatherClient instance.
    """
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        try:
            from dotenv import load_dotenv

            _dotenv_path = Path(__file__).resolve().parents[2] / ".env"
            load_dotenv(dotenv_path=_dotenv_path, override=False)
        except Exception as exc:
            logger.debug("dotenv load skipped: %s", exc)

        _CLIENT_INSTANCE = WeatherClient()
    return _CLIENT_INSTANCE
