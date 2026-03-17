"""
Time series demand forecaster for Spirit Airlines FLL Hub routes.

Uses statsmodels SARIMAX for monthly passenger demand forecasting with:
- Seasonal decomposition (trend + seasonality + residual)
- External regressors: fuel price proxy, economic indicator proxy
- 12-month ahead forecasts with 80% / 95% confidence intervals
- Route-level load factor forecasting

Usage:
    python src/models/demand_forecaster.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ROUTES: list[str] = [
    "FLL-ATL", "FLL-LAS", "FLL-LAX", "FLL-ORD", "FLL-DFW",
    "FLL-MCO", "FLL-JFK", "FLL-BOS", "FLL-DTW", "FLL-MIA",
]


# ---------------------------------------------------------------------------
# External regressors (synthetic proxies)
# ---------------------------------------------------------------------------

def _generate_fuel_price_index(months: pd.DatetimeIndex) -> pd.Series:
    """Generate a synthetic jet fuel price index (random walk with mean reversion).

    Args:
        months: DatetimeIndex of monthly periods.

    Returns:
        Series of fuel price index values (base = 100).
    """
    rng = np.random.default_rng(99)
    n = len(months)
    log_ret = rng.normal(0.002, 0.045, n)  # monthly log returns
    log_price = np.cumsum(log_ret)
    # Mean reversion toward 0
    for i in range(1, n):
        log_price[i] = log_price[i - 1] * 0.97 + log_ret[i]
    fuel_index = 100 * np.exp(log_price)
    return pd.Series(fuel_index, index=months, name="fuel_index")


def _generate_economic_index(months: pd.DatetimeIndex) -> pd.Series:
    """Generate a synthetic consumer confidence / economic index.

    Args:
        months: DatetimeIndex of monthly periods.

    Returns:
        Series of economic index values (base = 100).
    """
    rng = np.random.default_rng(77)
    n = len(months)
    trend = np.linspace(100, 108, n)  # mild upward trend
    cycle = 3 * np.sin(2 * np.pi * np.arange(n) / 36)  # ~3yr business cycle
    noise = rng.normal(0, 1.5, n)
    econ = trend + cycle + noise
    return pd.Series(econ, index=months, name="econ_index")


# ---------------------------------------------------------------------------
# RouteDemandForecaster
# ---------------------------------------------------------------------------

class RouteDemandForecaster:
    """Monthly passenger demand forecaster for a single Spirit Airlines route.

    Fits a SARIMAX(1,1,1)(1,1,1,12) model with external regressors.
    Falls back to Holt-Winters ETS if SARIMAX does not converge.

    Attributes:
        route: Route string (e.g., 'FLL-ATL').
        model_fit: Fitted statsmodels model result object.
        history: Historical monthly passenger series used for training.
        seats_history: Historical monthly seats series.
    """

    def __init__(self, route: str) -> None:
        """Initialise the forecaster for a specific route.

        Args:
            route: Route identifier (e.g., 'FLL-ATL').
        """
        self.route = route
        self.model_fit: Optional[Any] = None
        self.history: Optional[pd.Series] = None
        self.seats_history: Optional[pd.Series] = None
        self._use_ets_fallback: bool = False

    def fit(self, capacity_df: pd.DataFrame) -> "RouteDemandForecaster":
        """Fit demand model on historical monthly capacity data.

        Args:
            capacity_df: Monthly capacity DataFrame with columns
                         [Route, YearMonth, Passengers, Seats].

        Returns:
            Self (for method chaining).
        """
        route_data = (
            capacity_df[capacity_df["Route"] == self.route]
            .sort_values(["Year", "Month"])
            .copy()
        )
        if len(route_data) < 12:
            logger.warning("Insufficient data for route %s (%d months)", self.route, len(route_data))
            return self

        # Build monthly DatetimeIndex
        route_data["period"] = pd.to_datetime(route_data["YearMonth"])
        route_data = route_data.set_index("period").sort_index()

        pax = route_data["Passengers"].astype(float)
        seats = route_data["Seats"].astype(float)

        self.history = pax
        self.seats_history = seats

        # Build external regressors for training period
        months_idx = pax.index
        fuel = _generate_fuel_price_index(months_idx)
        econ = _generate_economic_index(months_idx)
        exog = pd.DataFrame({"fuel": fuel.values, "econ": econ.values}, index=months_idx)

        logger.info("Fitting demand model for %s (%d months) …", self.route, len(pax))

        try:
            sarimax_model = SARIMAX(
                pax,
                exog=exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.model_fit = sarimax_model.fit(disp=False, maxiter=200)
            self._use_ets_fallback = False
            logger.info("SARIMAX converged for %s", self.route)
        except Exception as exc:
            logger.warning("SARIMAX failed for %s (%s); using ETS fallback", self.route, exc)
            ets_model = ExponentialSmoothing(
                pax,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            )
            self.model_fit = ets_model.fit(optimized=True)
            self._use_ets_fallback = True

        return self

    def forecast(self, horizon: int = 12) -> pd.DataFrame:
        """Generate demand forecasts for the next N months.

        Args:
            horizon: Number of months ahead to forecast.

        Returns:
            DataFrame with columns: period, forecast_passengers,
            lower_80, upper_80, lower_95, upper_95, forecast_lf.
        """
        if self.model_fit is None:
            raise RuntimeError(f"Model not fitted for route {self.route}. Call fit() first.")

        last_date = self.history.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )

        if self._use_ets_fallback:
            pred = self.model_fit.forecast(horizon)
            fc_values = pred.values
            # ETS confidence intervals: approximate with ±1.28σ and ±1.96σ
            resid_std = float(np.std(self.model_fit.resid))
            lower_80 = fc_values - 1.28 * resid_std * np.sqrt(np.arange(1, horizon + 1))
            upper_80 = fc_values + 1.28 * resid_std * np.sqrt(np.arange(1, horizon + 1))
            lower_95 = fc_values - 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
            upper_95 = fc_values + 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
        else:
            # Future exogenous regressors (continuation of random walk)
            all_months = pd.date_range(
                start=self.history.index[0],
                periods=len(self.history) + horizon,
                freq="MS",
            )
            fuel_all = _generate_fuel_price_index(all_months)
            econ_all = _generate_economic_index(all_months)
            future_exog = pd.DataFrame({
                "fuel": fuel_all.values[-horizon:],
                "econ": econ_all.values[-horizon:],
            })
            pred = self.model_fit.get_forecast(steps=horizon, exog=future_exog)
            summary = pred.summary_frame(alpha=0.20)  # 80% CI
            fc_values = summary["mean"].values
            lower_80 = summary["mean_ci_lower"].values
            upper_80 = summary["mean_ci_upper"].values
            summary_95 = self.model_fit.get_forecast(steps=horizon, exog=future_exog).summary_frame(alpha=0.05)
            lower_95 = summary_95["mean_ci_lower"].values
            upper_95 = summary_95["mean_ci_upper"].values

        # Clip negatives
        fc_values = np.maximum(fc_values, 0)
        lower_80 = np.maximum(lower_80, 0)
        lower_95 = np.maximum(lower_95, 0)

        # Estimate seats (assume ~same avg seats as last 3 months)
        avg_seats_per_pax = (self.seats_history.iloc[-3:] / self.history.iloc[-3:]).mean()
        forecast_seats = fc_values * avg_seats_per_pax
        forecast_lf = np.where(forecast_seats > 0, fc_values / forecast_seats, np.nan)

        result = pd.DataFrame({
            "period": future_dates,
            "route": self.route,
            "forecast_passengers": fc_values.round(0).astype(int),
            "lower_80": lower_80.round(0).astype(int),
            "upper_80": upper_80.round(0).astype(int),
            "lower_95": lower_95.round(0).astype(int),
            "upper_95": upper_95.round(0).astype(int),
            "forecast_seats": forecast_seats.round(0).astype(int),
            "forecast_lf": np.round(forecast_lf, 4),
        })
        return result

    def decompose(self) -> Optional[object]:
        """Perform seasonal decomposition on the historical series.

        Returns:
            statsmodels DecomposeResult or None if insufficient data.
        """
        if self.history is None or len(self.history) < 24:
            return None
        try:
            result = seasonal_decompose(
                self.history,
                model="additive",
                period=12,
                extrapolate_trend="freq",
            )
            return result
        except Exception as exc:
            logger.warning("Decomposition failed for %s: %s", self.route, exc)
            return None


# ---------------------------------------------------------------------------
# Fleet-level forecaster
# ---------------------------------------------------------------------------

class NetworkDemandForecaster:
    """Fits and forecasts demand for all FLL hub routes.

    Attributes:
        forecasters: Dict mapping route → RouteDemandForecaster.
    """

    def __init__(self) -> None:
        """Initialise an empty network forecaster."""
        self.forecasters: dict[str, RouteDemandForecaster] = {}

    def fit_all(self, capacity_df: pd.DataFrame) -> "NetworkDemandForecaster":
        """Fit demand forecasters for all routes present in capacity data.

        Args:
            capacity_df: Monthly capacity DataFrame.

        Returns:
            Self.
        """
        routes_in_data = capacity_df["Route"].unique()
        for route in routes_in_data:
            logger.info("Fitting demand model: %s", route)
            forecaster = RouteDemandForecaster(route)
            forecaster.fit(capacity_df)
            self.forecasters[route] = forecaster
        logger.info("Fitted %d route forecasters", len(self.forecasters))
        return self

    def forecast_all(self, horizon: int = 12) -> pd.DataFrame:
        """Generate forecasts for all fitted routes.

        Args:
            horizon: Months ahead to forecast.

        Returns:
            Concatenated DataFrame with forecasts for all routes.
        """
        frames = []
        for route, forecaster in self.forecasters.items():
            try:
                fc = forecaster.forecast(horizon)
                frames.append(fc)
            except Exception as exc:
                logger.error("Forecast failed for %s: %s", route, exc)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_route_forecaster(self, route: str) -> RouteDemandForecaster:
        """Get the fitted RouteDemandForecaster for a specific route.

        Args:
            route: Route identifier.

        Returns:
            Fitted RouteDemandForecaster.

        Raises:
            KeyError: If route has not been fitted.
        """
        if route not in self.forecasters:
            raise KeyError(f"Route {route} not found. Call fit_all() first.")
        return self.forecasters[route]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from typing import Any

    cap_path = PROCESSED_DIR / "capacity_processed.parquet"
    if not cap_path.exists():
        logger.error("Capacity data not found at %s. Run etl.py first.", cap_path)
        import sys
        sys.exit(1)

    capacity_df = pd.read_parquet(cap_path)

    ndf = NetworkDemandForecaster()
    ndf.fit_all(capacity_df)

    forecasts = ndf.forecast_all(horizon=12)
    print(forecasts.to_string(index=False))

    out = MODEL_DIR / "demand_forecasts.parquet"
    forecasts.to_parquet(out, index=False)
    logger.info("Saved demand forecasts → %s", out)
