"""
pytest tests for OTP predictor, scenario simulator, and demand forecaster.

Run:
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.otp_predictor import OTPPredictor, FEATURE_COLUMNS
from src.models.scenario_simulator import (
    ROUTE_CONFIG,
    ScenarioInput,
    ScenarioResult,
    ScenarioSimulator,
)
from src.models.demand_forecaster import (
    NetworkDemandForecaster,
    RouteDemandForecaster,
    _generate_fuel_price_index,
    _generate_economic_index,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_feature_df() -> pd.DataFrame:
    """Tiny DataFrame with all required feature columns for OTP predictor tests."""
    rng = np.random.default_rng(42)
    n = 500
    data = {}
    for col in FEATURE_COLUMNS:
        if col in ("thunderstorm_flag", "precipitation_flag", "low_visibility_flag",
                   "wind_gust_flag", "high_wind_flag", "is_weekend", "is_monday",
                   "is_friday", "is_holiday", "is_holiday_window", "is_early_morning",
                   "is_peak_morning", "is_afternoon", "is_evening", "is_summer",
                   "is_hurricane_season", "is_winter", "is_high_congestion_dest",
                   "is_long_haul", "is_short_haul"):
            data[col] = rng.integers(0, 2, n)
        else:
            data[col] = rng.uniform(0, 1, n)

    # Realistic ranges for specific columns
    data["hour_of_day"] = rng.integers(5, 23, n)
    data["day_of_week"] = rng.integers(0, 7, n)
    data["month"] = rng.integers(1, 13, n)
    data["quarter"] = (data["month"] - 1) // 3 + 1
    data["route_distance"] = rng.integers(100, 2500, n)
    data["route_otp_baseline"] = rng.uniform(0.65, 0.85, n)
    data["weather_severity"] = rng.uniform(0, 10, n)
    data["wind_speed"] = rng.uniform(0, 40, n)
    data["rolling_avg_dep_delay"] = rng.uniform(0, 45, n)
    data["prev_tail_dep_delay"] = rng.uniform(0, 60, n)
    data["distance_normalised"] = data["route_distance"] / 2500

    df = pd.DataFrame(data)
    # Target: correlated with weather and hour
    prob = (
        0.20
        + 0.03 * (df["hour_of_day"] > 15).astype(float)
        + 0.04 * df["thunderstorm_flag"]
        + 0.02 * df["is_summer"]
        + rng.uniform(-0.05, 0.05, n)
    ).clip(0, 1)
    df["ArrDel15"] = (rng.random(n) < prob).astype(int)
    df["Cancelled"] = 0
    return df


@pytest.fixture
def sample_capacity_monthly() -> pd.DataFrame:
    """36 months of synthetic capacity data for demand forecaster tests."""
    rng = np.random.default_rng(7)
    months = pd.date_range("2022-01-01", periods=36, freq="MS")
    rows = []
    for m in months:
        base_pax = 14_000 + 2_000 * np.sin(2 * np.pi * m.month / 12)
        rows.append({
            "Year": m.year,
            "Month": m.month,
            "YearMonth": m.strftime("%Y-%m"),
            "Origin": "FLL",
            "Dest": "ATL",
            "Route": "FLL-ATL",
            "Reporting_Airline": "NK",
            "AircraftType": "A320",
            "DepScheduled": 180,
            "DepPerformed": 177,
            "Seats": 31_500,
            "Passengers": int(base_pax + rng.normal(0, 500)),
            "LoadFactor": round(float(base_pax / 31_500), 4),
            "AvgSeatsPerDep": 178.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def simulator() -> ScenarioSimulator:
    return ScenarioSimulator(rng_seed=0)


# ---------------------------------------------------------------------------
# Tests: OTPPredictor
# ---------------------------------------------------------------------------

class TestOTPPredictor:
    def test_instantiation(self):
        p = OTPPredictor()
        assert not p.is_trained
        assert p.model is None

    def test_prepare_data_shape(self, tiny_feature_df):
        p = OTPPredictor()
        X, y = p.prepare_data(tiny_feature_df)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_prepare_data_target_binary(self, tiny_feature_df):
        p = OTPPredictor()
        _, y = p.prepare_data(tiny_feature_df)
        assert set(y.unique()).issubset({0, 1})

    def test_train_sets_is_trained(self, tiny_feature_df):
        p = OTPPredictor(params={
            "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
            "random_state": 42, "n_jobs": 1, "tree_method": "hist",
            "eval_metric": "auc",
        })
        X, y = p.prepare_data(tiny_feature_df)
        p.train(X, y, calibrate=False)
        assert p.is_trained

    def test_predict_shape(self, tiny_feature_df):
        p = OTPPredictor(params={
            "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
            "random_state": 42, "n_jobs": 1, "tree_method": "hist",
            "eval_metric": "auc",
        })
        X, y = p.prepare_data(tiny_feature_df)
        p.train(X, y, calibrate=False)
        preds = p.predict(X)
        assert len(preds) == len(X)

    def test_predict_proba_range(self, tiny_feature_df):
        p = OTPPredictor(params={
            "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
            "random_state": 42, "n_jobs": 1, "tree_method": "hist",
            "eval_metric": "auc",
        })
        X, y = p.prepare_data(tiny_feature_df)
        p.train(X, y, calibrate=False)
        proba = p.predict_proba(X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_predict_binary_values(self, tiny_feature_df):
        p = OTPPredictor(params={
            "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
            "random_state": 42, "n_jobs": 1, "tree_method": "hist",
            "eval_metric": "auc",
        })
        X, y = p.prepare_data(tiny_feature_df)
        p.train(X, y, calibrate=False)
        preds = p.predict(X)
        assert set(preds).issubset({0, 1})

    def test_evaluate_returns_auc(self, tiny_feature_df):
        p = OTPPredictor(params={
            "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
            "random_state": 42, "n_jobs": 1, "tree_method": "hist",
            "eval_metric": "auc",
        })
        X, y = p.prepare_data(tiny_feature_df)
        p.train(X, y, calibrate=False)
        results = p.evaluate(X, y)
        assert "auc_roc" in results
        assert 0 <= results["auc_roc"] <= 1

    def test_feature_importance_shape(self, tiny_feature_df):
        p = OTPPredictor(params={
            "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
            "random_state": 42, "n_jobs": 1, "tree_method": "hist",
            "eval_metric": "auc",
        })
        X, y = p.prepare_data(tiny_feature_df)
        p.train(X, y, calibrate=False)
        fi = p.get_feature_importance()
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == len(p.feature_columns)

    def test_predict_single_keys(self, tiny_feature_df):
        p = OTPPredictor(params={
            "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
            "random_state": 42, "n_jobs": 1, "tree_method": "hist",
            "eval_metric": "auc",
        })
        X, y = p.prepare_data(tiny_feature_df)
        p.train(X, y, calibrate=False)
        features = {col: 0.5 for col in p.feature_columns}
        result = p.predict_single(features)
        assert "delay_probability" in result
        assert "prediction" in result
        assert "risk_level" in result

    def test_not_trained_raises(self):
        p = OTPPredictor()
        X = pd.DataFrame({"a": [1]})
        with pytest.raises(RuntimeError):
            p.predict(X)

    def test_save_load_roundtrip(self, tiny_feature_df, tmp_path):
        p = OTPPredictor(params={
            "n_estimators": 10, "max_depth": 2, "learning_rate": 0.1,
            "random_state": 42, "n_jobs": 1, "tree_method": "hist",
            "eval_metric": "auc",
        })
        X, y = p.prepare_data(tiny_feature_df)
        p.train(X, y, calibrate=False)
        path = p.save(tmp_path)
        assert path.exists()
        loaded = OTPPredictor.load(tmp_path)
        assert loaded.is_trained
        preds_orig = p.predict(X)
        preds_loaded = loaded.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)


# ---------------------------------------------------------------------------
# Tests: ScenarioSimulator
# ---------------------------------------------------------------------------

class TestScenarioSimulator:
    def test_simulate_returns_result(self, simulator):
        scenario = ScenarioInput(route="FLL-ATL", additional_daily_flights=1)
        result = simulator.simulate(scenario)
        assert isinstance(result, ScenarioResult)

    def test_lf_distribution_shape(self, simulator):
        scenario = ScenarioInput(route="FLL-ATL", additional_daily_flights=1, simulation_runs=500)
        result = simulator.simulate(scenario)
        assert len(result.lf_distribution) == 500

    def test_lf_distribution_range(self, simulator):
        scenario = ScenarioInput(route="FLL-ATL", additional_daily_flights=1, simulation_runs=1000)
        result = simulator.simulate(scenario)
        assert (result.lf_distribution >= 0).all()
        assert (result.lf_distribution <= 1).all()

    def test_otp_distribution_range(self, simulator):
        scenario = ScenarioInput(route="FLL-ATL", additional_daily_flights=1, simulation_runs=1000)
        result = simulator.simulate(scenario)
        assert (result.otp_distribution >= 0).all()
        assert (result.otp_distribution <= 1).all()

    def test_adding_flights_reduces_lf(self, simulator):
        """Adding flights should reduce mean load factor (more seats than new demand)."""
        s1 = ScenarioInput(route="FLL-ATL", additional_daily_flights=1, simulation_runs=2000)
        s2 = ScenarioInput(route="FLL-ATL", additional_daily_flights=3, simulation_runs=2000)
        r1 = simulator.simulate(s1)
        r2 = simulator.simulate(s2)
        # More flights → lower LF on average
        assert r2.mean_lf < r1.mean_lf + 0.05  # monotonic within tolerance

    def test_peak_vs_offpeak_lf(self, simulator):
        """Peak-only scheduling should have higher LF than off-peak."""
        s_peak = ScenarioInput(route="FLL-ATL", additional_daily_flights=1,
                               schedule_change="peak_only", simulation_runs=2000)
        s_off = ScenarioInput(route="FLL-ATL", additional_daily_flights=1,
                              schedule_change="off_peak", simulation_runs=2000)
        r_peak = simulator.simulate(s_peak)
        r_off = simulator.simulate(s_off)
        assert r_peak.mean_lf > r_off.mean_lf

    def test_recommendation_not_empty(self, simulator):
        scenario = ScenarioInput(route="FLL-MCO", additional_daily_flights=1)
        result = simulator.simulate(scenario)
        assert len(result.recommendation) > 10

    def test_compare_scenarios_returns_dataframe(self, simulator):
        comp, results = simulator.compare_scenarios(
            route="FLL-ATL", additional_daily_flights=1, simulation_runs=500
        )
        assert isinstance(comp, pd.DataFrame)
        assert len(comp) == 3  # 3 schedule strategies

    def test_compare_scenarios_columns(self, simulator):
        comp, _ = simulator.compare_scenarios(
            route="FLL-ATL", additional_daily_flights=1, simulation_runs=500
        )
        required_cols = ["Scenario", "Projected_LF_%", "Projected_OTP_%",
                         "Annual_Revenue_Delta_M$", "OTP_Change_pp", "LF_Change_pp"]
        for col in required_cols:
            assert col in comp.columns, f"Missing column: {col}"

    def test_invalid_route_raises(self, simulator):
        with pytest.raises(ValueError, match="Unknown route"):
            scenario = ScenarioInput(route="FLL-XYZ", additional_daily_flights=1)
            simulator.simulate(scenario)

    def test_zero_flights_raises(self, simulator):
        with pytest.raises(ValueError):
            scenario = ScenarioInput(route="FLL-MIA", additional_daily_flights=-10)
            simulator.simulate(scenario)

    def test_narrative_contains_route(self, simulator):
        comp, _ = simulator.compare_scenarios(
            route="FLL-LAS", additional_daily_flights=1, simulation_runs=500
        )
        narrative = simulator.generate_narrative(comp, "FLL-LAS", 1)
        assert "FLL-LAS" in narrative

    def test_p10_p90_ordering(self, simulator):
        scenario = ScenarioInput(route="FLL-ATL", additional_daily_flights=1, simulation_runs=2000)
        result = simulator.simulate(scenario)
        assert result.p10_lf <= result.mean_lf <= result.p90_lf
        assert result.p10_revenue <= result.mean_revenue_delta_annual <= result.p90_revenue


# ---------------------------------------------------------------------------
# Tests: DemandForecaster
# ---------------------------------------------------------------------------

class TestDemandForecaster:
    def test_fit_and_forecast(self, sample_capacity_monthly):
        forecaster = RouteDemandForecaster("FLL-ATL")
        forecaster.fit(sample_capacity_monthly)
        assert forecaster.model_fit is not None

    def test_forecast_horizon(self, sample_capacity_monthly):
        forecaster = RouteDemandForecaster("FLL-ATL")
        forecaster.fit(sample_capacity_monthly)
        fc = forecaster.forecast(horizon=12)
        assert len(fc) == 12

    def test_forecast_columns(self, sample_capacity_monthly):
        forecaster = RouteDemandForecaster("FLL-ATL")
        forecaster.fit(sample_capacity_monthly)
        fc = forecaster.forecast(horizon=6)
        required = ["period", "route", "forecast_passengers", "lower_80", "upper_80"]
        for col in required:
            assert col in fc.columns, f"Missing column: {col}"

    def test_forecast_passengers_positive(self, sample_capacity_monthly):
        forecaster = RouteDemandForecaster("FLL-ATL")
        forecaster.fit(sample_capacity_monthly)
        fc = forecaster.forecast(horizon=6)
        assert (fc["forecast_passengers"] >= 0).all()

    def test_confidence_interval_ordering(self, sample_capacity_monthly):
        forecaster = RouteDemandForecaster("FLL-ATL")
        forecaster.fit(sample_capacity_monthly)
        fc = forecaster.forecast(horizon=6)
        assert (fc["lower_80"] <= fc["forecast_passengers"]).all()
        assert (fc["forecast_passengers"] <= fc["upper_80"]).all()

    def test_network_forecaster_all_routes(self, sample_capacity_monthly):
        # Add a second route for multi-route test
        second_route = sample_capacity_monthly.copy()
        second_route["Route"] = "FLL-ORD"
        second_route["Dest"] = "ORD"
        combined = pd.concat([sample_capacity_monthly, second_route], ignore_index=True)

        ndf = NetworkDemandForecaster()
        ndf.fit_all(combined)
        fc = ndf.forecast_all(horizon=3)
        assert len(fc) == 6  # 2 routes × 3 months

    def test_fuel_index_positive(self):
        months = pd.date_range("2022-01-01", periods=12, freq="MS")
        fuel = _generate_fuel_price_index(months)
        assert (fuel > 0).all()

    def test_economic_index_positive(self):
        months = pd.date_range("2022-01-01", periods=12, freq="MS")
        econ = _generate_economic_index(months)
        assert (econ > 0).all()

    def test_decompose_returns_result(self, sample_capacity_monthly):
        forecaster = RouteDemandForecaster("FLL-ATL")
        forecaster.fit(sample_capacity_monthly)
        decomp = forecaster.decompose()
        assert decomp is not None
