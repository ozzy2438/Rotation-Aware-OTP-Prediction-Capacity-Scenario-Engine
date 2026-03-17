"""
Monte Carlo Scenario Simulator for Spirit Airlines FLL Hub capacity planning.

Simulates the impact of adding or removing flights on a route, modelling:
- Passenger redistribution (demand capture from new capacity)
- Load factor changes at network level
- OTP degradation from increased airport congestion
- Revenue impact (RASK × ASK)
- Full probability distributions via Monte Carlo simulation

Usage:
    python src/models/scenario_simulator.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

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

# Spirit Airlines network economics (approximate)
RASK_USD: float = 0.1025           # Revenue per Available Seat-Mile (USD)
CASM_USD: float = 0.0920           # Cost per Available Seat-Mile (USD)
SPIRIT_AVG_SEATS: int = 178        # A320-family average seats
SPIRIT_AVG_STAGE_LENGTH: float = 1200.0  # Network average stage length (miles)

# OTP degradation model coefficients
OTP_CONGESTION_FACTOR: float = 0.003  # OTP loss per additional daily departure at FLL (~0.3pp)

# Passenger demand elasticity to schedule frequency
FREQUENCY_DEMAND_ELASTICITY: float = 0.30  # +10% frequency → +3% demand

ROUTE_CONFIG: dict[str, dict] = {
    "FLL-ATL": {"distance": 581,  "base_lf": 0.87, "base_otp": 0.74, "daily_flights": 6,  "annual_pax": 420_000},
    "FLL-LAS": {"distance": 2316, "base_lf": 0.91, "base_otp": 0.78, "daily_flights": 3,  "annual_pax": 210_000},
    "FLL-LAX": {"distance": 2342, "base_lf": 0.88, "base_otp": 0.76, "daily_flights": 2,  "annual_pax": 148_000},
    "FLL-ORD": {"distance": 1182, "base_lf": 0.85, "base_otp": 0.70, "daily_flights": 4,  "annual_pax": 290_000},
    "FLL-DFW": {"distance": 1235, "base_lf": 0.83, "base_otp": 0.73, "daily_flights": 4,  "annual_pax": 280_000},
    "FLL-MCO": {"distance": 170,  "base_lf": 0.89, "base_otp": 0.77, "daily_flights": 5,  "annual_pax": 350_000},
    "FLL-JFK": {"distance": 1069, "base_lf": 0.86, "base_otp": 0.68, "daily_flights": 3,  "annual_pax": 210_000},
    "FLL-BOS": {"distance": 1240, "base_lf": 0.84, "base_otp": 0.71, "daily_flights": 2,  "annual_pax": 145_000},
    "FLL-DTW": {"distance": 1154, "base_lf": 0.82, "base_otp": 0.72, "daily_flights": 3,  "annual_pax": 195_000},
    "FLL-MIA": {"distance": 21,   "base_lf": 0.78, "base_otp": 0.80, "daily_flights": 2,  "annual_pax": 88_000},
}

SCHEDULE_FACTORS: dict[str, dict] = {
    "all_day":   {"lf_capture": 1.00, "otp_multiplier": 1.00, "label": "All-Day (Spread)"},
    "peak_only": {"lf_capture": 1.22, "otp_multiplier": 1.15, "label": "Peak Hours Only"},
    "off_peak":  {"lf_capture": 0.72, "otp_multiplier": 0.85, "label": "Off-Peak Only"},
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ScenarioInput:
    """Inputs for a Monte Carlo capacity scenario.

    Attributes:
        route: Route identifier (e.g., 'FLL-ATL').
        additional_daily_flights: Number of flights to add (can be negative to remove).
        schedule_change: Timing strategy ('all_day', 'peak_only', 'off_peak').
        simulation_runs: Number of Monte Carlo iterations.
        label: Human-readable scenario name.
    """
    route: str
    additional_daily_flights: int
    schedule_change: str = "all_day"
    simulation_runs: int = 10_000
    label: str = ""

    def __post_init__(self) -> None:
        if self.schedule_change not in SCHEDULE_FACTORS:
            raise ValueError(f"schedule_change must be one of {list(SCHEDULE_FACTORS.keys())}")
        if not self.label:
            delta = self.additional_daily_flights
            sign = "+" if delta >= 0 else ""
            self.label = (
                f"{self.route}: {sign}{delta} daily flight(s), "
                f"{SCHEDULE_FACTORS[self.schedule_change]['label']}"
            )


@dataclass
class ScenarioResult:
    """Results from a Monte Carlo scenario simulation.

    Attributes:
        scenario: The ScenarioInput that produced this result.
        baseline_lf: Historical baseline load factor.
        baseline_otp: Historical baseline OTP.
        mean_lf: Mean simulated load factor.
        mean_otp: Mean simulated OTP.
        mean_revenue_delta_annual: Mean annual revenue change (USD).
        lf_distribution: Array of simulated load factors.
        otp_distribution: Array of simulated OTPs.
        revenue_distribution: Array of simulated annual revenue deltas (USD).
        p10_lf: 10th percentile LF.
        p90_lf: 90th percentile LF.
        p10_revenue: 10th percentile revenue delta.
        p90_revenue: 90th percentile revenue delta.
        recommendation: One-line recommendation string.
    """
    scenario: ScenarioInput
    baseline_lf: float
    baseline_otp: float
    mean_lf: float
    mean_otp: float
    mean_revenue_delta_annual: float
    lf_distribution: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    otp_distribution: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    revenue_distribution: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    p10_lf: float = 0.0
    p90_lf: float = 0.0
    p10_revenue: float = 0.0
    p90_revenue: float = 0.0
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class ScenarioSimulator:
    """Monte Carlo capacity scenario simulator for Spirit FLL hub.

    Models the effects of schedule changes on load factor, OTP, and revenue
    using stochastic simulation with parametric uncertainty.
    """

    def __init__(self, rng_seed: int = 42) -> None:
        """Initialise the simulator.

        Args:
            rng_seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate(self, scenario: ScenarioInput) -> ScenarioResult:
        """Run a Monte Carlo simulation for the given scenario.

        Args:
            scenario: Scenario configuration.

        Returns:
            ScenarioResult with full distributions and summary statistics.
        """
        cfg = ROUTE_CONFIG.get(scenario.route)
        if cfg is None:
            raise ValueError(f"Unknown route: {scenario.route}. "
                             f"Valid routes: {list(ROUTE_CONFIG.keys())}")

        sched = SCHEDULE_FACTORS[scenario.schedule_change]
        n = scenario.simulation_runs

        baseline_lf  = cfg["base_lf"]
        baseline_otp = cfg["base_otp"]
        baseline_daily = cfg["daily_flights"]
        distance = cfg["distance"]

        new_daily = baseline_daily + scenario.additional_daily_flights
        if new_daily <= 0:
            raise ValueError(
                f"Cannot have {new_daily} daily flights. Reduce additional_daily_flights."
            )

        # Frequency ratio: how much capacity increases
        freq_ratio = new_daily / baseline_daily  # e.g. 7/6 = 1.167 for +1 on 6-flight route

        # Frequency-stimulated total demand growth (elasticity model)
        # +10% frequency → +3% passengers  (FREQUENCY_DEMAND_ELASTICITY = 0.30)
        demand_stimulus = 1.0 + FREQUENCY_DEMAND_ELASTICITY * (freq_ratio - 1.0)

        # ── Monte Carlo uncertainty sources ──────────────────────────────
        demand_noise = self.rng.normal(1.0, 0.05, n)   # ±5% demand uncertainty
        yield_noise  = self.rng.normal(1.0, 0.04, n)   # ±4% yield (ticket price)
        cost_noise   = self.rng.normal(1.0, 0.03, n)   # ±3% cost uncertainty

        # ── Load Factor ──────────────────────────────────────────────────
        # The new flight(s) start at a LF determined by:
        #   base proportional share × demand_stimulus × schedule timing factor
        # all_day lf_capture=1.00 → new flight LF ≈ baseline × stim/freq_ratio
        # peak_only lf_capture=1.22 → new flight fills better at peak
        # off_peak  lf_capture=0.72 → new flight fills worse at off-peak
        new_flight_lf_mean = float(np.clip(
            baseline_lf * demand_stimulus / freq_ratio * sched["lf_capture"],
            0.20, 0.97,
        ))

        # Simulate distributions
        sim_new_flight_lf = (new_flight_lf_mean * demand_noise).clip(0.20, 0.98)

        # Existing flights lose a tiny amount of LF due to minor demand redistribution
        existing_dilution = 0.005 * abs(scenario.additional_daily_flights)
        sim_existing_lf = ((baseline_lf - existing_dilution) * demand_noise).clip(0.20, 0.98)

        # Network-average LF: weighted by number of flights
        sim_lf = (
            baseline_daily * sim_existing_lf
            + scenario.additional_daily_flights * sim_new_flight_lf
        ) / new_daily
        sim_lf = sim_lf.clip(0.20, 0.98)

        # ── OTP ──────────────────────────────────────────────────────────
        # Congestion drag: each additional departure at FLL hub ≈ −0.3pp OTP
        otp_congestion_drag = scenario.additional_daily_flights * OTP_CONGESTION_FACTOR
        # Schedule timing: peak hours add congestion, off-peak reduces it
        otp_schedule_adj = (sched["otp_multiplier"] - 1.0) * 0.025
        sim_otp = (
            baseline_otp
            - otp_congestion_drag
            - otp_schedule_adj
            + self.rng.normal(0, 0.012, n)
        )
        sim_otp = np.clip(sim_otp, 0.40, 0.99)

        # ── Revenue ───────────────────────────────────────────────────────
        # Annual ASMs for the incremental flight(s) only
        incremental_asm_annual = (
            abs(scenario.additional_daily_flights) * SPIRIT_AVG_SEATS * distance * 365
        )
        direction = float(np.sign(scenario.additional_daily_flights))

        # Revenue earned on the incremental capacity at projected LF
        incremental_revenue = (
            direction * incremental_asm_annual * RASK_USD * sim_new_flight_lf * yield_noise
        )
        # Operating cost of the incremental capacity (fuel, crew, maintenance)
        incremental_cost = direction * incremental_asm_annual * CASM_USD * cost_noise

        # Marginal revenue change on existing fleet (demand stimulus vs dilution)
        existing_asm_annual = baseline_daily * SPIRIT_AVG_SEATS * distance * 365
        existing_revenue_delta = (
            existing_asm_annual * RASK_USD * (sim_existing_lf - baseline_lf) * yield_noise
        )

        revenue_delta = incremental_revenue - incremental_cost + existing_revenue_delta

        # ── Aggregate statistics ─────────────────────────────────────────
        mean_lf = float(np.mean(sim_lf))
        mean_otp = float(np.mean(sim_otp))
        mean_rev = float(np.mean(revenue_delta))

        p10_lf = float(np.percentile(sim_lf, 10))
        p90_lf = float(np.percentile(sim_lf, 90))
        p10_rev = float(np.percentile(revenue_delta, 10))
        p90_rev = float(np.percentile(revenue_delta, 90))

        recommendation = self._generate_recommendation(
            scenario=scenario,
            baseline_lf=baseline_lf,
            baseline_otp=baseline_otp,
            mean_lf=mean_lf,
            mean_otp=mean_otp,
            mean_rev=mean_rev,
        )

        return ScenarioResult(
            scenario=scenario,
            baseline_lf=baseline_lf,
            baseline_otp=baseline_otp,
            mean_lf=mean_lf,
            mean_otp=mean_otp,
            mean_revenue_delta_annual=mean_rev,
            lf_distribution=sim_lf,
            otp_distribution=sim_otp,
            revenue_distribution=revenue_delta,
            p10_lf=p10_lf,
            p90_lf=p90_lf,
            p10_revenue=p10_rev,
            p90_revenue=p90_rev,
            recommendation=recommendation,
        )

    def compare_scenarios(
        self,
        route: str,
        additional_daily_flights: int = 1,
        simulation_runs: int = 10_000,
    ) -> tuple[pd.DataFrame, list[ScenarioResult]]:
        """Compare all schedule strategies for a given capacity change.

        Runs three scenarios (all_day, peak_only, off_peak) and returns a
        comparison DataFrame plus the full result objects.

        Args:
            route: Route identifier.
            additional_daily_flights: Number of flights to add.
            simulation_runs: Monte Carlo iterations per scenario.

        Returns:
            Tuple of (comparison DataFrame, list of ScenarioResult objects).
        """
        logger.info(
            "Comparing %d scenarios for %s (+%d flights) …",
            3, route, additional_daily_flights,
        )

        results = []
        for schedule_type in SCHEDULE_FACTORS:
            scenario = ScenarioInput(
                route=route,
                additional_daily_flights=additional_daily_flights,
                schedule_change=schedule_type,
                simulation_runs=simulation_runs,
            )
            result = self.simulate(scenario)
            results.append(result)
            logger.info("  %s → LF=%.1f%% OTP=%.1f%% Rev=$%.1fM",
                        schedule_type,
                        result.mean_lf * 100,
                        result.mean_otp * 100,
                        result.mean_revenue_delta_annual / 1e6)

        comparison = self._build_comparison_table(results)
        return comparison, results

    def _build_comparison_table(self, results: list[ScenarioResult]) -> pd.DataFrame:
        """Build a structured comparison DataFrame from scenario results.

        Args:
            results: List of ScenarioResult objects.

        Returns:
            DataFrame with one row per scenario and key metrics.
        """
        cfg = ROUTE_CONFIG.get(results[0].scenario.route, {})
        baseline_daily = cfg.get("daily_flights", 1)

        rows = []
        for r in results:
            sched = SCHEDULE_FACTORS[r.scenario.schedule_change]
            new_daily = baseline_daily + r.scenario.additional_daily_flights
            rows.append({
                "Scenario": sched["label"],
                "Schedule": r.scenario.schedule_change,
                "Daily_Flights_Before": baseline_daily,
                "Daily_Flights_After": new_daily,
                "Baseline_LF_%": round(r.baseline_lf * 100, 1),
                "Projected_LF_%": round(r.mean_lf * 100, 1),
                "LF_Change_pp": round((r.mean_lf - r.baseline_lf) * 100, 1),
                "LF_P10_%": round(r.p10_lf * 100, 1),
                "LF_P90_%": round(r.p90_lf * 100, 1),
                "Baseline_OTP_%": round(r.baseline_otp * 100, 1),
                "Projected_OTP_%": round(r.mean_otp * 100, 1),
                "OTP_Change_pp": round((r.mean_otp - r.baseline_otp) * 100, 1),
                "Annual_Revenue_Delta_M$": round(r.mean_revenue_delta_annual / 1e6, 2),
                "Rev_P10_M$": round(r.p10_revenue / 1e6, 2),
                "Rev_P90_M$": round(r.p90_revenue / 1e6, 2),
                "Recommendation": r.recommendation,
            })

        return pd.DataFrame(rows)

    def _generate_recommendation(
        self,
        scenario: ScenarioInput,
        baseline_lf: float,
        baseline_otp: float,
        mean_lf: float,
        mean_otp: float,
        mean_rev: float,
    ) -> str:
        """Generate a concise trade-off recommendation string.

        Args:
            scenario: The scenario configuration.
            baseline_lf: Historical load factor.
            baseline_otp: Historical OTP.
            mean_lf: Projected mean load factor.
            mean_otp: Projected mean OTP.
            mean_rev: Projected annual revenue delta (USD).

        Returns:
            One-line recommendation string.
        """
        lf_pp = (mean_lf - baseline_lf) * 100
        otp_pp = (mean_otp - baseline_otp) * 100
        rev_m = mean_rev / 1e6
        sched_label = SCHEDULE_FACTORS[scenario.schedule_change]["label"]
        route = scenario.route
        delta = scenario.additional_daily_flights
        sign = "+" if delta >= 0 else ""

        lf_str = f"{sign}{lf_pp:+.1f}pp"
        otp_str = f"{otp_pp:+.1f}pp"
        rev_str = f"${rev_m:+.1f}M"

        if mean_rev > 0 and otp_pp > -3:
            verdict = "RECOMMENDED"
        elif mean_rev > 0 and otp_pp <= -3:
            verdict = "PROCEED WITH CAUTION"
        elif mean_rev <= 0:
            verdict = "NOT RECOMMENDED"
        else:
            verdict = "REVIEW"

        return (
            f"{verdict}: {sign}{delta} daily {route} flight ({sched_label}) → "
            f"LF {lf_str}, OTP {otp_str}, Revenue {rev_str}/yr"
        )

    def generate_narrative(
        self,
        comparison_df: pd.DataFrame,
        route: str,
        additional_daily_flights: int,
    ) -> str:
        """Generate a human-readable narrative summary of scenario comparison.

        Args:
            comparison_df: Output of compare_scenarios()[0].
            route: Route identifier.
            additional_daily_flights: Flights added.

        Returns:
            Multi-line formatted narrative string.
        """
        best_row = comparison_df.loc[
            comparison_df["Annual_Revenue_Delta_M$"].idxmax()
        ]
        lines = [
            f"=== Scenario Analysis: {route} +{additional_daily_flights} Daily Flight(s) ===",
            "",
            "BASELINE STATE:",
            f"  Load Factor:  {best_row['Baseline_LF_%']:.1f}%",
            f"  OTP:          {best_row['Baseline_OTP_%']:.1f}%",
            "",
            "SCENARIO COMPARISON:",
        ]
        for _, row in comparison_df.iterrows():
            lines.append(
                f"  [{row['Schedule'].upper():10}] LF={row['Projected_LF_%']:.1f}% "
                f"({row['LF_Change_pp']:+.1f}pp) | "
                f"OTP={row['Projected_OTP_%']:.1f}% ({row['OTP_Change_pp']:+.1f}pp) | "
                f"Rev={row['Annual_Revenue_Delta_M$']:+.1f}M"
            )
        lines += [
            "",
            "BEST OPTION:",
            f"  {best_row['Recommendation']}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    simulator = ScenarioSimulator(rng_seed=42)

    print("\n" + "=" * 60)
    print("FLL-ATL Capacity Expansion: +1 Daily Flight")
    print("=" * 60)

    comparison, results = simulator.compare_scenarios(
        route="FLL-ATL",
        additional_daily_flights=1,
        simulation_runs=10_000,
    )

    print(comparison[[
        "Scenario", "Projected_LF_%", "LF_Change_pp",
        "Projected_OTP_%", "OTP_Change_pp", "Annual_Revenue_Delta_M$",
    ]].to_string(index=False))

    narrative = simulator.generate_narrative(comparison, "FLL-ATL", 1)
    print("\n" + narrative)

    # Save comparison
    out_dir = PROJECT_ROOT / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(out_dir / "fll_atl_scenario_comparison.parquet", index=False)
    logger.info("Saved scenario comparison → data/models/fll_atl_scenario_comparison.parquet")
