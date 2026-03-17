"""
Spirit Airlines FLL Hub: OTP Prediction & Demand-Capacity Scenario Engine
==========================================================================

Main entry point — orchestrates the full pipeline or individual steps.

Commands:
    generate-data   Generate synthetic flight, capacity, and weather data
    run-etl         Load, clean, merge, and register data in DuckDB
    train-models    Train OTP predictor and demand forecaster
    run-simulator   Run scenario simulation for a given route
    launch-dashboard Launch the Streamlit analytics dashboard

Examples:
    python main.py generate-data
    python main.py run-etl
    python main.py train-models
    python main.py run-simulator --route FLL-ATL --flights 1 --schedule peak_only
    python main.py launch-dashboard
    python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import socket
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   Spirit Airlines FLL Hub Analytics                             ║
║   OTP Prediction & Demand-Capacity Scenario Engine v1.0          ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_generate_data(args: argparse.Namespace) -> int:
    """Generate synthetic flight, capacity, and weather datasets.

    Args:
        args: Parsed CLI arguments (unused).

    Returns:
        Exit code (0 = success).
    """
    logger.info("Generating synthetic data ...")
    from data.scripts.generate_synthetic_data import main as gen_main
    gen_main()
    logger.info("Data generation complete. Check data/raw/ for output files.")
    return 0


def cmd_run_etl(args: argparse.Namespace) -> int:
    """Run the ETL pipeline: load, clean, merge, and register in DuckDB.

    Args:
        args: Parsed CLI arguments (unused).

    Returns:
        Exit code (0 = success).
    """
    logger.info("Running ETL pipeline ...")
    from src.pipeline.etl import run_etl

    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not (raw_dir / "flights.csv").exists():
        logger.warning("Raw data not found. Running data generation first ...")
        from data.scripts.generate_synthetic_data import main as gen_main
        gen_main()

    con = run_etl()
    count = con.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
    routes = con.execute("SELECT COUNT(DISTINCT Route) FROM flights").fetchone()[0]
    logger.info("ETL complete. %d flight records across %d routes loaded into DuckDB.", count, routes)
    con.close()
    return 0


def cmd_train_models(args: argparse.Namespace) -> int:
    """Train the OTP predictor and demand forecaster.

    Args:
        args: Parsed CLI arguments (unused).

    Returns:
        Exit code (0 = success).
    """
    logger.info("Training models ...")

    import pandas as pd
    from sklearn.model_selection import train_test_split

    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Check / build features
    feat_path = processed_dir / "flights_features.parquet"
    proc_path = processed_dir / "flights_processed.parquet"

    if not feat_path.exists():
        if not proc_path.exists():
            logger.warning("Processed data not found. Running ETL pipeline ...")
            cmd_run_etl(args)
        logger.info("Building features ...")
        from src.pipeline.features import build_features
        df = pd.read_parquet(proc_path)
        df = build_features(df, include_propagation=True, include_tail_lag=True)
        df.to_parquet(feat_path, index=False)
        logger.info("Feature file saved -> %s", feat_path)
    else:
        df = pd.read_parquet(feat_path)

    # OTP Predictor
    logger.info("Training XGBoost OTP predictor ...")
    from src.models.otp_predictor import OTPPredictor

    predictor = OTPPredictor()
    X, y = predictor.prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train_s, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=42
    )

    predictor.train(X_train, y_train_s, X_val, y_val, calibrate=True)
    metrics = predictor.evaluate(X_test, y_test)
    logger.info(
        "OTP Predictor -- Test AUC-ROC: %.4f | F1: %.4f",
        metrics["auc_roc"], metrics["f1_score"],
    )
    predictor.save()

    # MLflow tracking
    try:
        from src.mlops.tracking import get_tracker
        tracker = get_tracker()
        with tracker.start_run(run_name=f"xgboost-otp-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"):
            tracker.log_params(predictor.params)
            tracker.log_metrics({
                "auc_roc": metrics["auc_roc"],
                "f1_score": metrics["f1_score"],
                "average_precision": metrics["average_precision"],
            })
            tracker.log_feature_count(len(predictor.feature_columns))
            tracker.log_model_artifact(PROJECT_ROOT / "data" / "models" / "otp_predictor.pkl")
        logger.info("MLflow run logged. View with: mlflow ui --port 5000")
    except Exception as exc:
        logger.warning("MLflow tracking skipped: %s", exc)

    # Demand Forecaster
    logger.info("Training demand forecasters ...")
    cap_path = processed_dir / "capacity_processed.parquet"
    if cap_path.exists():
        from src.models.demand_forecaster import NetworkDemandForecaster
        capacity_df = pd.read_parquet(cap_path)
        ndf = NetworkDemandForecaster()
        ndf.fit_all(capacity_df)
        forecasts = ndf.forecast_all(horizon=12)
        models_dir = PROJECT_ROOT / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        forecasts.to_parquet(models_dir / "demand_forecasts.parquet", index=False)
        logger.info("Demand forecasts saved -> data/models/demand_forecasts.parquet")
    else:
        logger.warning("Capacity data not found -- skipping demand forecaster training.")

    logger.info("Model training complete.")
    return 0


def cmd_run_simulator(args: argparse.Namespace) -> int:
    """Run Monte Carlo scenario simulation.

    Args:
        args: Namespace with route, flights, schedule, runs attributes.

    Returns:
        Exit code (0 = success).
    """
    from src.models.scenario_simulator import ScenarioSimulator

    route = args.route
    additional = args.flights
    schedule = args.schedule
    sim_runs = args.runs

    logger.info(
        "Running scenario: %s | +%d flights | %s schedule | %d runs",
        route, additional, schedule, sim_runs,
    )

    simulator = ScenarioSimulator(rng_seed=42)
    comparison, _results = simulator.compare_scenarios(
        route=route,
        additional_daily_flights=additional,
        simulation_runs=sim_runs,
    )

    narrative = simulator.generate_narrative(comparison, route, additional)
    print("\n" + narrative + "\n")

    print("Scenario Comparison Table:")
    print(comparison[[
        "Scenario", "Projected_LF_%", "LF_Change_pp",
        "Projected_OTP_%", "OTP_Change_pp", "Annual_Revenue_Delta_M$",
    ]].to_string(index=False))

    return 0


def _is_port_available(port: int) -> bool:
    """Return True when the given localhost TCP port can be bound."""
    # Detect an active listener first, then verify we can claim the port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _resolve_dashboard_port(preferred_port: int, max_attempts: int = 20) -> int:
    """Pick the preferred port or the next available one."""
    for offset in range(max_attempts):
        candidate = preferred_port + offset
        if _is_port_available(candidate):
            return candidate
    raise RuntimeError(
        f"No available dashboard port found between {preferred_port} and "
        f"{preferred_port + max_attempts - 1}."
    )


def cmd_mlops_status(args: argparse.Namespace) -> int:
    """Print current MLOps pipeline status: model registry, best run, data freshness.

    Args:
        args: Parsed CLI arguments (unused).

    Returns:
        Exit code (0 = success).
    """
    import json
    from datetime import datetime

    print("\n" + "=" * 62)
    print("  Spirit Airlines FLL Hub — MLOps Status")
    print("=" * 62)

    # -- Trained model --
    meta_path = PROJECT_ROOT / "data" / "models" / "otp_predictor_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        er   = meta.get("eval_results", {})
        feats = meta.get("feature_columns", [])
        mtime = datetime.fromtimestamp(meta_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"\n  Model            XGBoost OTP Predictor")
        print(f"  Last trained     {mtime}")
        print(f"  AUC-ROC          {er.get('auc_roc', '—')}")
        print(f"  F1 Score         {er.get('f1_score', '—')}")
        print(f"  Avg Precision    {er.get('average_precision', '—')}")
        print(f"  Features         {len(feats)}")
    else:
        print("\n  Model            NOT FOUND — run: python main.py train-models")

    # -- MLflow best run --
    print()
    try:
        from src.mlops.tracking import get_tracker
        best = get_tracker().get_best_run()
        if best:
            print(f"  MLflow best run  {best.get('run_id', '')[:12]}...")
            print(f"  Best AUC-ROC     {best.get('auc_roc', '—')}")
            print(f"  Best F1          {best.get('f1_score', '—')}")
        else:
            print("  MLflow           No runs recorded yet")
        print(f"  MLflow UI        mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000")
    except Exception as exc:
        print(f"  MLflow           Unavailable ({exc})")

    # -- Data freshness --
    print()
    data_files = {
        "flights_processed.parquet": PROJECT_ROOT / "data" / "processed" / "flights_processed.parquet",
        "capacity_processed.parquet": PROJECT_ROOT / "data" / "processed" / "capacity_processed.parquet",
        "spirit_otp.duckdb": PROJECT_ROOT / "data" / "spirit_otp.duckdb",
        "demand_forecasts.parquet": PROJECT_ROOT / "data" / "models" / "demand_forecasts.parquet",
    }
    for name, path in data_files.items():
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size  = path.stat().st_size / 1024
            print(f"  {name:<32}  {mtime}  ({size:.0f} KB)")
        else:
            print(f"  {name:<32}  MISSING")

    # -- CI/CD --
    print()
    gha_path = PROJECT_ROOT / ".github" / "workflows" / "retrain_pipeline.yml"
    if gha_path.exists():
        print("  GitHub Actions   .github/workflows/retrain_pipeline.yml  ✓")
        print("  Schedule         Every Monday 03:00 UTC (+ manual trigger)")
    else:
        print("  GitHub Actions   NOT CONFIGURED")

    print("\n" + "=" * 62 + "\n")
    return 0


def cmd_launch_dashboard(args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard.

    Args:
        args: Namespace with port attribute.

    Returns:
        Exit code (0 = success).
    """
    app_path = PROJECT_ROOT / "src" / "dashboard" / "app.py"
    requested_port = getattr(args, "port", 8501)
    port = _resolve_dashboard_port(requested_port)

    if port != requested_port:
        logger.warning(
            "Port %d is busy. Launching Streamlit dashboard on port %d instead.",
            requested_port,
            port,
        )
    else:
        logger.info("Launching Streamlit dashboard on port %d ...", port)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "false",
    ]
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="spirit-otp",
        description=(
            "Spirit Airlines FLL Hub: OTP Prediction & Scenario Engine\n"
            "Demonstration project for Jetstar Airlines data analyst application."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py generate-data\n"
            "  python main.py run-etl\n"
            "  python main.py train-models\n"
            "  python main.py run-simulator --route FLL-ATL --flights 1 --schedule peak_only\n"
            "  python main.py launch-dashboard\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # generate-data
    sub.add_parser(
        "generate-data",
        help="Generate synthetic Spirit Airlines flight, capacity, and weather data",
    )

    # run-etl
    sub.add_parser(
        "run-etl",
        help="Run the ETL pipeline: clean, merge, and load data into DuckDB",
    )

    # train-models
    sub.add_parser(
        "train-models",
        help="Train XGBoost OTP predictor and SARIMAX demand forecasters",
    )

    # run-simulator
    sim_parser = sub.add_parser(
        "run-simulator",
        help="Run Monte Carlo capacity scenario simulation",
    )
    sim_parser.add_argument(
        "--route", "-r",
        default="FLL-ATL",
        choices=["FLL-ATL", "FLL-LAS", "FLL-LAX", "FLL-ORD", "FLL-DFW",
                 "FLL-MCO", "FLL-JFK", "FLL-BOS", "FLL-DTW", "FLL-MIA"],
        help="Route to simulate (default: FLL-ATL)",
    )
    sim_parser.add_argument(
        "--flights", "-f",
        type=int, default=1,
        help="Number of additional daily flights (default: 1; negative = removal)",
    )
    sim_parser.add_argument(
        "--schedule", "-s",
        default="peak_only",
        choices=["all_day", "peak_only", "off_peak"],
        help="Schedule strategy (default: peak_only)",
    )
    sim_parser.add_argument(
        "--runs", "-n",
        type=int, default=10_000,
        help="Number of Monte Carlo simulation runs (default: 10000)",
    )

    # mlops-status
    sub.add_parser(
        "mlops-status",
        help="Print MLOps pipeline status: model registry, MLflow runs, data freshness",
    )

    # launch-dashboard
    dash_parser = sub.add_parser(
        "launch-dashboard",
        help="Launch the Streamlit analytics dashboard",
    )
    dash_parser.add_argument(
        "--port", "-p",
        type=int, default=8501,
        help="Port to run Streamlit on (default: 8501)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

COMMAND_MAP = {
    "generate-data":  cmd_generate_data,
    "run-etl":        cmd_run_etl,
    "train-models":   cmd_train_models,
    "run-simulator":  cmd_run_simulator,
    "mlops-status":   cmd_mlops_status,
    "launch-dashboard": cmd_launch_dashboard,
}


def main() -> int:
    """Parse CLI arguments and dispatch to the appropriate command handler.

    Returns:
        Integer exit code.
    """
    print(BANNER)
    parser = build_parser()
    args = parser.parse_args()

    handler = COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        return 130
    except Exception as exc:
        logger.error("Command '%s' failed: %s", args.command, exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
