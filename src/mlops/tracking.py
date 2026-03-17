"""MLflow experiment tracking for OTP predictor.

Wraps MLflow to record XGBoost training runs, hyperparameters, metrics,
and model artifacts under the 'spirit-otp-predictor' experiment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRACKER_INSTANCE: "OTPTracker | None" = None


class OTPTracker:
    """MLflow wrapper for OTP predictor experiment tracking."""

    def __init__(
        self,
        experiment_name: str = "spirit-otp-predictor",
        tracking_uri: str | None = None,
    ) -> None:
        """Set up MLflow tracking.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: MLflow tracking URI. Defaults to local mlruns/ folder.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._experiment_id: str | None = None

        try:
            import mlflow

            if self.tracking_uri is None:
                project_root = Path(__file__).resolve().parents[2]
                db_path = project_root / "mlruns" / "mlflow.db"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                self.tracking_uri = f"sqlite:///{db_path}"

            mlflow.set_tracking_uri(self.tracking_uri)

            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self._experiment_id = experiment.experiment_id

            mlflow.set_experiment(self.experiment_name)
            logger.info(
                "MLflow tracking initialised — experiment '%s' (id=%s)",
                self.experiment_name,
                self._experiment_id,
            )
        except Exception as exc:
            logger.warning("MLflow initialisation failed: %s", exc)

    def start_run(self, run_name: str):
        """Return an MLflow active run context manager tagged for the OTP project.

        Args:
            run_name: Human-readable run identifier.

        Returns:
            mlflow.ActiveRun context manager, or a no-op context if MLflow fails.
        """
        try:
            import mlflow

            return mlflow.start_run(
                run_name=run_name,
                tags={
                    "model": "XGBoost-OTP",
                    "project": "Spirit-FLL-Hub",
                },
            )
        except Exception as exc:
            logger.warning("MLflow start_run failed: %s", exc)
            return _NoOpContext()

    def log_params(self, params: dict) -> None:
        """Log XGBoost hyperparameters (int/float/str/bool values only).

        Args:
            params: Dictionary of model hyperparameters.
        """
        try:
            import mlflow

            filtered = {
                k: v
                for k, v in params.items()
                if isinstance(v, (int, float, str, bool))
            }
            if filtered:
                mlflow.log_params(filtered)
        except Exception as exc:
            logger.warning("MLflow log_params failed: %s", exc)

    def log_metrics(self, metrics: dict) -> None:
        """Log evaluation metrics (auc_roc, f1_score, average_precision).

        Args:
            metrics: Dictionary containing metric values.
        """
        try:
            import mlflow

            allowed = {"auc_roc", "f1_score", "average_precision"}
            filtered = {k: float(v) for k, v in metrics.items() if k in allowed}
            if filtered:
                mlflow.log_metrics(filtered)
        except Exception as exc:
            logger.warning("MLflow log_metrics failed: %s", exc)

    def log_feature_count(self, n: int) -> None:
        """Log the number of features used in training.

        Args:
            n: Feature count.
        """
        try:
            import mlflow

            mlflow.log_metric("n_features", int(n))
        except Exception as exc:
            logger.warning("MLflow log_feature_count failed: %s", exc)

    def log_model_artifact(self, model_path: Path) -> None:
        """Log a .pkl model file as an MLflow artifact.

        Args:
            model_path: Path to the serialised model file.
        """
        try:
            import mlflow

            model_path = Path(model_path)
            if model_path.exists():
                mlflow.log_artifact(str(model_path))
            else:
                logger.warning("Model artifact not found, skipping: %s", model_path)
        except Exception as exc:
            logger.warning("MLflow log_model_artifact failed: %s", exc)

    def get_best_run(self) -> dict:
        """Query the experiment for the run with the highest auc_roc.

        Returns:
            Dict with keys run_id, auc_roc, f1_score, params.
            Returns empty dict if no runs exist or MLflow fails.
        """
        try:
            import mlflow

            runs = mlflow.search_runs(
                experiment_names=[self.experiment_name],
                order_by=["metrics.auc_roc DESC"],
                max_results=1,
            )
            if runs.empty:
                return {}

            row = runs.iloc[0]
            return {
                "run_id": row.get("run_id", ""),
                "auc_roc": row.get("metrics.auc_roc", None),
                "f1_score": row.get("metrics.f1_score", None),
                "params": {
                    k.replace("params.", ""): v
                    for k, v in row.items()
                    if str(k).startswith("params.")
                },
            }
        except Exception as exc:
            logger.warning("MLflow get_best_run failed: %s", exc)
            return {}


class _NoOpContext:
    """Fallback context manager used when MLflow is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def get_tracker() -> OTPTracker:
    """Return a cached module-level OTPTracker instance.

    Returns:
        Shared OTPTracker.
    """
    global _TRACKER_INSTANCE
    if _TRACKER_INSTANCE is None:
        _TRACKER_INSTANCE = OTPTracker()
    return _TRACKER_INSTANCE
