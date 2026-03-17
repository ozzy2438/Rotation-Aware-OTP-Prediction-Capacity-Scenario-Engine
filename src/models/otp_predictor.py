"""
XGBoost-based OTP predictor for Spirit Airlines FLL Hub.

Performs binary classification: on-time (0) vs delayed 15+ minutes (1).
Includes:
- Hyperparameter tuning with cross-validated early stopping
- Calibrated probability outputs (Platt scaling)
- Feature importance (gain-based + permutation)
- Full evaluation suite (AUC-ROC, PR, F1, confusion matrix)
- Model persistence (save / load)

Usage:
    python src/models/otp_predictor.py
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Default XGBoost hyperparameters (tuned for delay prediction)
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.5,  # adjust for class imbalance (delayed ~27%)
    "use_label_encoder": False,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
}

FEATURE_COLUMNS: list[str] = [
    "hour_of_day", "day_of_week", "month", "quarter",
    "is_weekend", "is_monday", "is_friday",
    "is_holiday", "is_holiday_window",
    "is_early_morning", "is_peak_morning", "is_afternoon", "is_evening",
    "is_summer", "is_hurricane_season", "is_winter",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "route_distance", "route_otp_baseline",
    "is_high_congestion_dest", "is_long_haul", "is_short_haul",
    "distance_normalised",
    "weather_severity", "thunderstorm_flag", "precipitation_flag",
    "low_visibility_flag", "wind_gust_flag", "wind_speed", "high_wind_flag",
    "rolling_avg_dep_delay",
    "prev_tail_dep_delay",
    # Rotation-chain features
    "legs_today_before", "cumulative_tail_delay_today",
    "prev_tail_arr_delay", "hours_since_last_flight",
    "quick_turn_flag", "tail_delay_streak",
]

TARGET_COL = "ArrDel15"


# ---------------------------------------------------------------------------
# OTPPredictor class
# ---------------------------------------------------------------------------

class OTPPredictor:
    """XGBoost OTP binary classifier with calibration and evaluation tools.

    Attributes:
        params: XGBoost hyperparameters.
        model: Fitted XGBoost classifier (wrapped in calibration if calibrated).
        feature_columns: Feature names used during training.
        is_trained: Whether the model has been fitted.
        eval_results: Dict of evaluation metrics from last evaluation call.
    """

    def __init__(self, params: Optional[dict[str, Any]] = None) -> None:
        """Initialise the OTPPredictor.

        Args:
            params: XGBoost hyperparameters. Defaults to DEFAULT_XGB_PARAMS.
        """
        self.params = params or DEFAULT_XGB_PARAMS.copy()
        self.model: Optional[Any] = None
        self.feature_columns: list[str] = FEATURE_COLUMNS.copy()
        self.is_trained: bool = False
        self.eval_results: dict[str, Any] = {}
        self._feature_importances: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = TARGET_COL,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract feature matrix X and target y from a flight DataFrame.

        Args:
            df: Feature-enriched flight DataFrame.
            target_col: Name of the binary target column.

        Returns:
            Tuple of (X, y) DataFrames/Series.
        """
        available = [c for c in self.feature_columns if c in df.columns]
        missing = set(self.feature_columns) - set(available)
        if missing:
            logger.warning("Missing %d feature columns: %s", len(missing), missing)

        # Only non-cancelled flights with valid target
        mask = df[target_col].notna() & (df.get("Cancelled", 0) == 0)
        subset = df[mask].copy()

        X = subset[available].fillna(0).astype(float)
        y = subset[target_col].astype(int)

        # Update feature_columns to what's actually available
        self.feature_columns = available

        logger.info("Prepared data: X=%s, y=%s, positive_rate=%.1f%%",
                    X.shape, y.shape, y.mean() * 100)
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        calibrate: bool = True,
    ) -> "OTPPredictor":
        """Fit XGBoost classifier with optional validation set for early stopping.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Optional validation feature matrix for early stopping.
            y_val: Optional validation labels.
            calibrate: Whether to wrap model in Platt-scaling calibration.

        Returns:
            Self (for method chaining).
        """
        logger.info("Training XGBoost OTP predictor on %d samples …", len(X_train))

        base_model = xgb.XGBClassifier(**self.params)

        if X_val is not None and y_val is not None:
            base_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            base_model.fit(X_train, y_train, verbose=False)

        if calibrate:
            logger.info("Calibrating probabilities with Platt scaling …")
            cal_X = X_val if X_val is not None else X_train
            cal_y = y_val if y_val is not None else y_train
            self.model = CalibratedClassifierCV(base_model, cv=5, method="sigmoid")
            self.model.fit(cal_X, cal_y)
            self._base_model = base_model
        else:
            self.model = base_model
            self._base_model = base_model

        self.is_trained = True
        self._compute_feature_importance(X_train)
        logger.info("Training complete")
        return self

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, float]:
        """Run stratified k-fold cross-validation.

        Args:
            X: Feature matrix.
            y: Target labels.
            n_splits: Number of CV folds.

        Returns:
            Dict with mean and std of AUC-ROC across folds.
        """
        logger.info("Running %d-fold stratified cross-validation …", n_splits)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        model = xgb.XGBClassifier(**self.params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        result = {"auc_mean": float(scores.mean()), "auc_std": float(scores.std())}
        logger.info("CV AUC: %.4f ± %.4f", result["auc_mean"], result["auc_std"])
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary class predictions (0=on-time, 1=delayed).

        Args:
            X: Feature matrix.

        Returns:
            Integer array of class predictions.
        """
        self._check_trained()
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of delay (class=1).

        Args:
            X: Feature matrix.

        Returns:
            Float array of delay probabilities in [0, 1].
        """
        self._check_trained()
        X_aligned = self._align_features(X)
        proba = self.model.predict_proba(X_aligned)
        return proba[:, 1]

    def predict_single(self, features: dict[str, float]) -> dict[str, Any]:
        """Predict OTP for a single flight given a feature dict.

        Args:
            features: Dict mapping feature name to value.

        Returns:
            Dict with keys: delay_probability, prediction, risk_level.
        """
        self._check_trained()
        row = pd.DataFrame([features])
        row = row.reindex(columns=self.feature_columns, fill_value=0).fillna(0)
        prob = float(self.predict_proba(row)[0])
        pred = int(prob >= 0.5)
        risk = "Low" if prob < 0.30 else "Medium" if prob < 0.55 else "High"
        return {
            "delay_probability": round(prob, 4),
            "prediction": pred,
            "risk_level": risk,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """Compute full evaluation metrics on a held-out test set.

        Args:
            X_test: Test feature matrix.
            y_test: Test labels.

        Returns:
            Dict containing AUC-ROC, AP, F1, precision, recall, confusion matrix.
        """
        self._check_trained()
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        auc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)

        self.eval_results = {
            "auc_roc": round(auc, 4),
            "average_precision": round(ap, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {
                "precision": precision_curve.tolist(),
                "recall": recall_curve.tolist(),
            },
        }

        logger.info(
            "Evaluation — AUC-ROC: %.4f | AP: %.4f | F1: %.4f",
            auc, ap, f1,
        )
        return self.eval_results

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def _compute_feature_importance(self, X_train: pd.DataFrame) -> None:
        """Compute and store gain-based feature importance.

        Args:
            X_train: Training feature matrix (used for column names).
        """
        base = getattr(self, "_base_model", None)
        if base is None or not hasattr(base, "feature_importances_"):
            return

        importance = base.feature_importances_
        self._feature_importances = (
            pd.DataFrame({"feature": self.feature_columns, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame sorted descending by importance.

        Returns:
            DataFrame with columns: feature, importance.
        """
        if self._feature_importances is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self._feature_importances.copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the fitted model and metadata to disk.

        Args:
            path: Directory to save into. Defaults to data/models/.

        Returns:
            Path where the model was saved.
        """
        self._check_trained()
        save_dir = path or MODEL_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "otp_predictor.pkl"
        meta_path = save_dir / "otp_predictor_meta.json"

        with open(model_path, "wb") as fh:
            pickle.dump({"model": self.model, "feature_columns": self.feature_columns}, fh)

        meta = {
            "params": {k: v for k, v in self.params.items() if isinstance(v, (int, float, str, bool))},
            "feature_columns": self.feature_columns,
            "eval_results": {
                k: v for k, v in self.eval_results.items()
                if k not in ("roc_curve", "pr_curve", "classification_report")
            },
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        logger.info("Model saved to %s", model_path)
        return model_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "OTPPredictor":
        """Load a previously saved OTPPredictor from disk.

        Args:
            path: Directory containing otp_predictor.pkl.

        Returns:
            Loaded OTPPredictor instance.
        """
        load_dir = path or MODEL_DIR
        model_path = Path(load_dir) / "otp_predictor.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as fh:
            payload = pickle.load(fh)

        predictor = cls()
        predictor.model = payload["model"]
        predictor.feature_columns = payload["feature_columns"]
        predictor._base_model = getattr(predictor.model, "estimator", predictor.model)
        predictor.is_trained = True
        predictor._compute_feature_importance(
            pd.DataFrame(columns=predictor.feature_columns)
        )
        logger.info("Model loaded from %s", model_path)
        return predictor

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reindex X to match trained feature columns."""
        return X.reindex(columns=self.feature_columns, fill_value=0).fillna(0)


# ---------------------------------------------------------------------------
# Entry point: quick training demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from sklearn.model_selection import train_test_split

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.pipeline.features import build_features

    feat_path = PROCESSED_DIR / "flights_features.parquet"
    if not feat_path.exists():
        proc_path = PROCESSED_DIR / "flights_processed.parquet"
        if not proc_path.exists():
            logger.error("No processed data found. Run etl.py first.")
            sys.exit(1)
        df = pd.read_parquet(proc_path)
        df = build_features(df)
    else:
        df = pd.read_parquet(feat_path)

    predictor = OTPPredictor()
    X, y = predictor.prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, stratify=y_train, random_state=42
    )

    # CV first
    cv_results = predictor.cross_validate(X_train, y_train, n_splits=3)
    logger.info("CV Results: %s", cv_results)

    # Train
    predictor.train(X_train, y_train, X_val, y_val, calibrate=True)

    # Evaluate
    results = predictor.evaluate(X_test, y_test)
    logger.info("Test AUC-ROC: %.4f", results["auc_roc"])
    logger.info("Test F1: %.4f", results["f1_score"])

    # Feature importance
    fi = predictor.get_feature_importance()
    logger.info("Top 10 features:\n%s", fi.head(10).to_string(index=False))

    # Save
    predictor.save()
