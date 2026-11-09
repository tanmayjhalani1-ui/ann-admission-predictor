from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("ADMISSION_DATA", ROOT / "Admission_Predict.csv"))
ARTIFACT_DIR = ROOT / "artifacts"
TARGET = "Chance of Admit"
FEATURES = [
    "GRE Score", "TOEFL Score", "University Rating", "SOP",
    "LOR", "CGPA", "Research",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    data = data[required].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=[TARGET])
    if data.empty or data[FEATURES].isna().all().any():
        raise ValueError("Dataset has no usable training rows or a fully empty feature.")
    if not data[TARGET].between(0, 1).all():
        raise ValueError("Admission target must be between 0 and 1.")
    return data[FEATURES], data[TARGET]


def build_candidates() -> dict[str, Pipeline]:
    def pipeline(estimator: object) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    return {
        "ridge": pipeline(Ridge(alpha=1.0)),
        "random_forest": pipeline(RandomForestRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1,
        )),
        "hist_gradient_boosting": pipeline(HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=250, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )),
        "ann_64_32": pipeline(MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
            alpha=0.001, learning_rate_init=0.003, max_iter=1500,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=35, random_state=42,
        )),
        "ann_128_64_32": pipeline(MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
            alpha=0.002, learning_rate_init=0.002, max_iter=1800,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=40, random_state=42,
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates = build_candidates()
    results: list[dict] = []
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        scores = cross_validate(
            clone(estimator), X_train, y_train, cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1, return_train_score=False,
        )
        model = clone(estimator)
        model.fit(X_train, y_train)
        fitted[name] = model
        results.append({
            "model": name,
            "cv_rmse": float(-scores["test_rmse"].mean()),
            "cv_rmse_std": float(scores["test_rmse"].std()),
            "cv_mae": float(-scores["test_mae"].mean()),
            "cv_r2": float(scores["test_r2"].mean()),
        })

    results.sort(key=lambda row: row["cv_rmse"])
    best_name = results[0]["model"]
    best_model = fitted[best_name]
    predictions = np.clip(best_model.predict(X_test), 0.0, 1.0)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, ARTIFACT_DIR / "admission_pipeline.joblib")
    metadata = {
        "target": TARGET,
        "features": FEATURES,
        "selected_model": best_name,
        "cv_results": results,
        "holdout_metrics": metrics,
        "dataset_rows": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_note": "Chance of Admit is a historical dataset score, not a calibrated real-world admission probability.",
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if not args.skip_mlflow:
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_experiment("graduate-admission-regression")
            with mlflow.start_run(run_name=f"selected-{best_name}"):
                mlflow.log_param("selected_model", best_name)
                mlflow.log_param("dataset_rows", len(X))
                mlflow.log_param("cv_strategy", "5-fold shuffled KFold")
                mlflow.log_metrics({f"holdout_{key}": value for key, value in metrics.items()})
                for row in results:
                    mlflow.log_metrics({
                        f"{row['model']}_cv_rmse": row["cv_rmse"],
                        f"{row['model']}_cv_mae": row["cv_mae"],
                        f"{row['model']}_cv_r2": row["cv_r2"],
                    })
                mlflow.sklearn.log_model(best_model, name="admission_pipeline")
                mlflow.log_artifact(str(ARTIFACT_DIR / "metadata.json"))
        except ImportError:
            print("MLflow is not installed; local training completed without experiment tracking.")

    print(json.dumps({"selected_model": best_name, "holdout_metrics": metrics, "cv_results": results}, indent=2))


if __name__ == "__main__":
    main()
