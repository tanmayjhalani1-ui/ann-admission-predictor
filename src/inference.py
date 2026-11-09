from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "admission_pipeline.joblib"
FEATURES = [
    "GRE Score", "TOEFL Score", "University Rating", "SOP",
    "LOR", "CGPA", "Research",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def predict_admission(model, values: dict) -> float:
    row = pd.DataFrame([{feature: values[feature] for feature in FEATURES}], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    return float(np.clip(prediction, 0.0, 1.0))
