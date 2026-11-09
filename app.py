import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import FEATURES, load_model, predict_admission

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Graduate Admission Intelligence", page_icon="🎓", layout="wide")
st.title("Graduate Admission Intelligence")
st.caption("Estimate the dataset's historical Chance of Admit score from academic and application attributes.")

try:
    model = load_model()
except FileNotFoundError:
    st.warning("The model has not been trained yet. Run `python -m src.train` first.")
    st.stop()

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    st.subheader("Applicant profile")
    c1, c2 = st.columns(2)
    with c1:
        gre = st.number_input("GRE score", 260, 340, 315)
        toefl = st.number_input("TOEFL score", 0, 120, 105)
        rating = st.slider("University rating", 1, 5, 3)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 8.5, 0.05)
    with c2:
        sop = st.slider("Statement of Purpose", 1.0, 5.0, 3.5, 0.5)
        lor = st.slider("Letter of Recommendation", 1.0, 5.0, 3.5, 0.5)
        research = st.selectbox("Research experience", [0, 1], format_func=lambda value: "Yes" if value else "No")
    if st.button("Estimate admission score", type="primary", use_container_width=True):
        values = dict(zip(FEATURES, [gre, toefl, rating, sop, lor, cgpa, research]))
        score = predict_admission(model, values)
        st.session_state["prediction"] = score

with right:
    st.subheader("Prediction")
    if "prediction" in st.session_state:
        score = st.session_state["prediction"]
        st.metric("Estimated Chance of Admit", f"{score:.1%}")
        st.progress(score)
        st.caption("This is a model estimate of the dataset target, not an actual university decision or calibrated probability.")
    else:
        st.info("Enter an applicant profile and run an estimate.")

st.divider()
st.subheader("Model evaluation")
metadata_path = ARTIFACTS / "metadata.json"
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected model", metadata["selected_model"].replace("_", " ").title())
    m2.metric("Holdout RMSE", f'{metadata["holdout_metrics"]["rmse"]:.4f}')
    m3.metric("Holdout MAE", f'{metadata["holdout_metrics"]["mae"]:.4f}')
    m4.metric("Holdout R²", f'{metadata["holdout_metrics"]["r2"]:.3f}')
    st.dataframe(pd.DataFrame(metadata["cv_results"]).sort_values("cv_rmse"), use_container_width=True, hide_index=True)
else:
    st.info("Train the model to generate evaluation results.")
