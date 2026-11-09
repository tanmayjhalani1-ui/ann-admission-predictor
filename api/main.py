from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import FEATURES, load_model, predict_admission

app = FastAPI(title="Graduate Admission Prediction API", version="1.0.0")
model = None


class Applicant(BaseModel):
    gre_score: int = Field(ge=260, le=340)
    toefl_score: int = Field(ge=0, le=120)
    university_rating: int = Field(ge=1, le=5)
    sop: float = Field(ge=1, le=5)
    lor: float = Field(ge=1, le=5)
    cgpa: float = Field(ge=0, le=10)
    research: int = Field(ge=0, le=1)


@app.on_event("startup")
def startup() -> None:
    global model
    try:
        model = load_model()
    except FileNotFoundError:
        model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(applicant: Applicant):
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable. Train the model first.")
    values = dict(zip(FEATURES, [
        applicant.gre_score, applicant.toefl_score, applicant.university_rating,
        applicant.sop, applicant.lor, applicant.cgpa, applicant.research,
    ]))
    score = predict_admission(model, values)
    return {
        "estimated_chance_of_admit": score,
        "estimated_percentage": round(score * 100, 2),
        "interpretation": "Historical dataset score; not a calibrated real-world admission probability.",
    }
