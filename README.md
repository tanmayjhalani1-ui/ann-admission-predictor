# Graduate Admission Prediction

An end-to-end regression project that estimates the historical **Chance of Admit** score from GRE, TOEFL, university rating, SOP, LOR, CGPA and research experience. The original notebook, Streamlit script and CSV are retained.

## Project structure

```text
.
├── Admission_Predict.csv
├── gre-admission-prediction.ipynb
├── greadmission.py
├── app.py
├── api/
│   └── main.py
├── src/
│   ├── train.py
│   └── inference.py
├── artifacts/                 # generated locally
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

Training compares Ridge Regression, Random Forest, HistGradientBoosting and two multilayer-perceptron (ANN) configurations. Preprocessing is contained in a reusable scikit-learn pipeline. Candidate selection uses shuffled 5-fold cross-validation on the training split; the selected model is evaluated on a separate holdout split. The fitted pipeline and metadata are written to `artifacts/`.

## MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
python -m src.train
```

Runs record candidate CV metrics, selected model information, holdout metrics and the fitted model artifact. These are experiment runs; this repository does not currently implement Model Registry promotion or version stages.

## Streamlit

```bash
streamlit run app.py
```

## FastAPI

```bash
uvicorn api.main:app --reload
```

- `GET /health`
- `POST /predict`

Example request:

```json
{
  "gre_score": 315,
  "toefl_score": 105,
  "university_rating": 3,
  "sop": 3.5,
  "lor": 3.5,
  "cgpa": 8.5,
  "research": 1
}
```

## Docker

```bash
docker compose up --build
```

Open the Streamlit application at `http://localhost:8501`. The image trains a model during build from the included CSV.

## Dataset and limitations

The repository's original CSV contains 400 observations. The target is the dataset's historical `Chance of Admit` score, not a verified admission outcome. A prediction must not be interpreted as a calibrated probability, a university decision, or a guarantee. The dataset is small and may not represent current admissions policies, universities, countries or applicant populations. Use this project as an ML engineering demonstration, not as an admissions decision system.

## Automation

GitHub Actions performs dependency installation, Python compilation, a reproducible training run and artifact checks. No test framework is required for this workflow.
