from src.app.schema import InputFeatures, Output
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI(title="Iris Classification API")

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

MODEL_PATH = "src/app/irishmodel/model.pkl"


@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
def home():
    return {"app": "iris classification"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Output)
def predict(data: InputFeatures):
    try:
        # Create 1-row dataframe
        input_df = pd.DataFrame([data.dict()])
        input_df = input_df[FEATURES]

        prediction = model.predict(input_df)[0]

        return {"species": int(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
