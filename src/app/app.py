from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from schema import InputFeatures,Output

app = FastAPI(title="Iris Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

MODEL_PATHS = {
    "logistic": "/Users/siddharthaganguli/Desktop/irish/src/model/mlruns/0/models/m-a255c44742b442c784239b9b50d1c1e2/artifacts/model.pkl",
    "decision_tree": "/Users/siddharthaganguli/Desktop/irish/src/model/mlruns/0/models/m-aa9048c735024b6f8768006894860189/artifacts/model.pkl"
}

models = {}

@app.on_event("startup")
def load_models():
    for name, path in MODEL_PATHS.items():
        models[name] = joblib.load(path)

@app.get("/")
def home():
    return {"app": "iris classification"}

@app.post("/predict", response_model=Output)
def predict(data: InputFeatures):
    if data.model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    input_df = pd.DataFrame([{
        "sepal_length": data.sepal_length,
        "sepal_width": data.sepal_width,
        "petal_length": data.petal_length,
        "petal_width": data.petal_width,
    }])

    prediction = models[data.model_name].predict(input_df)[0]
    return {"species": int(prediction)}
