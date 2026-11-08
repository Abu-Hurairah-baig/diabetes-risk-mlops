import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Diabetes Risk API")
model = joblib.load("champion.pkl")


class Patient(BaseModel):
    age: int = Field(gt=0, le=120)
    bmi: float = Field(gt=0)
    glucose: float = Field(gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Patient):
    data = pd.DataFrame([{"age": p.age, "bmi": p.bmi, "glucose": p.glucose}])
    high_risk = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    return {"high_risk": high_risk, "risk_probability": round(probability, 3)}