import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join("models", "model.joblib")
pipeline = joblib.load(MODEL_PATH)

app = FastAPI(title="Student Performance Prediction API")


class StudentInput(BaseModel):
    age: int = Field(..., ge=10, le=25)
    Medu: int = Field(..., ge=0, le=4)
    Fedu: int = Field(..., ge=0, le=4)
    traveltime: int = Field(..., ge=1, le=4)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0, le=5)
    famrel: int = Field(..., ge=1, le=5)
    freetime: int = Field(..., ge=1, le=5)
    goout: int = Field(..., ge=1, le=5)
    Dalc: int = Field(..., ge=1, le=5)
    Walc: int = Field(..., ge=1, le=5)
    health: int = Field(..., ge=1, le=5)
    absences: int = Field(..., ge=0)
    G1: int = Field(..., ge=0, le=20)
    G2: int = Field(..., ge=0, le=20)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(student: StudentInput):
    data = student.dict()
    df = pd.DataFrame([data])

    pred = pipeline.predict(df)[0]
    grade = float(pred)

    if grade < 10:
        status = "Poor Performance"
    elif grade < 14:
        status = "Average Performance"
    else:
        status = "Good Performance"

    return {
        "predicted_grade": round(grade, 2),
        "status": status
    }
