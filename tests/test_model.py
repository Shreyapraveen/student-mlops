import os
import joblib
import pandas as pd


def test_model_file_exists():
    assert os.path.exists(os.path.join("models", "model.joblib"))


def test_model_can_predict():
    pipeline = joblib.load(os.path.join("models", "model.joblib"))
    sample = {
        "age": 17,
        "Medu": 3,
        "Fedu": 2,
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 2,
        "health": 4,
        "absences": 5,
        "G1": 12,
        "G2": 13
    }
    df = pd.DataFrame([sample])
    pred = pipeline.predict(df)[0]
    assert pred >= 0