import os
import joblib
import pandas as pd
import numpy as np

PIPELINE_PATH = "model_pipeline.pkl"
DATA_PATH = "clean_data.csv"

def test_pipeline_exists():
    assert os.path.exists(PIPELINE_PATH), "❌ Pipeline artifact missing"

def test_pipeline_predict():
    pipeline = joblib.load(PIPELINE_PATH)
    df = pd.read_csv(DATA_PATH).head(5)  # small sample
    preds = pipeline.predict(df)
    assert len(preds) == 5
    assert all(isinstance(p, (float, np.floating, int, np.integer)) for p in preds)
