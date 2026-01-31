import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os

PIPELINE_PATH = "model_pipeline.pkl"
DATA_PATH = "clean_data.csv"

def evaluate():
    print("📊 Starting model evaluation...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"{PIPELINE_PATH} not found")

    df = pd.read_csv(DATA_PATH)
    y = df["price"]

    # Load pipeline
    pipeline = joblib.load(PIPELINE_PATH)

    # Predict
    y_pred = pipeline.predict(df)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print(f"✅ Evaluation complete | RMSE: {rmse:.2f}, R2: {r2:.4f}")


if __name__ == "__main__":
    evaluate()
