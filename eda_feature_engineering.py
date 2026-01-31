import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import mlflow

CLEAN_DATA_PATH = "clean_data.csv"
FEATURE_DIR = "artifacts/features"

def perform_eda(df: pd.DataFrame):
    eda_report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "price_mean": df["price"].mean(),
        "price_median": df["price"].median(),
        "price_std": df["price"].std(),
        "missing_values": df.isnull().sum().to_dict()
    }
    return eda_report

def feature_engineering():
    print("🔍 Starting EDA & Feature Engineering")

    df = pd.read_csv(CLEAN_DATA_PATH)

    # ---------- EDA ----------
    eda_report = perform_eda(df)

    os.makedirs(FEATURE_DIR, exist_ok=True)

    with open(f"{FEATURE_DIR}/eda_report.txt", "w") as f:
        for k, v in eda_report.items():
            f.write(f"{k}: {v}\n")

    mlflow.log_dict(eda_report, "eda_report.json")

    # ---------- Feature Engineering ----------
    y = df["price"]

    text_data = df["description"]
    numeric_data = df.drop(columns=["price", "description"])

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english"
    )

    X_text = vectorizer.fit_transform(text_data)

    X = hstack([X_text, numeric_data.values])

    # Save artifacts
    with open(f"{FEATURE_DIR}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(f"{FEATURE_DIR}/X_features.pkl", "wb") as f:
        pickle.dump(X, f)

    with open(f"{FEATURE_DIR}/y_target.pkl", "wb") as f:
        pickle.dump(y, f)

    print("✅ Features and EDA artifacts saved")

if __name__ == "__main__":
    feature_engineering()
