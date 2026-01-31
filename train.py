import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import pickle
import os
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack

# Paths
DATA_PATH = "clean_data.csv"
MODEL_PATH = "model.pkl"
TFIDF_PATH = "tfidf.pkl"
FEATURE_DIR = "artifacts/features"
FEATURE_PATH = f"{FEATURE_DIR}/X_features.pkl"

mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("RealEstatePriceNLP")


def train():
    print("🚀 Starting model training...")

    os.makedirs(FEATURE_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)

    y = df["price"]
    text_features = df["description"]
    numeric_features = df.drop(columns=["price", "description"])
    numeric_features = numeric_features.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float64")
    X_num = numeric_features.values

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=100)
    X_text = tfidf.fit_transform(text_features)

    # Combine
    X = hstack([X_text, X_num])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save test features for pytest
    with open(FEATURE_PATH, "wb") as f:
        pickle.dump(X_test, f)

    with mlflow.start_run():
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Save local model and TFIDF
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        joblib.dump(tfidf, TFIDF_PATH)

        # Log artifacts and metrics
        mlflow.log_param("model", "XGBRegressor")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_artifact(MODEL_PATH)
        mlflow.log_artifact(TFIDF_PATH)
        mlflow.log_artifact(FEATURE_PATH)

        # Register model
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="RealEstatePriceModel")

        print(f"✅ Training complete | RMSE: {rmse:.2f}")


if __name__ == "__main__":
    train()
