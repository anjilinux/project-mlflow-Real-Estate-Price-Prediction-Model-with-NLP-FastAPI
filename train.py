import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import os

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack

# =========================
# Config
# =========================
DATA_PATH = "clean_data.csv"
TFIDF_PATH = "tfidf.pkl"

# MLflow config
mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("RealEstatePriceNLP")


def train():
    print("🚀 Starting model training...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Target
    y = df["price"]

    # Features
    text_features = df["description"]
    numeric_features = df.drop(columns=["price", "description"])

    # Ensure numeric safety
    numeric_features = (
        numeric_features
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("float64")
    )

    X_num = numeric_features.values

    # NLP TF-IDF
    tfidf = TfidfVectorizer(max_features=100)
    X_text = tfidf.fit_transform(text_features)

    # Combine features
    X = hstack([X_text, X_num])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # MLflow Run
    # =========================
    with mlflow.start_run():
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Log params & metrics
        mlflow.log_param("model", "XGBRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", rmse)

        # Save & log TF-IDF
        joblib.dump(tfidf, TFIDF_PATH)
        mlflow.log_artifact(TFIDF_PATH)

        # Register model (CREATES MODEL REGISTRY ENTRY)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="RealEstatePriceModel"
        )

        print(f"✅ Training complete | RMSE: {rmse:.2f}")


if __name__ == "__main__":
    train()
