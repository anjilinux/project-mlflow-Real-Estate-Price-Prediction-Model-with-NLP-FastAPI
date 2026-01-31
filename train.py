import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
import mlflow.sklearn
import os

# Paths
DATA_PATH = "clean_data.csv"
PIPELINE_PATH = "model_pipeline.pkl"

# MLflow
mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("RealEstatePriceNLP")

# Columns
TEXT_COL = "description"
NUMERIC_COLS = ["area", "bhk", "bath"]

def train():
    print("🚀 Starting model training...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)
    X = df[[TEXT_COL] + NUMERIC_COLS]
    y = df["price"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing + model pipeline
    preprocessor = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=100), TEXT_COL),
        ("numeric", StandardScaler(), NUMERIC_COLS)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Save pipeline
    joblib.dump(pipeline, PIPELINE_PATH)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("model", "XGBRegressor")
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name="RealEstatePriceModel")

    print(f"✅ Training complete | RMSE: {rmse:.2f}")


if __name__ == "__main__":
    train()
