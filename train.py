# train_pipeline.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

DATA_PATH = "clean_data.csv"
PIPELINE_PATH = "model_pipeline.pkl"

mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("RealEstatePriceNLP")

def train_pipeline():
    df = pd.read_csv(DATA_PATH)

    y = df["price"]
    X = df.drop(columns=["price"])

    # Numeric features
    numeric_features = ["area", "bhk", "bath"]

    # Column transformer: text + numeric
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(max_features=100), "description"),
        ("numeric", "passthrough", numeric_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = ((y_test - y_pred) ** 2).mean() ** 0.5

        # Save pipeline
        joblib.dump(pipeline, PIPELINE_PATH)

        # Log MLflow
        mlflow.log_param("model", "XGBRegressor Pipeline")
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name="RealEstatePriceModel")

        print(f"✅ Training complete | RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_pipeline()
