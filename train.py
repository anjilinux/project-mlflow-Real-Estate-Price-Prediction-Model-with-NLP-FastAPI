import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack
import numpy as np
import os

# Paths
DATA_PATH = "clean_data.csv"

# MLflow config
mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("RealEstatePriceNLP")

def train():
    print("🚀 Starting model training...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Target
    y = df["price"]

    # Separate text & numeric features
    text_features = df["description"]
    numeric_features = df.drop(columns=["price", "description"])

    # NLP: TF-IDF
    tfidf = TfidfVectorizer(max_features=100)
    X_text = tfidf.fit_transform(text_features)

    # Numeric features
    X_num = numeric_features.values

    # Combine features
    X = hstack([X_text, X_num])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # MLflow logging
        mlflow.log_param("model", "XGBRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(model, "model")

        print(f"✅ Training complete | RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train()
