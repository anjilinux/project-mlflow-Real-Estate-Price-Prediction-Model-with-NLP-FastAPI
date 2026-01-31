import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import os

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# =========================
# Config
# =========================
DATA_PATH = "clean_data.csv"

mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("RealEstatePriceNLP")


def evaluate():
    print("📊 Starting model evaluation...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    # Load data
    df = pd.read_csv(DATA_PATH)

    y = df["price"]
    text_features = df["description"]
    numeric_features = df.drop(columns=["price", "description"])

    # Ensure numeric safety
    numeric_features = numeric_features.apply(pd.to_numeric, errors="coerce")
    numeric_features = numeric_features.fillna(0)
    X_num = numeric_features.astype("float64").values

    # TF-IDF (same params as training)
    tfidf = TfidfVectorizer(max_features=100)
    X_text = tfidf.fit_transform(text_features)

    # Combine features
    X = hstack([X_text, X_num])

    # Load latest trained model
    model = mlflow.sklearn.load_model("runs:/latest/model")

    # Predict
    y_pred = model.predict(X)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # Log evaluation metrics
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("eval_rmse", rmse)
        mlflow.log_metric("eval_r2", r2)

    print(f"✅ Evaluation complete | RMSE: {rmse:.2f}, R2: {r2:.4f}")


if __name__ == "__main__":
    evaluate()
