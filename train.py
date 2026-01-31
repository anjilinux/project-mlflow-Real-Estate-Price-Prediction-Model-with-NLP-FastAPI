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
import joblib

# =========================
# Config
# =========================
DATA_PATH = "clean_data.csv"
VECTORIZER_PATH = "tfidf.pkl"

mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("RealEstatePriceNLP")


def train():
    print("🚀 Starting model training...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)

    y = df["price"]
    text_features = df["description"]
    numeric_features = df.drop(columns=["price", "description"])

    # Numeric safety
    numeric_features = numeric_features.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_num = numeric_features.astype("float64").values

    # NLP
    tfidf = TfidfVectorizer(max_features=100)
    X_text = tfidf.fit_transform(text_features)

    # Save vectorizer
    joblib.dump(tfidf, VECTORIZER_PATH)

    # Combine features
    X = hstack([X_text, X_num])

    X_tra_
