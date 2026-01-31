import pandas as pd
import os
import re

RAW_DATA_PATH = "real_estate.csv"
PROCESSED_DATA_PATH = "clean_data.csv"

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess():
    print("🧹 Starting data preprocessing...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    # Handle missing values
    df["description"] = df["description"].fillna("")
    df["area"] = df["area"].fillna(df["area"].median())
    df["bhk"] = df["bhk"].fillna(df["bhk"].median())
    df["bath"] = df["bath"].fillna(df["bath"].median())

    # Clean NLP text
    df["description"] = df["description"].apply(clean_text)

    # One-hot encode location
    df = pd.get_dummies(df, columns=["location"], drop_first=True)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"✅ Clean data saved to {PROCESSED_DATA_PATH}")
    print(f"📊 Final shape: {df.shape}")

if __name__ == "__main__":
    preprocess()
