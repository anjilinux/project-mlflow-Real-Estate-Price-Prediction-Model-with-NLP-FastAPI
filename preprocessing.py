import pandas as pd
import os
import re

RAW_PATH = "data/raw/real_estate.csv"
PROCESSED_PATH = "data/processed/clean_data.csv"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def preprocess():
    df = pd.read_csv(RAW_PATH)

    # Handle missing values
    df.fillna({
        "description": "",
        "area": df["area"].median(),
        "bhk": df["bhk"].median(),
        "bath": df["bath"].median()
    }, inplace=True)

    # Clean NLP text
    df["description"] = df["description"].apply(clean_text)

    # One-hot encode location
    df = pd.get_dummies(df, columns=["location"], drop_first=True)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("✅ Clean data saved to:", PROCESSED_PATH)

if __name__ == "__main__":
    preprocess()
