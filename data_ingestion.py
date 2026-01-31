import pandas as pd
import os
from datetime import datetime

RAW_DATA_PATH = "real_estate.csv"

REQUIRED_COLUMNS = [
    "price", "area", "bhk", "bath", "location", "description"
]

def validate_schema(df: pd.DataFrame):
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

def ingest_data(source_path: str):
    print("📥 Starting data ingestion...")

    df = pd.read_csv(source_path)

    # Validate schema
    validate_schema(df)

    # Create raw directory if not exists
    os.makedirs("data/raw", exist_ok=True)

    # Save raw data
    df.to_csv(RAW_DATA_PATH, index=False)

    print(f"✅ Data ingested successfully at {RAW_DATA_PATH}")
    print(f"🕒 Ingestion time: {datetime.now()}")

    return df

if __name__ == "__main__":
    # Example usage (local file ingestion)
    SOURCE_PATH = "real_estate_source.csv"
    ingest_data(SOURCE_PATH)
