import pandas as pd
import joblib

PIPELINE_PATH = "model_pipeline.pkl"

def load_pipeline():
    return joblib.load(PIPELINE_PATH)

def predict_price(area, bhk, bath, description):
    pipeline = load_pipeline()
    sample_df = pd.DataFrame([{
        "description": description,
        "area": area,
        "bhk": bhk,
        "bath": bath
    }])
    price = pipeline.predict(sample_df)[0]
    return float(price)


if __name__ == "__main__":
    sample_input = {
        "area": 1200,
        "bhk": 2,
        "bath": 2,
        "description": "luxury apartment near metro station"
    }
    price = predict_price(**sample_input)
    print(f"🏠 Predicted Price: ₹{price:,.2f}")
