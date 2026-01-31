# predict_pipeline.py
import joblib
import pandas as pd

PIPELINE_PATH = "model_pipeline.pkl"

def predict_price(area, bhk, bath, description):
    pipeline = joblib.load(PIPELINE_PATH)
    df_input = pd.DataFrame([{
        "area": area,
        "bhk": bhk,
        "bath": bath,
        "description": description
    }])
    price = pipeline.predict(df_input)[0]
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
