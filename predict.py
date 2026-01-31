import pickle
import joblib
import numpy as np
from scipy.sparse import hstack

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "tfidf.pkl"

def load_artifacts():
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    # Load TF-IDF
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_price(area, bhk, bath, description):
    model, vectorizer = load_artifacts()
    text_vec = vectorizer.transform([description])
    numeric_features = np.array([[area, bhk, bath]])
    X = hstack([text_vec, numeric_features])
    return float(model.predict(X)[0])

if __name__ == "__main__":
    sample_input = {
        "area": 1200,
        "bhk": 2,
        "bath": 2,
        "description": "luxury apartment near metro station"
    }
    price = predict_price(**sample_input)
    print(f"🏠 Predicted Price: ₹{price:,.2f}")
