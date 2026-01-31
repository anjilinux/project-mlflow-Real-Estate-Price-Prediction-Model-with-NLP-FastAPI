import pickle
import numpy as np
from scipy.sparse import hstack

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_price(area, bhk, bath, description):
    model, vectorizer = load_artifacts()

    # NLP features
    text_vec = vectorizer.transform([description])

    # Numeric features
    numeric_features = np.array([[area, bhk, bath]])

    # Combine features
    X = hstack([text_vec, numeric_features])

    prediction = model.predict(X)

    return float(prediction[0])


if __name__ == "__main__":
    # Example test run
    sample_input = {
        "area": 1200,
        "bhk": 2,
        "bath": 2,
        "description": "luxury apartment near metro station"
    }

    price = predict_price(**sample_input)
    print(f"🏠 Predicted Price: ₹{price:,.2f}")
