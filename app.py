from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("artifacts/model.pkl", "rb"))
vectorizer = pickle.load(open("artifacts/vectorizer.pkl", "rb"))

@app.post("/predict")
def predict(data: dict):
    text_vec = vectorizer.transform([data["description"]])
    features = [[data["area"], data["bhk"], data["bath"]]]
    X = hstack([text_vec, features])
    price = model.predict(X)
    return {"predicted_price": float(price[0])}
