# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_price

app = FastAPI(title="Real Estate Price Prediction API", version="1.0")

# =========================
# Request Body Model
# =========================
class PriceRequest(BaseModel):
    area: float
    bhk: int
    bath: int
    description: str

# =========================
# Health Check
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "message": "API is running"}

# =========================
# Prediction Endpoint
# =========================
@app.post("/predict")
def predict(req: PriceRequest):
    try:
        price = predict_price(
            area=req.area,
            bhk=req.bhk,
            bath=req.bath,
            description=req.description
        )
        return {"predicted_price": price}
    except Exception as e:
        return {"error": str(e)}
