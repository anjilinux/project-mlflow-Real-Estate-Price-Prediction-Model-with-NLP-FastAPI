import os
import pickle
import numpy as np

MODEL_PATH = "model.pkl"
FEATURE_PATH = "artifacts/features/X_features.pkl"


def test_model_artifact_exists():
    assert os.path.exists(MODEL_PATH), "❌ Model artifact missing"


def test_model_load():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    assert model is not None


def test_model_prediction_shape():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURE_PATH, "rb") as f:
        X = pickle.load(f)

    preds = model.predict(X)
    assert len(preds) == X.shape[0]


def test_model_prediction_type():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURE_PATH, "rb") as f:
        X = pickle.load(f)

    preds = model.predict(X)
    assert isinstance(preds[0], (int, float, np.integer, np.floating))
