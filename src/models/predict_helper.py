
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parents[2] / "models" / "rf_pipeline.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_price(model, location, sq_feet, bhk_num, price_per_sqft=None, default_pps=None):
    # If caller doesn't supply price_per_sqft, use default or model fallback
    X = pd.DataFrame([{
        "location": location,
        "sq_feet": float(sq_feet),
        "bhk_num": float(bhk_num),
        "price_per_sqft": float(price_per_sqft) if price_per_sqft is not None else float(default_pps)
    }])
    log_pred = model.predict(X)[0]
    return {"pred_log": float(log_pred), "pred_price_inr": int(round(np.expm1(log_pred)))}

if __name__ == "__main__":
    m = load_model()
    print(predict_price(m, "Porvorim", 1200, 2, price_per_sqft=2000))
