# src/models/save_and_predict.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("../models/rf_pipeline.pkl")   # where we'll save
SCALAR_PATH = Path("../models/preprocessor.pkl") # optional if you saved separately

def save_pipeline(pipeline, path=MODEL_PATH):
    """
    Save the full pipeline (preprocessor + model) to disk.
    pipeline: sklearn.pipeline.Pipeline (preprocessor + regressor)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Saved pipeline to {path}")

def load_pipeline(path=MODEL_PATH):
    return joblib.load(path)

def predict_price_from_inputs(pipeline, location, sq_feet, bhk_num):
    """
    Inputs:
      - pipeline: the loaded sklearn Pipeline (preprocessor + regressor)
      - location: string (must match training categories; unknown handled by OHE ignore)
      - sq_feet: numeric
      - bhk_num: int or numeric
    Returns:
      - dict with predicted_log_price and predicted_price_inr (rounded int)
    """
    X = pd.DataFrame([{
        "location": location,
        "sq_feet": float(sq_feet),
        "bhk_num": float(bhk_num),
        # price_per_sqft can be computed from a typical local rate, but we compute from model expectation:
    }])
    # Compute price_per_sqft feature as: we don't know price, so set price_per_sqft to median for that locality if available.
    # If you saved a lookup table, use it. Here we compute using global median as fallback.
    # pipeline expects 'price_per_sqft' column -- if your pipeline doesn't, remove below.
    try:
        # try to access training median lookup (if exists)
        medians = pipeline.named_steps.get('median_lookup', None)
        # If you didn't add a median lookup step, fallback to global median
        X['price_per_sqft'] = pipeline.named_steps['preprocessor'].transformers_[0][2]  # dummy, fallback
    except Exception:
        # fallback: use a sensible default — global median
        # NOTE: Replace this with a real locality->median if you saved one.
        global_median_pps =  df_model['price_per_sqft'].median() if 'df_model' in globals() else 2000
        X['price_per_sqft'] = global_median_pps

    # If your pipeline expects only location, sq_feet, bhk_num, price_per_sqft in that order, keep X columns matching.
    pred_log = pipeline.predict(X)[0]
    pred_price = float(np.expm1(pred_log))
    return {"pred_log": float(pred_log), "pred_price": int(round(pred_price))}

# Example - run once after you have 'model_rf' in notebook
if __name__ == "__main__":
    # this block used manually from terminal/notebook to save model
    import sys
    print("This script saves a trained pipeline. Call save_pipeline(pipeline).")
