import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("🏠 Goa Property Price Predictor")
st.markdown("Enter details below to estimate the price (INR)")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("models/rf_pipeline.pkl")

model = load_model()

# UI inputs
location = st.text_input("Location", "Porvorim")
sq_feet = st.number_input("Square Feet", min_value=100, step=50)
bhk = st.number_input("BHK", min_value=1, step=1)
pps = st.number_input("Price per sq ft (optional)", min_value=0, step=50)

if st.button("Predict Price"):
    df = pd.DataFrame([{
        "location": location,
        "sq_feet": sq_feet,
        "bhk_num": bhk,
        "price_per_sqft": pps if pps > 0 else 2000  # fallback
    }])

    log_pred = model.predict(df)[0]
    price = np.expm1(log_pred)

    st.success(f"### Estimated Price: ₹{price:,.0f}")
