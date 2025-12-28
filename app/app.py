import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Goa Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ---------------- GLOBAL DARK THEME CSS ----------------
st.markdown("""
<style>

/* ---------- APP BACKGROUND ---------- */
.stApp {
    background: linear-gradient(135deg, #0b0f1a, #111827);
    color: #e5e7eb;
}

/* ---------- HEADINGS ---------- */
h1, h2, h3 {
    color: #f9fafb;
    letter-spacing: 0.5px;
}

/* ---------- GLASS CARD ---------- */
.glass {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 26px;
    margin-top: 24px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.08);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass:hover {
    transform: translateY(-2px);
    box-shadow: 0 30px 60px rgba(0,0,0,0.6);
}

/* ---------- INPUTS ---------- */
input, select {
    background-color: #020617 !important;
    color: #f9fafb !important;
    border-radius: 10px !important;
}

/* ---------- BUTTON ---------- */
button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    color: white !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.25s ease;
}
button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(37,99,235,0.4);
}

/* ---------- PRICE CARD ---------- */
.price-box {
    margin-top: 28px;
    padding: 26px;
    border-radius: 18px;
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: #ecfdf5;
    font-size: 30px;
    font-weight: 700;
    text-align: center;
    letter-spacing: 0.4px;
    animation: fadeUp 0.5s ease;
}

/* ---------- ANIMATION ---------- */
@keyframes fadeUp {
    from {opacity: 0; transform: translateY(12px);}
    to {opacity: 1; transform: translateY(0);}
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
MODEL_PATH = Path("models/rf_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏖️ Goa Property ML")
st.sidebar.markdown(
    "<span style='color:#60a5fa'>Random Forest • Streamlit • Scikit-learn</span>",
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
st.sidebar.caption("Built by Aditya 🚀")

# ---------------- MAIN CONTENT ----------------
st.markdown(
    "<h1 style='text-align:center;'>🏠 Goa Property Price Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; opacity:0.75;'>Premium ML-based estimation for Goa real estate</p>",
    unsafe_allow_html=True
)

# ---------------- FORM CARD ----------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox(
            "📍 Location",
            ["Porvorim", "Mapusa", "Taleigao", "Calangute", "Siolim", "Panjim", "Margao", "Dabolim"]
        )
        bhk_num = st.slider("🛏️ BHK", 1, 6, 2)

    with col2:
        sq_feet = st.number_input("📐 Area (sq ft)", min_value=300, value=1200, step=50)
        price_per_sqft = st.number_input("💰 Price per sq ft", min_value=500, value=2000, step=100)

    submitted = st.form_submit_button("🔮 Predict Price")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if submitted:
    input_df = pd.DataFrame([{
        "location": location,
        "sq_feet": sq_feet,
        "bhk_num": bhk_num,
        "price_per_sqft": price_per_sqft
    }])

    log_price_pred = model.predict(input_df)[0]
    price_pred = np.expm1(log_price_pred)

    st.markdown(
        f"<div class='price-box'>💰 Estimated Price: ₹ {price_pred:,.0f}</div>",
        unsafe_allow_html=True
    )

    st.caption("⚠️ ML estimate based on historical data. Actual prices may vary.")
