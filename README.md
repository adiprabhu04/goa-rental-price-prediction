# 🏠 Goa Property Price Predictor

An end-to-end Machine Learning application that predicts residential property prices in Goa using historical real estate data.  
The project covers the full ML lifecycle — from data analysis and modeling to deployment as an interactive web app.

🔗 **Live Demo:**  
👉 https://goa-property-price.streamlit.app  

---

## 🚀 Project Overview

Real estate prices vary significantly based on location, size, and property configuration.  
This project uses a **Random Forest regression model** to estimate property prices in Goa based on:

- 📍 Location
- 📐 Area (square feet)
- 🛏️ Number of bedrooms (BHK)
- 💰 Price per square foot

The trained model is deployed as a **Streamlit web application** for real-time predictions.

---

## ✨ Key Features

- End-to-end ML pipeline (EDA → Modeling → Deployment)
- Feature preprocessing using `ColumnTransformer`
- Log-transformed target variable for better regression performance
- Random Forest Regressor with strong generalization
- Model explainability using SHAP
- Interactive, dark-themed Streamlit UI
- Deployed on Streamlit Cloud

---

## 🧠 Tech Stack

**Languages & Libraries**
- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- SHAP

**Model**
- Random Forest Regressor

**Frontend / Deployment**
- Streamlit
- Streamlit Cloud

---

## 📊 Model Performance

- **Metric:** RMSE (log-scale)
- **RMSE:** ~ **0.07**
- Log transformation used to stabilize variance and reduce skewness

---

## 🖥️ App Preview

The app allows users to:
- Select a Goa location
- Enter property details
- Instantly get an estimated price



---

## 🗂️ Project Structure

goa-rental-price-prediction/
│
├── app/
│ └── app.py # Streamlit application
│
├── data/
│ ├── raw/ # Raw dataset
│ └── processed/ # Cleaned & processed data
│
├── models/
│ └── rf_pipeline.pkl # Trained ML pipeline
│
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_EDA.ipynb
│ └── 03_Modeling.ipynb
│
├── src/
│ └── models/
│ └── train_and_save.py # Model training script
│
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy code

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/adiprabhu04/goa-rental-price-prediction.git
cd goa-rental-price-prediction
2️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Streamlit app
bash
Copy code
streamlit run app/app.py
📈 Future Improvements
Add confidence intervals for predictions

Integrate map-based location selection

Compare multiple models (XGBoost, Gradient Boosting)

Add downloadable PDF price reports

Extend to rental price prediction

👤 Author
Aditya Prabhudessai
B.Tech IT | Aspiring Data Scientist / ML Engineer

GitHub: https://github.com/adiprabhu04

LinkedIn: https://www.linkedin.com/in/aditya-prabhudessai/

⚠️ Disclaimer
This application provides ML-based estimates using historical data.
Actual property prices may vary based on market conditions, negotiations, and additional factors not captured in the dataset.

⭐ If you found this project interesting, feel free to star the repo!