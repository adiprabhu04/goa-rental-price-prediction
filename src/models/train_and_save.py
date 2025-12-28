import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# paths
DATA_PATH = Path("data/processed/goa_cleaned.csv")
MODEL_PATH = Path("models/rf_pipeline.pkl")

print("Loading data...")
df = pd.read_csv(DATA_PATH)

df["log_price"] = np.log1p(df["price"])

X = df[["location", "sq_feet", "bhk_num", "price_per_sqft"]]
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = ["sq_feet", "bhk_num", "price_per_sqft"]
categorical_features = ["location"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )),
    ]
)

print("Training model...")
model.fit(X_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
print("RMSE (log):", rmse)

MODEL_PATH.parent.mkdir(exist_ok=True)
dump(model, MODEL_PATH)

print("✅ Model trained and saved at:", MODEL_PATH)
