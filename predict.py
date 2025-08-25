# predict.py

import os
import sys
import joblib
import pandas as pd

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

MODEL_PATH = os.path.join(DATA_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.pkl")

# -------------------------------
# Load Model & Preprocessor
# -------------------------------
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    print(f"❌ Error loading model/preprocessor: {e}")
    sys.exit(1)

# -------------------------------
# Prediction Function
# -------------------------------
def predict_churn(input_data):
    """
    input_data : pd.DataFrame or dict
        New customer data (same features as training).
    
    Returns : dict
        Prediction label and probability.
    """

    # If dict provided, convert to DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    # Apply preprocessing
    X_processed = preprocessor.transform(input_data)

    # Predict churn
    pred_class = model.predict(X_processed)[0]
    pred_proba = model.predict_proba(X_processed)[0][1]

    return {
        "prediction": int(pred_class),
        "churn_probability": round(float(pred_proba), 4)
    }

# -------------------------------
# Example Run (local test)
# -------------------------------
if __name__ == "__main__":
    # Example test input with the same columns as training data
    sample = {
        "age":20,
        "income": 45000,
        "tenure_months": 24,
        "credit_limit": 100000,
        "balances": 25000,
        "utilization_rate": 0.25,
        "num_late_payments_3m": 1,
        "num_late_payments_12m": 3,
        "max_dpd": 30,
        "last_default_flag": 0,
        "txn_count": 45,
        "avg_ticket": 1200,
        "category_spend_grocery": 5000,
        "category_spend_fuel": 2000,
        "category_spend_travel": 1000,
        "intl_spend_pct": 0.15,
        "num_products": 3,
        "has_credit_card": 1,
        "has_auto_loan": 0,
        "has_mortgage": 1,
        "mobile_app_logins_30d": 20,
        "complaints_6m": 0,
        "nps_score": 8,
        "offer_click_rate": 0.1,
        "cpi": 105,
        "unemployment_rate": 0.07,
        "ramadan_flag": 0,
        "eid_flag": 1,
        "balance_to_limit_ratio": 0.3,
        "total_spend": 7000,
        "spend_to_income_ratio": 0.15,
        "late_payment_risk": 0.2,
        "gender": "Male",
        "employment_type": "Salaried",
        "segment": "Mass"
    }

    result = predict_churn(sample)
    print("✅ Prediction Result:", result)

