# batch_predict.py
import os
import pandas as pd
import joblib

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "best_model.pkl")

# --- Load Artifacts ---
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)

def feature_engineering(df):
    """
    Apply same feature engineering as in training.py
    """
    df = df.copy()
    df["balance_to_limit_ratio"] = df["balances"] / (df["credit_limit"] + 1)
    df["total_spend"] = (df["category_spend_grocery"] + df["category_spend_fuel"] + df["category_spend_travel"])
    df["spend_to_income_ratio"] = df["total_spend"] / (df["income"] + 1)
    df["intl_spend_flag"] = (df["intl_spend_pct"] > 0.20).astype(int)  # 20% threshold
    df["late_payment_risk"] = df["num_late_payments_3m"] * 2 + df["num_late_payments_12m"]
    return df

def batch_predict(input_csv, output_csv):
    """
    Run batch predictions on input CSV and save results to output CSV.
    """
    # Load data
    df = pd.read_csv(input_csv)

    # Apply feature engineering
    df_fe = feature_engineering(df)

    # Preprocess features
    X = preprocessor.transform(df_fe)

    # Predict
    predictions = model.predict(X)
    prediction_probs = model.predict_proba(X)[:, 1]  # churn probability

    # Save results
    df["churn_prediction"] = predictions
    df["churn_probability"] = prediction_probs

    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv}")


if __name__ == "__main__":
    input_csv = os.path.join(BASE_DIR, "data", "processed", "batch_input.csv")
    output_csv = os.path.join(BASE_DIR, "data", "processed", "batch_predictions.csv")

    batch_predict(input_csv, output_csv)
