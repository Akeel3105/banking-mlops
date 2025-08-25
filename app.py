import os
import pandas as pd
import streamlit as st
import joblib

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(DATA_DIR, "best_model.pkl")

# --- Load Artifacts ---
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)

# --- Feature Engineering Function ---
def feature_engineering(df):
    df = df.copy()
    df["balance_to_limit_ratio"] = df["balances"] / (df["credit_limit"] + 1)
    df["total_spend"] = (df["category_spend_grocery"] + df["category_spend_fuel"] + df["category_spend_travel"])
    df["spend_to_income_ratio"] = df["total_spend"] / (df["income"] + 1)
    df["intl_spend_flag"] = (df["intl_spend_pct"] > 0.20).astype(int)  # 20% threshold
    df["late_payment_risk"] = df["num_late_payments_3m"] * 2 + df["num_late_payments_12m"]
    return df

# --- Prediction Function ---
def predict_single(input_data):
    df = pd.DataFrame([input_data])
    df_fe = feature_engineering(df)
    X = preprocessor.transform(df_fe)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[:, 1][0]
    return prediction, probability

def predict_batch(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df_fe = feature_engineering(df)
    X = preprocessor.transform(df_fe)
    predictions = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    df["churn_prediction"] = predictions
    df["churn_probability"] = probs
    return df

# --- Streamlit UI ---
st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("📊 Customer Churn Prediction")

tab1, tab2 = st.tabs(["🔹 Single Prediction", "📂 Batch Prediction"])

# --- Tab 1: Single Prediction ---
with tab1:
    st.subheader("Enter Customer Details")
    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender= st.selectbox("Gender", ["Male", "Female"])
        income= st.number_input("Monthly Income", min_value=0, value=50000)
        employment_type= st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Unemployed"])
        tenure_months= st.number_input("Tenure (months)", min_value=0, value=24)
        segment= st.selectbox("Segment", ["Mass", "Affluent", "HNI"])
        credit_limit= st.number_input("Credit Limit", min_value=0, value=200000)
        balances= st.number_input("Balances", min_value=0, value=50000)
        utilization_rate= st.slider("Utilization Rate", 0.0, 1.0, 0.3)
        num_late_payments_3m= st.number_input("Late Payments (3m)", min_value=0, value=1)
        num_late_payments_12m= st.number_input("Late Payments (12m)", min_value=0, value=3)
        max_dpd= st.number_input("Max DPD", min_value=0, value=15)
        last_default_flag= st.selectbox("Last Default Flag", [0, 1])
        txn_count= st.number_input("Transaction Count", min_value=0, value=100)
        avg_ticket= st.number_input("Average Ticket Size", min_value=0, value=500)


    with col2:

        category_spend_grocery= st.number_input("Spend on Grocery", min_value=0, value=1000)
        category_spend_fuel= st.number_input("Spend on Fuel", min_value=0, value=500)
        category_spend_travel= st.number_input("Spend on Travel", min_value=0, value=2000)
        intl_spend_pct= st.slider("International Spend %", 0.0, 1.0, 0.1)
        num_products= st.number_input("Number of Products", min_value=1, value=2)
        has_credit_card= st.selectbox("Has Credit Card?", [0, 1])
        has_auto_loan= st.selectbox("Has Auto Loan?", [0, 1])
        has_mortgage= st.selectbox("Has Mortgage?", [0, 1])
        mobile_app_logins_30d= st.number_input("Mobile App Logins (30d)", min_value=0, value=15)
        complaints_6m= st.number_input("Complaints (6m)", min_value=0, value=0)
        nps_score= st.number_input("NPS Score", min_value=-100, max_value=100, value=20)
        offer_click_rate= st.slider("Offer Click Rate", 0.0, 1.0, 0.2)
        cpi= st.number_input("CPI", min_value=0.0, value=5.5)
        unemployment_rate= st.number_input("Unemployment Rate", min_value=0.0, value=6.2)
        ramadan_flag= st.selectbox("Ramadan Flag", [0, 1])
        eid_flag= st.selectbox("Eid Flag", [0, 1])

    if st.button("Predict Churn"):

        input_data = {
            "age": age,
            "gender": gender,
            "income": income,
            "employment_type": employment_type,
            "tenure_months": tenure_months,
            "segment": segment,
            "credit_limit": credit_limit,
            "balances": balances,
            "utilization_rate": utilization_rate,
            "num_late_payments_3m": num_late_payments_3m,
            "num_late_payments_12m": num_late_payments_12m,
            "max_dpd": max_dpd,
            "last_default_flag": last_default_flag,
            "txn_count": txn_count,
            "avg_ticket": avg_ticket,
            "category_spend_grocery": category_spend_grocery,
            "category_spend_fuel": category_spend_fuel,
            "category_spend_travel": category_spend_travel,
            "intl_spend_pct": intl_spend_pct,
            "num_products": num_products,
            "has_credit_card": has_credit_card,
            "has_auto_loan": has_auto_loan,
            "has_mortgage": has_mortgage,
            "mobile_app_logins_30d": mobile_app_logins_30d,
            "complaints_6m": complaints_6m,
            "nps_score": nps_score,
            "offer_click_rate": offer_click_rate,
            "cpi": cpi,
            "unemployment_rate": unemployment_rate,
            "ramadan_flag": ramadan_flag,
            "eid_flag": eid_flag
        }

        prediction, probability = predict_single(input_data)
        st.write("### ✅ Prediction Result:")
        st.write(f"Churn Prediction: **{'Yes' if prediction==1 else 'No'}**")
        st.write(f"Churn Probability: **{probability:.2f}**")

# --- Tab 2: Batch Prediction ---
with tab2:
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        result_df = predict_batch(uploaded_file)
        st.write("### 📊 Prediction Results")
        st.dataframe(result_df.head())

        # Download button
        csv_download = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv_download,
            file_name="batch_predictions.csv",
            mime="text/csv"
        )
