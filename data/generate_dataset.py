# data/generate_dataset.py

import os
import numpy as np
import pandas as pd
from faker import Faker

# Initialize faker
fake = Faker()

# Set seed for reproducibility
np.random.seed(42)

# Number of records
N = 10000

def generate_dataset(n=N):
    # ---- Customer master ----
    customer_id = [f"CUST_{i:05d}" for i in range(1, n+1)]
    age = np.random.randint(18, 70, n)
    gender = np.random.choice(['Male', 'Female'], n)
    income = np.random.randint(2000, 20000, n)  # monthly income
    employment_type = np.random.choice(['Salaried', 'Self-Employed', 'Unemployed'], n, p=[0.6, 0.3, 0.1])
    tenure_months = np.random.randint(1, 240, n)
    segment = np.random.choice(['Mass', 'Affluent', 'HNI'], n, p=[0.7, 0.25, 0.05])

    # ---- Credit history ----
    credit_limit = np.random.randint(500, 50000, n)
    balances = np.random.randint(0, 50000, n)
    utilization_rate = balances / (credit_limit + 1)  # avoid div by zero
    num_late_payments_3m = np.random.poisson(0.2, n)
    num_late_payments_12m = np.random.poisson(1.0, n)
    max_dpd = np.random.choice([0, 15, 30, 60, 90], n, p=[0.7, 0.15, 0.1, 0.04, 0.01])
    last_default_flag = np.random.choice([0, 1], n, p=[0.95, 0.05])

    # ---- Transactions ----
    txn_count = np.random.poisson(20, n)
    avg_ticket = np.random.randint(10, 500, n)
    category_spend_grocery = np.random.randint(0, 2000, n)
    category_spend_fuel = np.random.randint(0, 1000, n)
    category_spend_travel = np.random.randint(0, 5000, n)
    intl_spend_pct = np.round(np.random.beta(2, 10, n), 2)

    # ---- Products ----
    num_products = np.random.randint(1, 5, n)
    has_credit_card = np.random.choice([0, 1], n, p=[0.3, 0.7])
    has_auto_loan = np.random.choice([0, 1], n, p=[0.8, 0.2])
    has_mortgage = np.random.choice([0, 1], n, p=[0.85, 0.15])

    # ---- Engagement / Service ----
    mobile_app_logins_30d = np.random.poisson(15, n)
    complaints_6m = np.random.poisson(0.1, n)
    nps_score = np.random.randint(-100, 100, n)
    offer_click_rate = np.round(np.random.beta(2, 8, n), 2)

    # ---- Macroeconomic (dummy proxy) ----
    cpi = np.random.normal(105, 5, n)
    unemployment_rate = np.random.uniform(3, 10, n)
    ramadan_flag = np.random.choice([0, 1], n, p=[0.9, 0.1])
    eid_flag = np.random.choice([0, 1], n, p=[0.95, 0.05])

    # ---- Labels ----
    # Simple logic: higher dpd, low income, many late payments -> higher churn/default
    churn_prob = (0.2*(num_late_payments_12m>2) + 
                  0.2*(max_dpd>30) + 
                  0.2*(income<5000) + 
                  0.2*(mobile_app_logins_30d<5) + 
                  np.random.rand(n)*0.2)
    churn = (churn_prob > 0.5).astype(int)

    default_prob = (0.3*(max_dpd>=30) + 
                    0.3*(last_default_flag==1) + 
                    0.2*(utilization_rate>0.8) + 
                    np.random.rand(n)*0.2)
    default = (default_prob > 0.5).astype(int)

    # ---- Combine into dataframe ----
    df = pd.DataFrame({
        "customer_id": customer_id,
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
        "eid_flag": eid_flag,
        "churn": churn,
        "default": default
    })

    return df

if __name__ == "__main__":
    # Ensure raw data folder exists
    raw_path = os.path.join("data", "raw")
    os.makedirs(raw_path, exist_ok=True)

    # Generate dataset
    df = generate_dataset()

    # Save to CSV
    #output_file = os.path.join(raw_path, "customers.csv")
    output_file = os.path.join(raw_path, "batch_input.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Dataset generated and saved to {output_file} with shape {df.shape}")
