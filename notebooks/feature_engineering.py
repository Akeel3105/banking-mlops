# notebooks/feature_engineering.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# -------------------------------
# Paths (root-relative)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))          # -> project root
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------------------
# 1) Load raw dataset
# -------------------------------
raw_file = os.path.join(RAW_DIR, "customers.csv")
df = pd.read_csv(raw_file)

# -------------------------------
# 2) Target & ID
# -------------------------------
# We use the existing binary label 'churn' as the SINGLE objective.
TARGET_COL = "churn"
ID_COL = "customer_id"

# -------------------------------
# 3) Feature Engineering (domain)
# -------------------------------
# NOTE: We keep column names exactly as in customers.csv
# and add engineered features without renaming any originals.

df["balance_to_limit_ratio"] = df["balances"] / (df["credit_limit"] + 1)
df["total_spend"] = (
    df["category_spend_grocery"]
    + df["category_spend_fuel"]
    + df["category_spend_travel"]
)
df["spend_to_income_ratio"] = df["total_spend"] / (df["income"] + 1)
df["intl_spend_flag"] = (df["intl_spend_pct"] > 0.20).astype(int)  # 20% threshold
df["late_payment_risk"] = df["num_late_payments_3m"] * 2 + df["num_late_payments_12m"]

# -------------------------------
# 4) Select Features (no leakage)
# -------------------------------
# Exclude ID, target, and 'default' (other label that would leak)
drop_cols = [ID_COL, TARGET_COL, "default"]

feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].copy()
y = df[TARGET_COL].astype(int).copy()

# -------------------------------
# 5) Preprocessing (fit → transform)
# -------------------------------
# Split features into numeric/categorical by dtype
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    # keep dense output to save as CSV easily
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop"
)

# Fit on full X (we're just preparing processed training features)
X_processed = preprocessor.fit_transform(X)

# Ensure dense matrix (in case)
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

# Recover feature names (robust fallback)
try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    # Fallback for older sklearn
    num_names = numeric_features
    cat_names = []
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_features)
    except Exception:
        # If still not available, generate simple names
        cat_names = [f"cat_{i}" for i in range(
            preprocessor.transformers_[1][1].named_steps["onehot"].categories_.size
        )]
    feature_names = np.concatenate([np.array([f"num__{c}" for c in num_names]),
                                    np.array([f"cat__{c}" for c in cat_names])])

X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# -------------------------------
# 6) Save processed features/target + preprocessor
# -------------------------------
X_file = os.path.join(PROCESSED_DIR, "X_features.csv")
y_file = os.path.join(PROCESSED_DIR, "y_target.csv")
pp_file = os.path.join(PROCESSED_DIR, "preprocessor.pkl")

X_processed_df.to_csv(X_file, index=False)
y.to_csv(y_file, index=False)
joblib.dump(preprocessor, pp_file)

print("✅ Saved:")
print(f"  - {X_file}  (shape={X_processed_df.shape})")
print(f"  - {y_file}  (shape={y.shape})")
print(f"  - {pp_file}")
