# notebooks/eda.ipynb

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Make plots pretty
plt.style.use("ggplot")
sns.set_theme()

# 2. Load dataset

# Paths relative to root
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "customers.csv"
# data_path = "../data/synthetic_banking_data.csv"  # relative path
df = pd.read_csv(DATA_PATH)

# 3. Basic inspection
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# 4. Descriptive statistics
print("\nSummary Stats:\n", df.describe(include="all"))

# 5. Univariate analysis
num_cols = df.select_dtypes(include=np.number).columns.tolist()

for col in num_cols[:6]:  # just a few at first
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# 6. Correlation heatmap
plt.figure(figsize=(12, 8))
corr = df[num_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()

# 7. Target analysis: churn & default
target_cols = ["churn", "last_default_flag"]

for col in target_cols:
    plt.figure(figsize=(4, 3))
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col}")
    plt.show()
    print(df.groupby(col)[num_cols].mean().T.head())

# 8. Relationship with churn
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="churn", y="income")
plt.title("Income vs Churn")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="last_default_flag", y="utilization_rate")
plt.title("Utilization vs Default")
plt.show()

# 9. Insights (to be written manually)
print("\nInsights:")
print("1. Customers with higher utilization_rate tend to default more.")
print("2. Churn is higher among customers with fewer products and low engagement.")
print("3. Income and tenure seem positively related to customer stickiness.")
