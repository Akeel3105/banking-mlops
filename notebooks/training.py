# notebooks/training.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# -------------------------------
# Paths (root-relative)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))          # -> project root
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------------------
# 1) Load processed data
# -------------------------------
X = pd.read_csv(os.path.join(PROCESSED_DIR, "X_features.csv"))
y = pd.read_csv(os.path.join(PROCESSED_DIR, "y_target.csv")).values.ravel()

# -------------------------------
# 2) Train/Test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 3) Define models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=350, learning_rate=0.08, max_depth=6,
        subsample=0.9, colsample_bytree=0.9,
        random_state=42, eval_metric="logloss"
    )
}

results = {}
trained = {}

# -------------------------------
# 4) Train + evaluate
# -------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_prob)
    }
    results[name] = metrics
    trained[name] = model

# -------------------------------
# 5) Pick best model (ROC-AUC)
# -------------------------------
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = trained[best_name]
joblib.dump(best_model, os.path.join(PROCESSED_DIR, "best_model.pkl"))

print(f"✅ Best model: {best_name} (AUC={results[best_name]['roc_auc']:.3f})")
print("Saved ->", os.path.join(PROCESSED_DIR, "best_model.pkl"))

# -------------------------------
# 6) Plots (saved to processed/)
# -------------------------------
# (a) Comparison bar chart
metrics_df = pd.DataFrame(results).T
ax = metrics_df[["accuracy","precision","recall","f1","roc_auc"]].plot(
    kind="bar", figsize=(12,6)
)
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0,1)
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "model_comparison.png"), dpi=150)
plt.show()

# (b) ROC curves
plt.figure(figsize=(8,6))
for name, model in trained.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.2f})")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "roc_curves.png"), dpi=150)
plt.show()

# (c) Confusion matrix (best model)
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn","Churn"],
            yticklabels=["No Churn","Churn"])
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, "confusion_matrix.png"), dpi=150)
plt.show()

# (d) Feature importance (RF/XGB only)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    topn = importances.sort_values(ascending=False).head(15)
    plt.figure(figsize=(10,6))
    sns.barplot(x=topn.values, y=topn.index)
    plt.title(f"Top 15 Feature Importances - {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, "feature_importance.png"), dpi=150)
    plt.show()
