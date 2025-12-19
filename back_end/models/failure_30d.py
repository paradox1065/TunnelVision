# model that predicts if there will be a failure in the next 30 days
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from train import build_features as bf
from scipy.sparse import csr_matrix
import joblib

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/bay_area_infrastructure_balanced.csv")
print("Loading CSV from:", csv_path)

# --- Load features ---
X, df, feature_cols = bf(csv_path, target="failure_30d")

y = df["failure_next_30d"].astype(int)

# --- Split by asset_id to avoid leakage ---
asset_ids = df["asset_id"].unique()
np.random.seed(42)
np.random.shuffle(asset_ids)

split_idx = int(0.8 * len(asset_ids))
train_assets = asset_ids[:split_idx]
test_assets = asset_ids[split_idx:]

train_mask = df["asset_id"].isin(train_assets)
test_mask = df["asset_id"].isin(test_assets)

# --- Convert to CSR so we can index properly ---
X = csr_matrix(X)

X_train = X[train_mask.values]
X_test = X[test_mask.values]
y_train = y[train_mask]
y_test = y[test_mask]

# --- Train Random Forest ---
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --- Metrics ---
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n\n\n--- Test Metrics ---")
print("Accuracy:", acc)
print("ROC AUC:", auc)
print("Confusion Matrix:\n", cm)

# --- Optional: cross-validated ROC AUC ---
from sklearn.model_selection import cross_val_score
cv_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
print("CV ROC AUC:", cv_auc.mean())
print("\n\n\n")


# -------------------------
# Save model + metadata
# -------------------------
model_dir = os.path.join(script_dir, "../models")
os.makedirs(model_dir, exist_ok=True)

# Save RandomForest model
joblib.dump(
    model,
    os.path.join(model_dir, "failure_30d_rfc.pkl")
)

# Save feature list (important for inference)
joblib.dump(
    feature_cols,
    os.path.join(model_dir, "failure_30d_features.pkl")
)

# Save metrics
import json

metrics = {
    "accuracy": float(acc),
    "roc_auc": float(auc),
    "confusion_matrix": cm.tolist()
}

with open(os.path.join(model_dir, "failure_30d_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
