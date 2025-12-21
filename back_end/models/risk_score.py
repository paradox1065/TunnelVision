# model that predicts risk_score
import os
import numpy as np
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import csr_matrix
from .train import build_features as bf
from sklearn.model_selection import cross_val_score
import joblib

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/bay_area_infrastructure_balanced.csv")
print("Loading CSV from:", csv_path)

# --- Build features ---
X, df, feature_cols = bf(csv_path, target="risk_score")

# --- Target ---
y = df["risk_score"].clip(5, 95)

# --- Asset-based split ---
asset_ids = df["asset_id"].unique()
np.random.seed(42)
np.random.shuffle(asset_ids)
split_idx = int(0.8 * len(asset_ids))
train_assets = asset_ids[:split_idx]
test_assets = asset_ids[split_idx:]
train_mask = df["asset_id"].isin(train_assets)
test_mask = df["asset_id"].isin(test_assets)

# --- Convert to CSR ---
from scipy.sparse import csr_matrix
X = csr_matrix(X)

X_train = X[train_mask.values]
X_test = X[test_mask.values]
y_train = y[train_mask.values]
y_test = y[test_mask.values]

# --- Model ---
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=2,
    subsample=0.6,
    random_state=42
)
model.fit(X_train, y_train)

# --- Predictions & metrics ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
cv_rmse = -cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

cv_r2 = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="r2"
)

print("\n\n\n--- Test Metrics ---")
print("CV RMSE:", cv_rmse.mean())
print("CV R²:", cv_r2.mean())
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
print("\n\n\n")

model_dir = os.path.join(script_dir, "../models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(
    model,
    os.path.join(model_dir, "risk_score_gbr.pkl")
)

joblib.dump(
    feature_cols,
    os.path.join(model_dir, "risk_score_features.pkl")
)



from pathlib import Path
from back_end.features_schema import build_feature_vector  # existing function from your code

# --- Paths ---
BASE_DIR = Path(__file__).parent

# --- Load trained model + label encoder ---
risk_score_model = joblib.load(BASE_DIR / "risk_score_gbr.pkl")
risk_score_le = joblib.load(BASE_DIR / "risk_score_features.pkl")

def predict_risk_score(X) -> str:
    """
    Predict the recommended action for a single asset feature dictionary.
    
    Args:
        feature_dict (dict): dictionary containing feature names and values.
        
    Returns:
        str: predicted recommended action (decoded from label encoder)
    """
    risk_score_idx = int(risk_score_model.predict(X)[0])
    return risk_score_le.inverse_transform([risk_score_idx])[0]


