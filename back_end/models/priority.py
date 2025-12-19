# model that determines the priority level of asset maintenance based on equipment features
# model that determines the priority level of asset maintenance based on equipment features
import pandas as pd
import os
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import csr_matrix
from tunnelvision_build_features import build_features as bf

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/bay_area_infrastructure_modified.csv")
print("Loading CSV from:", csv_path)

# --- Build features ---
X, df, feature_cols = bf(csv_path, target="recommended_priority")

y = df["recommended_priority"].astype(int)

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

model = rfr(
    n_estimators=150,
    max_depth=2,
    random_state=42
)
model.fit(X_train, y_train)

import math
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
import numpy as np
from sklearn.metrics import mean_absolute_error

# y_train, y_test are your training and testing target arrays
baseline_pred = np.full_like(y_test, y_train.mean())
baseline_mae = mean_absolute_error(y_test, baseline_pred)
print("Baseline MAE:", baseline_mae)
