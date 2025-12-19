# model that determines recommended actions based on equipment features
# model that determines the type of failure based on equipment features
import os
import numpy as np
from train import build_features as bf
from sklearn.preprocessing import LabelEncoder as le
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import joblib


# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/bay_area_infrastructure_balanced.csv")
# --- Build features ---
X, df, feature_cols = bf(csv_path, target="recommended_action")

le_failure = le()
df['recommended_action_encoded'] = le_failure.fit_transform(df['recommended_action'])

# X is your features from build_features()
y_failure = df["recommended_action_encoded"]

drop_feats = [
    "length_age", 
    "old_asset", 
    "asset_age_years", 
    "struct_env_pressure", 
    "env_stress", 
    "recent_repair"
]

# get indices of features to keep
keep_mask = [f not in drop_feats for f in feature_cols]

# reduce feature matrix
X_reduced = X[:, keep_mask]

# update feature list
feature_cols_reduced = [f for f in feature_cols if f not in drop_feats]

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_failure, test_size = 0.2, random_state=42)
model = xgb(max_depth = 6, 
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)
final_threshold = [0.21, 0.38, 0.22, 0.34]

adjusted = probs / final_threshold
y_pred = np.argmax(adjusted, axis=1)

print("\n\n\n--- Test Metrics ---")
print(classification_report(y_test, y_pred))
cv_acc = cross_val_score(model, X, y_failure, cv=5, scoring="accuracy")
print("CV Accuracy:", cv_acc.mean())
cv_f1 = cross_val_score(model, X, y_failure, cv=5, scoring="f1_macro")
print("CV Macro F1:", cv_f1.mean())
print("\n\n\n")

model_dir = os.path.join(script_dir, "../models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "recommended_action_xgb.pkl"))

joblib.dump(
    le_failure,
    os.path.join(model_dir, "recommended_action_label_encoder.pkl")
)

joblib.dump(
    feature_cols_reduced,
    os.path.join(model_dir, "recommended_action_features.pkl")
)
