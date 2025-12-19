# model that determines the type of failure based on equipment features
# model that determines the type of failure based on equipment features
import os
import numpy as np
from tunnelvision_build_features import build_features as bf
from sklearn.preprocessing import LabelEncoder as le
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import joblib


# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "../data/bay_area_infrastructure_modified.csv")
print("Loading CSV from:", csv_path)

# --- Build features ---
X, df, feature_cols = bf(csv_path, target="failure_type_predicted")


# --- Merge rare classes ---
merge_map = {
    "crack": "structural_damage",
    "corrosion": "structural_damage",
    "erosion": "structural_damage"
}

df["failure_type_merged"] = df["failure_type_predicted"].replace(merge_map)

# --- Encode merged labels ---
le_failure = le()
df["failure_type_encoded"] = le_failure.fit_transform(
    df["failure_type_merged"]
)

y_failure = df["failure_type_encoded"]


# --- Train / test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_failure, test_size=0.2, random_state=42
)

# --- Class weights ---
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight = dict(zip(classes, weights))

num_classes = len(classes)

# --- Model ---
model = xgb(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    sample_weight=np.array([class_weight[y] for y in y_train])
)

# --- Predictions ---
probs = model.predict_proba(X_test)
final_thresholds = np.array([0.15, 0.13, 0.57, 0.13, 0.25])

adjusted = probs / final_thresholds
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

joblib.dump(model, os.path.join(model_dir, "failure_type_predicted_xgb.pkl"))

joblib.dump(
    le_failure,
    os.path.join(model_dir, "failure_type_label_encoder.pkl")
)

joblib.dump(
    feature_cols,
    os.path.join(model_dir, "failure_type_features.pkl")
)
