import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.sparse import hstack

# --- Load ---
df = pd.read_csv("synthetic_city_maintenance.csv")
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

# --- Drop irrelevant ---
df = df.drop(columns=["cost_estimate"], errors="ignore")

# --- Fix missing text ---
df["issue_description"] = df["issue_description"].fillna("na")

# --- Keyword flags FIRST ---
keywords = ["leak", "flicker", "crack", "overflow", "burning", "broken", "jammed", "noise"]
for kw in keywords:
    df[f"kw_{kw}"] = df["issue_description"].str.contains(kw, case=False).astype(int)

# --- Date / cyclical encoding ---
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# --- Interaction features ---
df["age_load"] = df["age_years"] * df["load_factor"]
df["temp_load"] = df["temperature_c"] * df["load_factor"]
df["pressure_load"] = df["pressure_psi"] * df["load_factor"]
df["maintenance_ratio"] = df["days_since_maintenance"] / (df["age_years"] + 1)

# --- Convert numeric columns ---
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
numeric_cols.remove("days_until_failure")  # target

# Fill NaNs
df[numeric_cols] = df[numeric_cols].fillna(-1)

# --- TF-IDF text ---
vectorizer = TfidfVectorizer(stop_words="english", max_features=200, ngram_range=(1, 2))
X_text = vectorizer.fit_transform(df["issue_description"])

# --- Numeric matrix (after all engineering) ---
X_numeric = df[numeric_cols].astype(float).values

# --- Combine ---
X_final = hstack([X_numeric, X_text])

# --- Target ---
df["days_until_failure"] = pd.to_numeric(df["days_until_failure"], errors="coerce").fillna(-1)
y = df["days_until_failure"]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# --- Random Forest ---
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)

# --- Metrics ---
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("RMSE:", rmse)
print("R²:", r2)

cv_scores = cross_val_score(rf, X_final, y, cv=5, scoring="r2")
print("CV R²:", cv_scores.mean())

joblib.dump(model, "model.pkl")
