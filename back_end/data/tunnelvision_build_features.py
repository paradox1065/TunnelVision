import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def build_features(csv_path):
    # --- Load ---
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    
    # --- Drop irrelevant ---
    #df = df.drop(columns=["asset_id"], errors="ignore")
    
    # --- Fix missing text ---
    df["issue_description"] = df["issue_description"].fillna("na")
    
    # --- Keyword flags FIRST ---
    keywords = ["leak", "flicker", "crack", "overflow", "burning", "broken", "jammed", "noise"]
    for kw in keywords:
        df[f"kw_{kw}"] = df["issue_description"].str.contains(kw, case=False).astype(int)
    
    
    # --- Interaction features ---
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["last_repair_date"] = pd.to_datetime(df["last_repair_date"], errors="coerce")
    
    df["month"] = df["snapshot_date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    df = df.sort_values(["asset_id", "snapshot_date"])
    
    #groupby + shift features
    df["rainfall_prev"] = df.groupby("asset_id")["rainfall_mm"].shift(1)
    df["soil_moisture_prev"] = df.groupby("asset_id")["soil_moisture_pc"].shift(1)
    df["failures_prev"] = df.groupby("asset_id")["num_prev_failures"].shift(1)
    #df["risk_prev"] = df.groupby("asset_id")["risk_score"].shift(1)
    
    # time features
    df["days_since_repair"] = (
        df["snapshot_date"] - df["last_repair_date"]
    ).dt.days.fillna(0)
    df["asset_age_years"] = (df["snapshot_date"].dt.year - df["install_year"]).clip(lower=0)
    
    # delta features
    df["rain_delta"] = (df["rainfall_mm"] - df["rainfall_prev"]).clip(lower=0)
    df["moisture_delta"] = (df["soil_moisture_pc"] - df["soil_moisture_prev"]).clip(lower=0)
    
    # weather stress features
    df["rain_stress"] = df["rainfall_mm"] * df["slope_grade"]
    
    
    lag_cols = [
        "rainfall_prev", "soil_moisture_prev",
        "failures_prev",
        "rain_delta", "moisture_delta"
    ]
    
    # length features
    df["length_age"] = df["length_m"] * df["asset_age_years"]
    df["length_slope"] = df["length_m"] * df["slope_grade"]
    
    df[lag_cols] = df[lag_cols].fillna(0)
    
    #df = df.drop(columns=["asset_id"], errors="ignore")
    
    # --- Encode categorical columns ---
    cat_cols = ["type", "material", "soil_type", "region", "traffic"]
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # add moisture stress column after encoding
    if "soil_type_clay" in df.columns:
        df["moisture_stress"] = df["soil_moisture_pc"] * df["soil_type_clay"]
    else:
        df["moisture_stress"] = df["soil_moisture_pc"]
    
    # --- Convert numeric columns ---
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # --- Fill NaNs and cap extreme values ---
    df[numeric_cols] = df[numeric_cols].fillna(-1)
    df[numeric_cols] = df[numeric_cols].clip(-1e6, 1e6)
    
    # --- TF-IDF text ---
    vectorizer = TfidfVectorizer(stop_words="english", max_features=200, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df["issue_description"])
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # --- Remove targets from features ---
    targets = [
        "failure_next_30d",
        "failure_type_predicted",
        "risk_score",
        "recommended_action",
        "recommended_priority",
        "emergency_response"
    ]
    
    numeric_cols = [c for c in numeric_cols if c not in targets]
    
    
    # --- Numeric matrix (after all engineering) ---
    X_numeric = df[numeric_cols].astype(float).values
    
    # --- Combine ---
    X_final = hstack([X_numeric, X_text])

    df.to_csv("bay_area_infrastructure_clean.csv")
    return X_final, df, vectorizer
