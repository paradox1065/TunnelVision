import pandas as pd
import numpy as np

def build_features_for_inference(feature_dict: dict) -> pd.DataFrame:
    """
    Build a feature DataFrame for a single example from a dict.
    Uses the same logic as train.py without touching train.py.
    """
    df = pd.DataFrame([feature_dict])
    
    # --- Base cleaning ---
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["last_repair_date"] = pd.to_datetime(df.get("last_repair_date"), errors="coerce")
    
    df["days_since_repair"] = (df["snapshot_date"] - df["last_repair_date"]).dt.days.fillna(999)
    df["asset_age_years"] = (df["snapshot_date"].dt.year - df["install_year"]).clip(lower=0)

    # --- SAFE engineered features ---
    # For single row, previous failures = 0
    df["failures_prev"] = 0
    df["recent_repair"] = (df["days_since_repair"] < 180).astype(int)
    df["old_asset"] = (df["asset_age_years"] > 40).astype(int)

    # --- Environmental stress ---
    df["rain_stress"] = df["rainfall_mm"] * df["slope_grade"]
    df["moisture_stress"] = df["soil_moisture_pc"] * df["slope_grade"]
    df["env_stress"] = df["rainfall_mm"] * 0.4 + df["soil_moisture_pc"] * 0.6
    df["temp_stress"] = df["avg_temp_c"] * df["slope_grade"]

    # --- Structural features ---
    df["structural_risk"] = df["asset_age_years"] * df["length_m"]
    df["length_age"] = df["length_m"] * df["asset_age_years"]
    df["length_slope"] = df["length_m"] * df["slope_grade"]
    df["failure_pressure"] = df["num_prev_failures"] * np.log1p(df["days_since_repair"])

    # --- Extra interaction features for risk_score ---
    df["age_length_slope"] = df["asset_age_years"] * df["length_m"] * df["slope_grade"]
    df["failures_env_stress"] = df["failures_prev"] * df["env_stress"]
    df["length_rain_stress"] = df["length_m"] * df["rainfall_mm"]
    df["moisture_age"] = df["soil_moisture_pc"] * df["asset_age_years"]
    df["struct_env_pressure"] = df["structural_risk"] * df["env_stress"]

    # --- Categorical encoding ---
    cat_cols = ["type", "material", "soil_type", "region", "traffic"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # --- Ensure numeric only, fill missing ---
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)

    return df