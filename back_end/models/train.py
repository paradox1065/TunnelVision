import pandas as pd
import numpy as np

def build_features(csv_path, target=None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # -------------------------
    # Base cleaning
    # -------------------------
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["last_repair_date"] = pd.to_datetime(df["last_repair_date"], errors="coerce")

    df["days_since_repair"] = (
        df["snapshot_date"] - df["last_repair_date"]
    ).dt.days.fillna(999)

    df["asset_age_years"] = (
        df["snapshot_date"].dt.year - df["install_year"]
    ).clip(lower=0)

    # -------------------------
    # SAFE engineered features
    # -------------------------
    df["failures_prev"] = df.groupby("asset_id")["num_prev_failures"].shift(1).fillna(0)
    df["recent_repair"] = (df["days_since_repair"] < 180).astype(int)
    df["old_asset"] = (df["asset_age_years"] > 40).astype(int)

    # -------------------------
    # Environmental stress
    # -------------------------
    df["rain_stress"] = df["rainfall_mm"] * df["slope_grade"]
    df["moisture_stress"] = df["soil_moisture_pc"] * df["slope_grade"]
    df["env_stress"] = df["rainfall_mm"] * 0.4 + df["soil_moisture_pc"] * 0.6
    df["temp_stress"] = df["avg_temp_c"] * df["slope_grade"]

    # -------------------------
    # Structural features
    # -------------------------
    df["structural_risk"] = df["asset_age_years"] * df["length_m"]
    df["length_age"] = df["length_m"] * df["asset_age_years"]
    df["length_slope"] = df["length_m"] * df["slope_grade"]
    df["failure_pressure"] = df["num_prev_failures"] * np.log1p(df["days_since_repair"])

    # -------------------------
    # Extra interaction features for risk_score
    # -------------------------
    # Combine age, slope, and length — older long steep assets are riskier
    df["age_length_slope"] = df["asset_age_years"] * df["length_m"] * df["slope_grade"]

    # Combine previous failures with environmental stress — assets with many failures in harsh conditions are higher risk
    df["failures_env_stress"] = df["failures_prev"] * df["env_stress"]

    # Length * rainfall stress — longer assets in heavy rain are riskier
    df["length_rain_stress"] = df["length_m"] * df["rainfall_mm"]

    # Soil moisture * asset age — old assets in wet soil are at risk
    df["moisture_age"] = df["soil_moisture_pc"] * df["asset_age_years"]

    # Combined structural + environmental pressure
    df["struct_env_pressure"] = df["structural_risk"] * df["env_stress"]

    # -------------------------
    # Categorical encoding
    # -------------------------
    cat_cols = ["type", "material", "soil_type", "region", "traffic"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # -------------------------
    # Numeric features only, drop targets and asset_id
    # -------------------------
    targets = [
        "failure_next_30d",
        "failure_type_predicted",
        "risk_score",
        "recommended_action",
        "recommended_priority"
    ]
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in targets + ["asset_id"]]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)

    X = df[feature_cols].values

    return X, df, feature_cols
