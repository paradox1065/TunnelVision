import pandas as pd
import numpy as np

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # -------------------------
    # Dates
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
    # Engineered features
    # -------------------------
    df["failures_prev"] = df.get("failures_prev", 0)
    df["recent_repair"] = (df["days_since_repair"] < 180).astype(int)
    df["old_asset"] = (df["asset_age_years"] > 40).astype(int)

    df["rain_stress"] = df["rainfall_mm"] * df["slope_grade"]
    df["moisture_stress"] = df["soil_moisture_pc"] * df["slope_grade"]
    df["env_stress"] = df["rainfall_mm"] * 0.4 + df["soil_moisture_pc"] * 0.6
    df["temp_stress"] = df["avg_temp_c"] * df["slope_grade"]

    df["structural_risk"] = df["asset_age_years"] * df["length_m"]
    df["length_age"] = df["length_m"] * df["asset_age_years"]
    df["length_slope"] = df["length_m"] * df["slope_grade"]
    df["failure_pressure"] = df["num_prev_failures"] * np.log1p(df["days_since_repair"])

    df["age_length_slope"] = df["asset_age_years"] * df["length_m"] * df["slope_grade"]
    df["failures_env_stress"] = df["failures_prev"] * df["env_stress"]
    df["length_rain_stress"] = df["length_m"] * df["rainfall_mm"]
    df["moisture_age"] = df["soil_moisture_pc"] * df["asset_age_years"]
    df["struct_env_pressure"] = df["structural_risk"] * df["env_stress"]

    # -------------------------
    # One-hot encoding
    # -------------------------
    cat_cols = ["type", "material", "soil_type", "region", "traffic"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # -------------------------
    # Cleanup
    # -------------------------
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
