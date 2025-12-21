from .preprocessing import preprocess_df
import pandas as pd
import joblib

df = pd.read_csv("back_end/data/bay_area_infrastructure_balanced.csv")
df_processed = preprocess_df(df)

targets = [
    "failure_next_30d",
    "failure_type_predicted",
    "risk_score",
    "recommended_action",
    "recommended_priority"
]

feature_cols = [c for c in df_processed.columns if c not in targets]

joblib.dump(feature_cols, "back_end/feature_cols.pkl")
