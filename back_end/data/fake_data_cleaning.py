import pandas as pd
from sklearn.model_selection import train_test_split as tts

# loading 
df = pd.read_csv("fake_data.csv")
df = df.replace(["NA", "NaN", "not_available", ""],pd.NA)

df.columns = df.columns.str.strip()

df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
df["pressure_psi"] = pd.to_numeric(df["pressure_psi"], errors="coerce")
df["cost_estimate"] = pd.to_numeric(df["cost_estimate"], errors="coerce")

#median_date = df["last_service_date"].median()
#df["last_service_date"] = df["last_service_date"].fillna(median_date)

df["last_service_date"] = pd.to_datetime(df["last_service_date"], errors="coerce")
df['year'] = df["last_service_date"].dt.year
df['month'] = df["last_service_date"].dt.month
df['day'] = df["last_service_date"].dt.day
df['day_of_week'] = df["last_service_date"].dt.dayofweek
df['days_since_maintenance'] = (pd.Timestamp.today() - df["last_service_date"]).dt.days
df = df.drop(columns=["last_service_date"])

df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].mean())
df["pressure_psi"] = df["pressure_psi"].fillna(df["pressure_psi"].mean())
df["age_years"] = df["age_years"].fillna(df["age_years"].mean())

df = pd.get_dummies(df, columns = ["home_id"])
df = pd.get_dummies(df, columns = ["room"])
df = pd.get_dummies(df, columns = ["component"])
df = pd.get_dummies(df, columns = ["issue_reported"])

# X = df.drop("cost_estimate", axis=1)
# y = df["cost_estimate"]

# X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=3)

df.to_csv("fake_clean_data.csv", index=False)
pd.set_option('display.max_columns', None)
print(df)