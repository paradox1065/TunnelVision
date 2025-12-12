import pandas as pd
import numpy as np
import random

# ---------------------------------
# CONFIG
# ---------------------------------
NUM_ROWS = 2000

asset_types = [
    "streetlight", 
    "pipe", 
    "pump", 
    "ac_unit", 
    "generator"
]

issue_keywords = {
    "streetlight": ["flicker", "dim", "burning", "broken"],
    "pipe": ["leak", "crack", "overflow", "rust"],
    "pump": ["noise", "jammed", "overheat"],
    "ac_unit": ["not cooling", "leaking", "overheat"],
    "generator": ["vibration", "smoke", "noise"]
}

# ---------------------------------
# FUNCTION TO GENERATE REALISTIC DAYS UNTIL FAILURE
# ---------------------------------
def compute_days_until_failure(age, load, temp, pressure, maint, keyword_weight):
    """
    The formula is arbitrary but consistent, which is what matters.
    Lower score = sooner failure.
    """
    base = (
        2000 
        - age * 15 
        - load * 25 
        - temp * 1.5 
        - pressure * 0.8 
        + maint * 2.0 
        - keyword_weight * 50
    )
    
    noise = np.random.randint(-7, 7)   # small real-world randomness
    return max(1, int(base + noise))   # never below 1


# ---------------------------------
# MAIN GENERATION LOOP
# ---------------------------------
rows = []

for _ in range(NUM_ROWS):
    asset = random.choice(asset_types)
    
    # numeric features (ranges chosen to feel realistic)
    age_years = np.random.randint(0, 40)
    load_factor = np.random.uniform(0.1, 1.0)
    temperature_c = np.random.randint(5, 45)
    
    pressure_psi = {
        "pipe": np.random.randint(30, 100),
        "pump": np.random.randint(50, 200),
        "streetlight": np.random.randint(1, 5),
        "ac_unit": np.random.randint(5, 30),
        "generator": np.random.randint(20, 70)
    }[asset]
    
    days_since_maintenance = np.random.randint(0, 365)
    
    # pick a keyword + build description
    keywords = issue_keywords[asset]
    chosen_kw = random.choice(keywords)
    description = f"{asset} showing {chosen_kw}"
    
    # convert keyword into weight for failure formula
    keyword_weight = keywords.index(chosen_kw) + 1
    
    # compute fake "days until failure"
    days_until_failure = compute_days_until_failure(
        age_years, load_factor, temperature_c,
        pressure_psi, days_since_maintenance,
        keyword_weight
    )
    
    # date features
    month = np.random.randint(1, 13)
    day_of_week = np.random.randint(0, 7)
    
    rows.append({
        "asset_type": asset,
        "age_years": age_years,
        "load_factor": load_factor,
        "temperature_c": temperature_c,
        "pressure_psi": pressure_psi,
        "days_since_maintenance": days_since_maintenance,
        "month": month,
        "day_of_week": day_of_week,
        "issue_description": description,
        "days_until_failure": days_until_failure
    })

# Convert to DataFrame and save
df = pd.DataFrame(rows)
df.to_csv("synthetic_city_maintenance.csv", index=False)

print("Generated synthetic_city_maintenance.csv with", len(df), "rows.")
