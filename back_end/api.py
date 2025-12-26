from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator
from typing import Optional, Tuple
from datetime import date
import pandas as pd
from .preprocessing import preprocess_df
from .model_utils import predict_all, get_location_from_region, get_temperature, get_region_from_location, get_traffic_from_region, FEATURE_COLS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request & Response Models
# -------------------------
class PredictionRequest(BaseModel):
    type: str
    material: str
    region: Optional[str] = None
    soil_type: str

    exact_location: Optional[Tuple[float, float]] = None
    last_repair_date: str
    snapshot_date: Optional[str] = None
    install_year: int
    length_m: Optional[float] = 10.0

    @model_validator(mode="after")
    def check_location_or_region(self):
        if self.exact_location is None and self.region is None:
            raise ValueError("Either exact_location or region must be provided.")
        return self

class PredictionResponse(BaseModel):
    failure_in_30_days: bool
    failure_type: str
    risk_score: int
    recommended_action: str
    priority: int

# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    # 1️⃣ Determine location & region
    if data.exact_location is not None:
        lat, lon = data.exact_location
        region = data.region or get_region_from_location(lat, lon)
    else:
        lat, lon = get_location_from_region(data.region)
        region = data.region

    traffic = get_traffic_from_region(region)
    temperature_c = get_temperature(lat, lon)
    snapshot_date = data.snapshot_date or date.today().strftime("%Y-%m-%d")

    # 2️⃣ Build feature dictionary
        # Material degradation factors
    material_risk = {
        "cast_iron": 1.5,
        "concrete": 1.3,
        "steel": 1.2,
        "pvc": 0.7,
        "hdpe": 0.6,
    }
    mat_factor = material_risk.get(data.material.lower(), 1.0)

    # Soil corrosion factors
    soil_risk = {
        "clay": 1.4,
        "sandy": 1.1,
        "loam": 1.0,
        "rocky": 0.9,
    }
    soil_factor = soil_risk.get(data.soil_type.lower(), 1.0)

    # Quick estimate of asset age for failure calculation
    asset_age = date.today().year - data.install_year
    days_since_repair = (date.today() - pd.to_datetime(data.last_repair_date)).days

    # Estimate failures based on age, material, and repair history
    # Training data mean = 9, so scale to that
    base_failures = 0

    if asset_age > 50:
        base_failures = int(12 + mat_factor * 3 + soil_factor * 2)
    elif asset_age > 40:
        base_failures = int(9 + mat_factor * 2 + soil_factor * 1)
    elif asset_age > 30:
        base_failures = int(6 + mat_factor * 1.5)
    elif asset_age > 20:
        base_failures = int(3 + mat_factor)
    elif asset_age > 10:
        base_failures = int(1 + mat_factor * 0.5)
    else:
        base_failures = 0

    # Add more if repair history is bad
    if days_since_repair > 365 * 7:
        base_failures += 3
    elif days_since_repair > 365 * 5:
        base_failures += 2
    elif days_since_repair > 365 * 3:
        base_failures += 1

    # High traffic + problematic soil = more failures
    if traffic == "high" and soil_factor > 1.2:
        base_failures += 2

    # Cap at 18 (max in training data)
    base_failures = min(base_failures, 18)

    # Environmental stress based on region
    rainfall = 30.0 if region in ["San Francisco", "Marin", "Sonoma"] else 20.0
    soil_moisture = 45.0 if data.soil_type.lower() == "clay" else 30.0
    slope = 4.0 if region in ["San Francisco", "Marin"] else 2.0

    # Build feature dictionary with SMART defaults
    feature_dict = {
        "type": data.type,
        "material": data.material,
        "region": region,
        "soil_type": data.soil_type,
        "traffic": traffic,
        "latitude": lat,
        "longitude": lon,
        "avg_temp_c": temperature_c,
        "rainfall_mm": rainfall,
        "soil_moisture_pc": soil_moisture,
        "slope_grade": slope,
        "num_prev_failures": base_failures,
        "failures_prev": base_failures,
        "last_repair_date": data.last_repair_date,
        "snapshot_date": snapshot_date,
        "install_year": data.install_year,
        "length_m": data.length_m,
    }

    df = pd.DataFrame([feature_dict])

    # 3️⃣ Preprocess
    df_processed = preprocess_df(df)

    # 4️⃣ Drop raw datetime columns (to avoid dtype issues)
    df_processed = df_processed.drop(columns=["snapshot_date", "last_repair_date"], errors="ignore")

    # 5️⃣ Align to training features
    df_processed = df_processed.reindex(columns=FEATURE_COLS, fill_value=0)

    # ✅ 6️⃣ Pass to unified prediction function
    return predict_all(df_processed)


app.mount("/static", StaticFiles(directory="front_end"), name="static")

# Serve HTML pages
@app.get("/")
async def read_index():
    return FileResponse("front_end/index.html")

@app.get("/About.html")
async def read_about():
    return FileResponse("front_end/About.html")

@app.get("/Form.html")
async def read_form():
    return FileResponse("front_end/Form.html")
