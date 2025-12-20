from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Optional, Tuple
from datetime import date

from .model_utils import predict_all, get_location_from_region, get_temperature, get_region_from_location, get_traffic_from_region

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
    # Determine latitude/longitude
    lat, lon = data.exact_location or get_location_from_region(data.region)
    region = data.region or get_region_from_location(lat, lon)
    traffic = get_traffic_from_region(region)
    temperature_c = get_temperature(lat, lon)

    snapshot_date = data.snapshot_date or date.today().strftime("%Y-%m-%d")

    # Build full feature dictionary (matches training expectations)
    feature_dict = {
        "type": data.type,
        "material": data.material,
        "region": region,
        "soil_type": data.soil_type,
        "traffic": traffic,

        "latitude": lat,
        "longitude": lon,
        "avg_temp_c": temperature_c,

        # Safe defaults for environmental + history features
        "rainfall_mm": 20.0,
        "soil_moisture_pc": 30.0,
        "slope_grade": 2.0,
        "num_prev_failures": 0,
        "failures_prev": 0,

        "last_repair_date": data.last_repair_date,
        "snapshot_date": snapshot_date,
        "install_year": data.install_year,
        "length_m": data.length_m,
    }

    # Predict all outputs using the unified pipeline
    return predict_all(feature_dict)
