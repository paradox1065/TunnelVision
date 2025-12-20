from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Optional, List
from datetime import date
from features_schema import build_feature_vector, assert_feature_length
from model_utils import predict_all, get_location_from_region, get_temperature

app = FastAPI()

# --- CORS setup ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://fictional-disco-976pvp495vj72x44p-5500.app.github.dev"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Model ---
class PredictionRequest(BaseModel):
    type: str
    material: str
    region: Optional[str] = None
    soil_type: str
    exact_location: Optional[List[float]] = None
    last_repair_date: str
    snapshot_date: Optional[str] = None
    install_year: int
    length_m: Optional[float] = None

    @model_validator(mode="after")
    def check_location_or_region(self):
        if self.exact_location is None and self.region is None:
            raise ValueError("Either exact_location or region must be provided.")
        return self

# --- Response Model ---
class PredictionResponse(BaseModel):
    failure_in_30_days: bool
    failure_type: str
    risk_score: int
    recommended_action: str
    priority: int

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    # --- Resolve location ---
    if data.exact_location is not None:
        lat, lon = data.exact_location
    else:
        lat, lon = get_location_from_region(data.region)

    # --- Get temperature ---
    temperature_c = get_temperature(lat, lon)

    # --- Snapshot date ---
    snapshot_date = data.snapshot_date or date.today().strftime("%m-%d-%Y")

    # --- Build feature dict ---
    feature_dict = {
        "type": data.type,
        "material": data.material,
        "region": data.region,
        "soil_type": data.soil_type,
        "latitude": lat,
        "longitude": lon,
        "temperature_c": temperature_c,
        "last_repair_date": data.last_repair_date,
        "snapshot_date": snapshot_date,
        "install_year": data.install_year,
        "length_m": data.length_m,
    }

    features = build_feature_vector(feature_dict)
    assert_feature_length(features)

    # --- Predict ---
    prediction = predict_all(features)
    return prediction
