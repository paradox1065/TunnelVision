from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Optional, Tuple
from datetime import date

# Relative imports
from .features_schema import build_feature_vector, assert_feature_length
from .model_utils import predict_all, get_location_from_region, get_temperature

app = FastAPI()

# Allow front-end to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local dev, later restrict to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class PredictionRequest(BaseModel):
    type: str
    material: str
    region: Optional[str] = None
    soil_type: str
    exact_location: Optional[Tuple[float, float]] = None
    last_repair_date: str
    snapshot_date: Optional[str] = None
    install_year: int
    length_m: Optional[float] = None

    @model_validator(mode="after")
    def check_location_or_region(self):
        if self.exact_location is None and self.region is None:
            raise ValueError("Either exact_location or region must be provided.")
        return self

# Output schema
class PredictionResponse(BaseModel):
    failure_in_30_days: bool
    failure_type: str
    risk_score: int
    recommended_action: str
    priority: int

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    # Resolve location
    if data.exact_location is not None:
        lat, lon = data.exact_location
    else:
        lat, lon = get_location_from_region(data.region)

    # Temperature
    temperature_c = get_temperature(lat, lon)

    # Default snapshot date
    snapshot_date = data.snapshot_date or date.today().strftime("%m-%d-%Y")

    # Feature dictionary
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

    prediction = predict_all(features)
    return prediction