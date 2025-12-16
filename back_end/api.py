from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import model_validator
from typing import Optional
from features_schema import build_feature_vector
from model_utils import predict_all, get_location_from_region, get_temperature

app = FastAPI()

# --- Allows front-end to access the back-end ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Define what input data is needed for prediction in the format variable_name: data_type ---
class PredictionRequest(BaseModel):
    type: str # required
    material: str # required
    region: Optional[str] = None # optional (required if there is no exact_location)
    soil_type: str # required
    exact_location: Optional[tuple[float, float]] = None # optional, strongly recommended
    date_of_last_repair: str # required
    snapshot_date: str # required
    install_year: int # or string?
    length_m: Optional[float] = None # optional, strongly recommended
    issue_description: str # required

    @model_validator(mode="after")
    def check_location_or_region(self):
        if self.exact_location is None and self.region is None:
            raise ValueError("Either exact_location or region must be provided.")
        return self

# --- Define the structure of the prediction response ---
class PredictionResponse(BaseModel):
    failure_in_30_days: bool
    failure_type: str
    risk_score: int
    recommended_action: str
    priority: int

@app.post("/predict", response_model=PredictionResponse)

# --- Extract features from data in the format data.variable_name ---
def predict(data: PredictionRequest):
    # --- Location resolution ---
    if data.exact_location is not None:
        lat, lon = data.exact_location
    else:
        lat, lon = get_location_from_region(data.region)

    # --- Temperature inference ---
    temperature_c = get_temperature(lat, lon)

    # --- Feature list construction ---
    features = build_feature_vector(
    type=data.type,
    material=data.material,
    region=data.region,
    soil_type=data.soil_type,
    lat=lat,
    lon=lon,
    temperature_c=temperature_c,
    date_of_last_repair=data.date_of_last_repair,
    snapshot_date=data.snapshot_date,
    install_year=data.install_year,
    issue_description=data.issue_description,
)


    # --- Call the predict_all function from model_utils.py ---
    prediction = predict_all(features)
    return prediction