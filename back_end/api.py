from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import model_validator
from typing import Optional
from datetime import date
from features_schema import build_feature_vector, assert_feature_length
from model_utils import predict_all, get_location_from_region, get_temperature

app = FastAPI()

# --- Allows front-end to access the back-end ---
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

# --- Define what input data is needed for prediction in the format variable_name: data_type ---
class PredictionRequest(BaseModel):
    type: str # required
    material: str # required
    region: Optional[str] = None # optional (required if there is no exact_location)
    soil_type: str # required
    exact_location: Optional[List[float]] = None # optional, strongly recommended
    last_repair_date: str # required, format "MM-DD-YYYY"
    snapshot_date: Optional[str] = None # default is today, will fix in predict()
    install_year: int # required
    length_m: Optional[float] = None # optional, strongly recommended

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

    # --- Snapshot date defaulting ---
    if data.snapshot_date is None:
        data.snapshot_date = date.today().strftime("%m-%d-%Y")

    # --- Feature list construction ---
    feature_dict = {
    "type": data.type,
    "material": data.material,
    "region": data.region,
    "soil_type": data.soil_type,
    "latitude": lat,
    "longitude": lon,
    "temperature_c": temperature_c,
    "last_repair_date": data.last_repair_date,
    "snapshot_date": data.snapshot_date,
    "install_year": data.install_year,
    "length_m": data.length_m,
    }
    features = build_feature_vector(feature_dict)
    assert_feature_length(features)

    # --- Call the predict_all function from model_utils.py ---
    prediction = predict_all(features)
    return prediction
