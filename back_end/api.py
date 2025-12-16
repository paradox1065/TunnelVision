from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import model_validator
from typing import Optional
from model_utils import predict_all

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
    temperature_c: float # required
    date_of_last_repair: str # required
    date_issue_was_observed: str # required
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
    emergency_required: bool

@app.post("/predict", response_model=PredictionResponse)

# --- Extract features from data in the format data.variable_name ---
def predict(data: PredictionRequest):
    if data.exact_location is not None:
        lat, lon = data.exact_location # extract latitude and longitude from the tuple
    else:
        lat, lon = None, None
    features = [
    data.type,
    data.material,
    data.region,
    data.soil_type,
    lat,
    lon,
    data.temperature_c,
    data.date_of_last_repair,
    data.date_issue_was_observed,
    data.install_year,
    data.length_m,
    data.issue_description,
    ]

    # --- Call the predict_all function from model_utils.py ---
    prediction = predict_all(features)
    return prediction