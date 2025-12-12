from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_utils import predict_features

app = FastAPI()

# allows front-end to access the back-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define what input data is needed for prediction in the format variable_name: data_type
class PredictionRequest(BaseModel):
    type: str
    material: str
    region: str
    soil_type: str
    exact_location: tuple[float, float] # maybe a string or a tuple of coordinates
    temperature_c: float
    date_of_last_repair: str
    date_issue_was_observed: str
    install_year: int # or string?
    issue_description: str

@app.post("/predict")
# extract features from data in the format data.variable_name (variable name refers to the ones defined in PredictionRequest)
def predict(data: PredictionRequest):
    lat, lon = data.exact_location # extract latitude and longitude from the tuple
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
    data.issue_description
    ]
    prediction = predict_features(features) # use the trained model to predict the output based on the features extracted from input_data
    return {"prediction": prediction}
