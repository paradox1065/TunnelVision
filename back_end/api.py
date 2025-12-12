from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# allows front-end to access the back-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    ... # Define what input data is needed for prediction in the format variable_name: data_type

@app.post("/predict")
def predict(data: PredictionRequest):
    features = [
        ... # extract features from input_data in the format data.variable_name (variable name refers to the ones defined in PredictionRequest)
    ]
    prediction = ... # use the trained model to predict the output based on the features extracted from input_data
    return {"prediction": prediction}
