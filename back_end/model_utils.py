import joblib

failure_model = joblib.load("models/failure_30d.pkl")
failure_type_model = joblib.load("models/failure_type.pkl")
risk_model = joblib.load("models/risk_score.pkl")
action_model = joblib.load("models/recommended_action.pkl")
priority_model = joblib.load("models/priority.pkl")
emergency_model = joblib.load("models/emergency.pkl")

def predict_all(features: list):
    return {
        "failure_in_30_days": bool(failure_model.predict([features])[0]),
        "failure_type": failure_type_model.predict([features])[0],
        "risk_score": int(risk_model.predict([features])[0]),
        "recommended_action": action_model.predict([features])[0],
        "priority": int(priority_model.predict([features])[0]),
        "emergency_required": bool(emergency_model.predict([features])[0]),
    }

def get_location_from_region(region: str) -> tuple[float, float]:
    # Aarushi's logic for getting lat, lon from region
    return lat, lon

def get_temperature(lat: float, lon: float) -> float:
    # Call weather API here
    return temperature_c