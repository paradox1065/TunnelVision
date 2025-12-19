import joblib
import requests

failure_model = joblib.load("models/failure_30d_features.pkl")
failure_type_model = joblib.load("models/failure_type_features.pkl")
risk_model = joblib.load("models/risk_score_features.pkl")
action_model = joblib.load("models/recommended_action_features.pkl")
priority_model = joblib.load("models/priority_features.pkl")

def predict_all(features: list):
    return {
        "failure_in_30_days": bool(failure_model.predict([features])[0]),
        "failure_type": failure_type_model.predict([features])[0],
        "risk_score": int(risk_model.predict([features])[0]),
        "recommended_action": action_model.predict([features])[0],
        "priority": int(priority_model.predict([features])[0]),
    }

REGION_COORDINATES = {
    "Contra Costa": (37.9199, -121.9358),
    "Alameda": (37.756944, -122.274444),
    "Sonoma": (38.445595361907564, -122.59574748819978),
    "Santa Clara": (37.3541100, -121.9552400),
    "Napa": (38.297804, -122.28636),
    "San Francisco": (37.715, -122.4285),
    "Marin": (37.868538, -122.5091404),
    "San Mateo": (37.563, 122.324),
    "Solano": (38.316, -122.018),
}

def get_location_from_region(region: str) -> tuple[float, float]:
    """
    Returns (lat, lon) for a given region.
    Falls back to a default location if region is unknown.
    """
    region = region.strip()

    if region in REGION_COORDINATES:
        return REGION_COORDINATES[region]

    # --- Fallback: center of your service area ---
    # (This prevents crashes and keeps predictions working)
    return (37.338207, -121.886330)  # San Jose as default

def get_temperature(lat: float, lon: float) -> float:
    try:
        url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m"
        )

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("current", {}).get("temperature_2m", 15.0)
    
    except Exception:
        return 15.0 # Default temperature if API call fails
