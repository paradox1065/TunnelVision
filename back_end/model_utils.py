from pathlib import Path
from back_end.models.action import predict_action
from back_end.models.failure_30d import predict_failure_30d
from back_end.models.failure_type import predict_failure_type
from back_end.models.priority import predict_priority
from back_end.models.risk_score import predict_risk_score
from .preprocessing import preprocess_df
import joblib
import requests
import pandas as pd

BASE_DIR = Path(__file__).parent

# Load the exact feature order used for training
FEATURE_COLS = joblib.load("back_end/feature_cols.pkl")

def calculate_priority_from_risk(risk: int) -> int:
    """Calculate priority from risk score using rules"""
    if risk >= 80:
        return 5
    elif risk >= 65:
        return 4
    elif risk >= 45:
        return 3
    elif risk >= 25:
        return 2
    else:
        return 1

# -------------------------
# Unified prediction function
# -------------------------
def predict_all(X):

    risk = int(predict_risk_score(X))  # calculate risk first

    """
    X should be a preprocessed DataFrame (1 row for a single asset) 
    with columns aligned to FEATURE_COLS.
    """
    return {
        "failure_in_30_days": predict_failure_30d(X),
        "failure_type": predict_failure_type(X),
        "risk_score": risk,
        "recommended_action": predict_action(X),
        "priority": calculate_priority_from_risk(risk),
    }


# --- Region coordinates ---
REGION_COORDINATES = {
    "Contra Costa": (37.9199, -121.9358),
    "Alameda": (37.756944, -122.274444),
    "Sonoma": (38.445595, -122.595747),
    "Santa Clara": (37.35411, -121.95524),
    "Napa": (38.297804, -122.28636),
    "San Francisco": (37.715, -122.4285),
    "Marin": (37.868538, -122.5091404),
    "San Mateo": (37.563, -122.324),
    "Solano": (38.316, -122.018),
}


def get_location_from_region(region: str) -> tuple[float, float]:
    return REGION_COORDINATES.get(region.strip(), (37.338207, -121.886330))


def get_temperature(lat: float, lon: float) -> float:
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current_weather=true"
        )
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json().get("current_weather", {}).get("temperature", 15.0)
    except Exception:
        return 15.0


def get_region_from_location(lat: float, lon: float):
    if 36.89238291632208 <= lat <= 37.48534080282651 and -122.20259387759805 <= lon <= -121.21382434174066:
        return "Santa Clara"
    elif 37.45422122137626 <= lat <= 37.90527 and -122.34177172635131 <= lon <= -121.46973192736596:
        return "Alameda"
    elif 38.11230588756946 <= lat <= 38.85190504809042 and -123.5324716054025 <= lon <= -122.34869474441764:
        return "Sonoma"
    elif 37.71888575931279 <= lat <= 38.10135722489106 and -122.4300892162658 <= lon <= -121.53333017888399:
        return "Contra Costa"
    elif 38.153660062350056 <= lat <= 38.86397170801056 and -122.64656355512427 <= lon <= -122.06154157974197:
        return "Napa"
    elif 37.70841124861289 <= lat <= 37.81141713562808 and -122.51793825593296 <= lon <= -122.3278475588094:
        return "San Francisco"
    elif 37.8152822778324 <= lat <= 38.32117636904628 and -123.03056485363471 <= lon <= -122.41258389372383:
        return "Marin"
    elif 37.10780636280471 <= lat <= 37.70994030426103 and -122.52143637988625 <= lon <= -122.20259387759806:
        return "San Mateo"
    elif 38.0398174918701 <= lat <= 38.54067561081478 and -122.40928351247523 <= lon <= -121.59217535437085:
        return "Solano"


def get_traffic_from_region(region: str) -> str:
    region_traffic = {
        "San Francisco": "high",
        "Santa Clara": "high",
        "Alameda": "medium",
        "Contra Costa": "medium",
        "San Mateo": "medium",
        "Marin": "low",
        "Napa": "low",
        "Sonoma": "low",
        "Solano": "low",
    }
    return region_traffic.get(region.strip(), "low")
