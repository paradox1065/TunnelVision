import joblib

failure_model = joblib.load("models/failure_30d.pkl")
failure_type_model = joblib.load("models/failure_type.pkl")
risk_model = joblib.load("models/risk_score.pkl")
action_model = joblib.load("models/recommended_action.pkl")
priority_model = joblib.load("models/priority.pkl")
emergency_model = joblib.load("models/emergency.pkl")