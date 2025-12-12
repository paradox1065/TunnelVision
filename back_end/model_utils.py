import joblib

rf = joblib.load("model.pkl")
def predict_features(features: list):
    '''features = [
        type, 
        material, 
        region, 
        soil_type, 
        exact_location, 
        temperature,_c, 
        date_of_last_repair, 
        date_issue_was_observed, 
        install_year, 
        issue_description
        ]''' # reminder for me, NOT CODE
    
    return str(rf.predict([features])[0])