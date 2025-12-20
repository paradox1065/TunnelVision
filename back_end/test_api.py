import requests
url = "http://127.0.0.1:5500/predict"
data = {
    "type": "Road",
    "material": "Asphalt",
    "region": "Santa Clara",
    "soil_type": "Gravel",
    "latitude": 37.35411,
    "longitude": -121.95524,
    "temperature_c": 15.0,
    "last_repair_date": "2020-01-01",
    "snapshot_date": "2025-12-19",
    "install_year": 2012,
    "length_m": 120.0
}
response = requests.post(url, json=data)
print(response.status_code)