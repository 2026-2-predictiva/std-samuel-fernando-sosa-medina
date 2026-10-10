"""Cliente de ejemplo para consumir la API de prediccion de precios."""

import requests

API_URL = "http://localhost:5000/predict"


def get_prediction(house, api_url=API_URL):
    response = requests.post(api_url, json=house)
    response.raise_for_status()
    return response.json()["price"]


if __name__ == "__main__":
    example_house = {
        "bedrooms": 3,
        "bathrooms": 1,
        "sqft_living": 1180,
        "sqft_lot": 5650,
        "floors": 1,
        "waterfront": 0,
        "condition": 3,
    }
    print(get_prediction(example_house))
