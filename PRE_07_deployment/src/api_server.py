"""API REST para servir predicciones del modelo de precios de casas."""

import pickle

import pandas as pd
from flask import Flask, jsonify, request

from .train_model import FEATURES

FOLDER = "PRE_07_deployment"

app = Flask(__name__)

with open(f"{FOLDER}/submission/house_predictor.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    x = pd.DataFrame([{feature: payload[feature] for feature in FEATURES}])
    prediction = model.predict(x)[0]
    return jsonify({"price": float(prediction)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
