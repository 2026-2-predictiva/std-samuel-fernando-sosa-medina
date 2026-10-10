"""Aplicacion web para estimar el precio de una casa."""

import pickle

import pandas as pd
from flask import Flask, render_template, request

from .train_model import FEATURES

FOLDER = "PRE_07_deployment"

app = Flask(__name__)

with open(f"{FOLDER}/submission/house_predictor.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        house = {
            "bedrooms": float(request.form["bedrooms"]),
            "bathrooms": float(request.form["bathrooms"]),
            "sqft_living": float(request.form["sqft_living"]),
            "sqft_lot": float(request.form["sqft_lot"]),
            "floors": float(request.form["floors"]),
            "waterfront": 1 if request.form["waterfront"] == "Yes" else 0,
            "condition": int(request.form["condition"]),
        }
        x = pd.DataFrame([{feature: house[feature] for feature in FEATURES}])
        prediction = f"${model.predict(x)[0]:,.2f}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
