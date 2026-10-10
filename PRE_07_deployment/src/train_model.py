"""Entrena un modelo para predecir el precio de una casa."""

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

FOLDER = "PRE_07_deployment"

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


def load_data():
    dataset = pd.read_csv(f"{FOLDER}/data/house_data.csv")
    x = dataset[FEATURES]
    y = dataset["price"]
    return x, y


def train_model(x, y):
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42,
    )
    model = RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def main():
    x, y = load_data()
    model = train_model(x, y)
    save_model(model, f"{FOLDER}/submission/house_predictor.pkl")


if __name__ == "__main__":
    main()
