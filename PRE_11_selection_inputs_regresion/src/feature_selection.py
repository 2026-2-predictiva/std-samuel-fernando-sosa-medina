"""Seleccion de variables para un modelo de regresion (MPG)."""

import pickle

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FOLDER = "PRE_11_selection_inputs_regresion"


def load_data():
    dataset = pd.read_csv(f"{FOLDER}/data/auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")
    y = dataset.pop("MPG")
    x = dataset
    return x, y


def build_estimator(n_features_to_select=5):
    return Pipeline(
        steps=[
            (
                "selector",
                RFE(
                    estimator=LinearRegression(),
                    n_features_to_select=n_features_to_select,
                ),
            ),
            ("regressor", LinearRegression()),
        ]
    )


def main():
    x, y = load_data()
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42,
    )

    estimator = build_estimator()
    estimator.fit(x_train, y_train)

    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    main()
