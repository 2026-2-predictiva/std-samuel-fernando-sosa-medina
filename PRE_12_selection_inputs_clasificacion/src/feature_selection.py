"""Seleccion de variables para un modelo de clasificacion (enfermedad cardiaca)."""

import pickle

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOLDER = "PRE_12_selection_inputs_clasificacion"


def load_data():
    dataset = pd.read_csv(f"{FOLDER}/data/heart_disease.csv")
    dataset = pd.get_dummies(dataset, columns=["thal"], prefix="thal")
    y = dataset.pop("target")
    x = dataset
    return x, y


def build_estimator(n_features_to_select=8):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "selector",
                RFE(
                    estimator=LogisticRegression(max_iter=1000),
                    n_features_to_select=n_features_to_select,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def main():
    x, y = load_data()
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )

    estimator = build_estimator()
    estimator.fit(x_train, y_train)

    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    main()
