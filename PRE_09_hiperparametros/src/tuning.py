"""Ajuste de hiperparametros para predecir la calidad del vino."""

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

FOLDER = "PRE_09_hiperparametros"

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}


def load_data():
    dataset = pd.read_csv(f"{FOLDER}/data/winequality-red.csv")
    x = dataset.drop(columns=["quality"])
    y = dataset["quality"]
    return x, y


def tune_model(x_train, y_train):
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=PARAM_GRID,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_


def main():
    x, y = load_data()
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )

    estimator = tune_model(x_train, y_train)

    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(estimator, file)


if __name__ == "__main__":
    main()
