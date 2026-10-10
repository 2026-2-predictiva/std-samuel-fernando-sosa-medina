"""Pipeline de clasificacion de sentimiento de frases financieras."""

import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

FOLDER = "PRE_10_pipelines"


def load_data():
    dataset = pd.read_csv(f"{FOLDER}/data/sentences.csv.zip")
    return dataset["phrase"], dataset["target"]


def build_pipeline():
    return Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("classifier", LinearSVC(random_state=42)),
        ]
    )


def main():
    x, y = load_data()
    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    with open(f"{FOLDER}/submission/estimator.pkl", "wb") as file:
        pickle.dump(pipeline, file)


if __name__ == "__main__":
    main()
