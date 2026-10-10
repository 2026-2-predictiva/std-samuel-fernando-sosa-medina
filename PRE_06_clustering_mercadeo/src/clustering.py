"""Segmentacion de mercado (adolescentes en redes sociales) via K-Means."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FOLDER = "PRE_06_clustering_mercadeo"

INTEREST_COLUMNS = [
    "basketball", "football", "soccer", "softball", "volleyball", "swimming",
    "cheerleading", "baseball", "tennis", "sports", "cute", "sex", "sexy",
    "hot", "kissed", "dance", "band", "marching", "music", "rock", "god",
    "church", "jesus", "bible", "hair", "dress", "blonde", "mall", "shopping",
    "clothes", "hollister", "abercrombie", "die", "death", "drunk", "drugs",
]


def load_data():
    dataset = pd.read_csv(f"{FOLDER}/data/snsdata.csv")
    dataset["age"] = dataset["age"].fillna(
        dataset.groupby("gradyear")["age"].transform("mean")
    )
    dataset["age"] = dataset["age"].fillna(dataset["age"].mean())
    dataset["female"] = (dataset["gender"] == "F").astype(int)
    dataset["no_gender"] = dataset["gender"].isna().astype(int)
    return dataset


def main(n_clusters=5):
    dataset = load_data()

    scaler = StandardScaler()
    scaled_interests = scaler.fit_transform(dataset[INTEREST_COLUMNS])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    dataset["cluster"] = kmeans.fit_predict(scaled_interests)

    segmented = dataset[
        ["gradyear", "gender", "age", "friends", "female", "no_gender", "cluster"]
        + INTEREST_COLUMNS
    ]
    segmented.to_csv(f"{FOLDER}/submission/segmented.csv")

    return segmented


if __name__ == "__main__":
    main()
