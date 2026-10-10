"""Clustering de patrones de demanda comercial horaria."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FOLDER = "PRE_05_clustering_demanda"
HOURS = [f"H{i:02d}" for i in range(1, 25)]


def load_data():
    dataset = pd.read_csv(f"{FOLDER}/data/demanda_comercial.csv.zip")
    dataset["Fecha"] = pd.to_datetime(dataset["Fecha"])
    return dataset


def plot_raw_demand(dataset):
    plt.figure(figsize=(10, 6))
    for _, row in dataset.sample(n=15, random_state=42).iterrows():
        plt.plot(HOURS, row[HOURS].values, alpha=0.6)
    plt.xlabel("Hora")
    plt.ylabel("Demanda")
    plt.title("Demanda comercial horaria (muestra de dias)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{FOLDER}/submission/demanda-comercial.png")
    plt.close()


def plot_example_patterns(dataset):
    plt.figure(figsize=(10, 6))
    for _, row in dataset.sample(n=5, random_state=7).iterrows():
        plt.plot(HOURS, row[HOURS].values, marker="o", label=str(row["Fecha"].date()))
    plt.xlabel("Hora")
    plt.ylabel("Demanda")
    plt.title("Ejemplos de patrones diarios de demanda")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FOLDER}/submission/demanda-comercial-patrones-ejemplo.png")
    plt.close()


def cluster_profiles(dataset, n_clusters=4):
    features = dataset[HOURS].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    dataset = dataset.copy()
    dataset["cluster"] = labels

    plt.figure(figsize=(10, 6))
    for cluster_id in sorted(dataset["cluster"].unique()):
        profile = dataset[dataset["cluster"] == cluster_id][HOURS].mean()
        plt.plot(HOURS, profile.values, marker="o", label=f"Cluster {cluster_id}")
    plt.xlabel("Hora")
    plt.ylabel("Demanda promedio")
    plt.title("Perfiles de demanda por cluster")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FOLDER}/submission/demanda-comercial-perfiles.png")
    plt.close()

    return dataset


def main():
    dataset = load_data()
    plot_raw_demand(dataset)
    plot_example_patterns(dataset)
    cluster_profiles(dataset)


if __name__ == "__main__":
    main()
