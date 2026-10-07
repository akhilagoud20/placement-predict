from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

from scipy.cluster.hierarchy import linkage, dendrogram


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "raw_placement_data.csv"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA)

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# Select clustering features
# ---------------------------------------------------------
features = [
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

X = df[features].copy()


# ---------------------------------------------------------
# Convert college tier to numeric
# ---------------------------------------------------------
if X["college_tier"].dtype == "object":
    X["college_tier"] = (
        X["college_tier"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )


# ---------------------------------------------------------
# Remove missing values
# ---------------------------------------------------------
X = X.dropna()

print("Data used for clustering:", X.shape)


# ---------------------------------------------------------
# Standardization
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# Use a sample for silhouette calculations
# This avoids extremely slow distance calculations
# on all 100,000 records.
# ---------------------------------------------------------
SAMPLE_SIZE = min(5000, len(X_scaled))

rng = np.random.RandomState(42)

sample_indices = rng.choice(
    len(X_scaled),
    size=SAMPLE_SIZE,
    replace=False
)

X_sample = X_scaled[sample_indices]


# ---------------------------------------------------------
# K-Means: test K = 2 to 10
# ---------------------------------------------------------
k_values = range(2, 11)

inertias = []
silhouette_scores = []

print("\nRunning K-Means experiments...")

for k in k_values:

    print(f"Running K-Means K={k}...")

    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels_sample = kmeans.fit_predict(X_sample)

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        X_sample,
        labels_sample
    )

    inertias.append(inertia)
    silhouette_scores.append(silhouette)


# ---------------------------------------------------------
# Elbow plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")

plt.grid(True)

plt.savefig(
    OUT / "clustering_kmeans_elbow.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Silhouette plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Score")

plt.grid(True)

plt.savefig(
    OUT / "clustering_kmeans_silhouette.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Find best K
# ---------------------------------------------------------
best_index = np.argmax(silhouette_scores)
best_k = list(k_values)[best_index]

print("\nBest K:", best_k)
print("Best silhouette score:", silhouette_scores[best_index])


# ---------------------------------------------------------
# Final K-Means using complete dataset
# ---------------------------------------------------------
print("\nRunning final K-Means...")

final_kmeans = KMeans(
    n_clusters=best_k,
    n_init=10,
    random_state=42
)

kmeans_labels = final_kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(
    X_sample,
    kmeans_labels[sample_indices]
)

kmeans_db = davies_bouldin_score(
    X_scaled,
    kmeans_labels
)

print("K-Means completed.")
print("K-Means Silhouette Score:", kmeans_silhouette)
print("K-Means Davies-Bouldin Score:", kmeans_db)


# ---------------------------------------------------------
# Agglomerative Clustering
# ---------------------------------------------------------
print("\nRunning Agglomerative Clustering...")

# Use a sample because hierarchical clustering on
# 100,000 records is extremely expensive.
hier_sample_size = min(5000, len(X_scaled))

hier_indices = rng.choice(
    len(X_scaled),
    size=hier_sample_size,
    replace=False
)

X_hier = X_scaled[hier_indices]

agg = AgglomerativeClustering(
    n_clusters=best_k
)

agg_labels = agg.fit_predict(X_hier)

agg_silhouette = silhouette_score(
    X_hier,
    agg_labels
)

agg_db = davies_bouldin_score(
    X_hier,
    agg_labels
)

print("Agglomerative completed.")
print("Agglomerative Silhouette Score:", agg_silhouette)
print("Agglomerative Davies-Bouldin Score:", agg_db)


# ---------------------------------------------------------
# Hierarchical Dendrogram
# ---------------------------------------------------------
print("\nCreating hierarchical dendrogram...")

dendrogram_sample_size = min(1000, len(X_scaled))

dendro_indices = rng.choice(
    len(X_scaled),
    size=dendrogram_sample_size,
    replace=False
)

X_dendro = X_scaled[dendro_indices]

linkage_matrix = linkage(
    X_dendro,
    method="ward"
)

plt.figure(figsize=(12, 6))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Distance")

plt.grid(True)

plt.savefig(
    OUT / "hierarchical_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------
print("\nRunning DBSCAN...")

dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(X_sample)

unique_labels = set(dbscan_labels)

if len(unique_labels) > 1:

    dbscan_silhouette = silhouette_score(
        X_sample,
        dbscan_labels
    )

    dbscan_db = davies_bouldin_score(
        X_sample,
        dbscan_labels
    )

else:

    dbscan_silhouette = np.nan
    dbscan_db = np.nan


noise_count = np.sum(dbscan_labels == -1)

print("DBSCAN completed.")
print("DBSCAN noise points:", noise_count)
print("DBSCAN Silhouette Score:", dbscan_silhouette)
print("DBSCAN Davies-Bouldin Score:", dbscan_db)


# ---------------------------------------------------------
# Comparison table
# ---------------------------------------------------------
comparison = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "Agglomerative",
        "DBSCAN"
    ],

    "Silhouette_Score": [
        kmeans_silhouette,
        agg_silhouette,
        dbscan_silhouette
    ],

    "Davies_Bouldin_Score": [
        kmeans_db,
        agg_db,
        dbscan_db
    ]
})


comparison.to_csv(
    OUT / "clustering_comparison.csv",
    index=False
)


# ---------------------------------------------------------
# Final output
# ---------------------------------------------------------
print("\n======================================")
print("Clustering Experiment Completed!")
print("======================================")

print("\nResults saved in:")
print(OUT)

print("\nGenerated files:")
print("1. clustering_kmeans_elbow.png")
print("2. clustering_kmeans_silhouette.png")
print("3. hierarchical_dendrogram.png")
print("4. clustering_comparison.csv")