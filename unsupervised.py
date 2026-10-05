import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/forest_fire_processed.csv")

features = [
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI"
]

X = df[features]


# ==========================================
# 2. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("=" * 60)
print("UNSUPERVISED LEARNING")
print("=" * 60)

print("\nOriginal shape:", X.shape)
print("Scaled shape:", X_scaled.shape)


# ==========================================
# 3. FIND GOOD NUMBER OF CLUSTERS
# ==========================================

inertias = []
silhouette_scores = []

k_values = range(2, 8)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)


# ==========================================
# 4. PRINT SILHOUETTE SCORES
# ==========================================

print("\nSilhouette Scores:")

for k, score in zip(k_values, silhouette_scores):

    print(
        f"K = {k}  "
        f"Silhouette Score = {score:.4f}"
    )


# ==========================================
# 5. ELBOW GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")

plt.tight_layout()

plt.savefig(
    "results/kmeans_elbow.png",
    dpi=300
)

plt.show()


# ==========================================
# 6. SELECT K
# ==========================================

best_k = list(k_values)[
    silhouette_scores.index(
        max(silhouette_scores)
    )
]

print("\nSelected K:", best_k)


# ==========================================
# 7. FINAL K-MEANS
# ==========================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters


# ==========================================
# 8. CLUSTER DISTRIBUTION
# ==========================================

print("\nCluster Distribution:")

print(
    df["Cluster"].value_counts().sort_index()
)


# ==========================================
# 9. PCA
# ==========================================

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance:")

print(
    pca.explained_variance_ratio_
)

print(
    "Total explained variance:",
    pca.explained_variance_ratio_.sum()
)


# ==========================================
# 10. PCA DATAFRAME
# ==========================================

pca_df = pd.DataFrame({

    "PC1": X_pca[:, 0],

    "PC2": X_pca[:, 1],

    "Cluster": clusters

})


# ==========================================
# 11. PCA CLUSTER VISUALIZATION
# ==========================================

plt.figure(figsize=(9, 6))

for cluster in sorted(
    pca_df["Cluster"].unique()
):

    cluster_data = pca_df[
        pca_df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        label=f"Cluster {cluster}"
    )


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "K-Means Clusters Visualized Using PCA"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/kmeans_pca.png",
    dpi=300
)

plt.show()


# ==========================================
# 12. SAVE CLUSTERED DATA
# ==========================================

df.to_csv(
    "results/clustered_forest_fire_data.csv",
    index=False
)

print("\nFiles saved:")

print("results/kmeans_elbow.png")
print("results/kmeans_pca.png")
print("results/clustered_forest_fire_data.csv")

print("\n" + "=" * 60)
print("CO4 COMPLETE")
print("=" * 60)