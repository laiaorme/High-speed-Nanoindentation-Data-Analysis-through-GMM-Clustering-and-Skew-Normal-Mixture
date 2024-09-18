import numpy as np
import pandas as pd
from Class_2D_skewnormal_mix import Skew_GMM
from Import_data import DFtoMatrix


filtered_df = pd.read_excel('filtered_data.xlsx', sheet_name='Sheet1')
X = DFtoMatrix(filtered_df)

# Adjust GMM from scratch
gmm_model = Skew_GMM(n_components=4, max_iter=100, init_method='kmeans++')
gmm_model.fit(X)

# Predict clusters
predictions = gmm_model.predict(X)

# Convertir las predicciones en una columna del DataFrame
filtered_df['CLUSTER'] = predictions
filtered_df.to_excel('clustering_data.xlsx', index=False)

# Statistics lists
cluster_means = []
cluster_covariances = []
cluster_skewness = []
cluster_percentages = []

total_data = len(X)

for k in range(gmm_model.n_components):
    cluster_k = X[filtered_df['CLUSTER'] == gmm_model.comp_names[k]]
    cluster_means.append(np.mean(cluster_k, axis=0))
    cluster_covariances.append(np.cov(cluster_k, rowvar=False))
    cluster_skewness.append(gmm_model.skew_vectors[k])
    cluster_percentages.append(len(cluster_k) / total_data * 100)

# Dataframe info clusters
df_clusters = pd.DataFrame({
    'Cluster index': range(gmm_model.n_components),
    'Mean': [list(mean) for mean in cluster_means],
    'Covariance Matrix': [cov.tolist() for cov in cluster_covariances],
    'Skewness Vector': [list(skew) for skew in cluster_skewness],
    'Percentage': cluster_percentages
})
df_clusters.to_excel('cluster_statistics.xlsx', index=False)

# Print statistics
for k in range(gmm_model.n_components):
    print(f"Cluster index {k}:")
    print(f"    Mean: {cluster_means[k]}")
    print(f"    Covariance Matrix:")
    print(cluster_covariances[k])
    print(f"    Skewness Vector: {cluster_skewness[k]}")
    print(f"    Percentage: {cluster_percentages[k]:.2f}%")
    print()



