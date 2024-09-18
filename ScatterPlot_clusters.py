import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Función para calcular la densidad de la distribución normal sesgada
def skew_normal_pdf(X, mean, cov, skew):
    inv_cov = np.linalg.inv(cov)
    norm_factor = np.sqrt((2 * np.pi)**len(mean) * np.linalg.det(cov))

    def multivariate_skewnorm_pdf(x):
        x_minus_mu = x - mean
        exp_part = np.exp(-0.5 * np.dot(x_minus_mu.T, np.dot(inv_cov, x_minus_mu)))
        skew_cdf = norm.cdf(np.dot(skew, np.dot(inv_cov, x_minus_mu)))
        return 2 * exp_part * skew_cdf / norm_factor

    return np.array([multivariate_skewnorm_pdf(xi) for xi in X])

# Cargar los DataFrames
df_clusters = pd.read_excel('cluster_statistics.xlsx')
filtered_df = pd.read_excel('clustering_data.xlsx')

# Convertir las columnas de medias, covarianza y skewness de listas a arrays numpy
df_clusters['Mean'] = df_clusters['Mean'].apply(lambda x: np.array(eval(x)))
df_clusters['Covariance Matrix'] = df_clusters['Covariance Matrix'].apply(lambda x: np.array(eval(x)))
df_clusters['Skewness Vector'] = df_clusters['Skewness Vector'].apply(lambda x: np.array(eval(x)))


# Crear un grid de puntos para calcular la densidad
x = np.linspace(filtered_df['HARDNESS'].min(), filtered_df['HARDNESS'].max(), 100)
y = np.linspace(filtered_df['MODULUS'].min(), filtered_df['MODULUS'].max(), 100)
X_grid, Y_grid = np.meshgrid(x, y)
grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]

# Mapa de colores para los clusters
cluster_colors = {'comp0': 'blue', 'comp1': 'red', 'comp2': 'yellow', 'comp3': 'magenta'}
# Graficar scatter plot
plt.figure(figsize=(10, 8))
for cluster, color in cluster_colors.items():
    cluster_data = filtered_df[filtered_df['CLUSTER'] == cluster]
    plt.scatter(cluster_data['HARDNESS'], cluster_data['MODULUS'], color=color, label=cluster,alpha=0.1)

# Calcular y dibujar contornos para cada clúster
for cluster in df_clusters['Cluster index']:
    mean = df_clusters.loc[df_clusters['Cluster index'] == cluster, 'Mean'].values[0]
    cov = df_clusters.loc[df_clusters['Cluster index'] == cluster, 'Covariance Matrix'].values[0]
    skew = df_clusters.loc[df_clusters['Cluster index'] == cluster, 'Skewness Vector'].values[0]

    Z = skew_normal_pdf(grid_points, mean, cov, skew).reshape(X_grid.shape)

    # Dibujar contornos
    plt.contour(X_grid, Y_grid, Z, levels=5, linewidths=1, colors='black')

plt.xlim(filtered_df['HARDNESS'].min(), filtered_df['HARDNESS'].max())
plt.ylim(filtered_df['MODULUS'].min(), filtered_df['MODULUS'].max())

plt.title('Scatter plot of Hardness vs Modulus with Skewed Gaussian Mixture Model Contours')
plt.xlabel('Hardness')
plt.ylabel('Modulus')
plt.legend()
plt.show()






