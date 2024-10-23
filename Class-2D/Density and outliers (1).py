from Import_data import DataToDF
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

excel = 'WCNicoarse-4D-80nm-150x150_filtered.xlsx'

df = DataToDF(excel)

# Plot all the data
plt.figure(figsize=(10, 10))
plt.scatter(df['HARDNESS'], df['MODULUS'], color='blue', alpha=0.05)
plt.show()

# Calculate densities
values = df[['HARDNESS', 'MODULUS']].values.T
kde = gaussian_kde(values)
density = kde(values)

# Stablish density threshold
threshold = np.percentile(density, 20)

# Filtrar puntos por densidad
filtered_df = df[density > threshold]
filtered_df.to_excel('filtered_data.xlsx', index=False)

# Nuevo scatterplot con los puntos filtrados
plt.figure(figsize=(10, 10))
plt.scatter(filtered_df['HARDNESS'], filtered_df['MODULUS'], alpha=0.1, color='blue')
plt.xlabel('HARDNESS')
plt.ylabel('MODULUS')
plt.title('Scatterplot of Hardness vs Modulus (Filtered)')
plt.show()