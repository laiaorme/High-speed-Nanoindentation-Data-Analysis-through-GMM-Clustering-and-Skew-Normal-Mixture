This code file applies Gaussian Mixture Model (GMM) clustering to hardness and modulus data to identify different phases in the material. It explores various numbers of clusters and uses statistical criteria like BIC, AIC, and a modified BIC to select the optimal number of clusters.

- **Key Functions**:
  - **GMM Fitting**: Fits GMM models with varying numbers of components (clusters), ranging from 1 to 20, using the `sklearn.mixture` module.
  - **Model Selection**: Calculates Bayesian Information Criterion (BIC), Akaike Information Criterion (AIC), and a modified BIC to evaluate and compare models, helping to select the optimal number of clusters.
  - **Cluster Analysis**: For the optimal model, calculates the mean, standard deviation, and percentage of data in each cluster for both hardness and modulus.
  - **Plotting**:
    - **Modulus vs. Hardness Plot**: Visualizes the GMM clustering results by plotting the identified clusters in a modulus vs. hardness scatter plot, with different colors representing different clusters.
    - **Spatial Plot**: Generates spatial maps of the sample, showing the distribution of the identified clusters across the X and Y coordinates.


