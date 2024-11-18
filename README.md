# High speed Nanoindentation Data Analysis through GMM and Skew Normal Mixture
This is the code used in the scientific paper "High-speed Nanoindentation Data Analysis of WC-base cemented carbides through Gaussian Mixture Model Clustering and Skew-Normal Mixture: Beyond Gaussian Deconvolution": https://doi.org/10.1016/j.ijrmhm.2024.106917

# Code Folders and Licensing Overview

## Folders:
### 1. `GMM`
- Contains code to apply the **Gaussian Mixture Model (GMM)** from sklearn library for clustering nanoindentation data. It determines the optimal number of clusters using BIC, AIC, and modified BIC criteria. Includes:
  - **Modulus vs. Hardness plots**: Displaying the clusters in material properties.
  - **Spatial plots**: Showing the distribution of clusters across the sample surface.

### 2. `Skewnormal`
- Contains code for applying a **skew-normal mixture model** to nanoindentation data. It fits the data using multiple skew-normal distributions, selects the optimal number of distributions, and generates:
  - **1D plots**: Fitting hardness and modulus values.
  - **Spatial contour plots**: Visualizing the distribution of material properties over the surface.

### 3. `Additional Codes`
- **Calculate binder content from SEM images**: Converts SEM images to black and white to identify and calculate the percentage of the binder phase.
- **Density and Outliers**: Analyzes hardness and modulus data, filters out low-density data points, and generates scatter plots before and after filtering.
- **Contour Plot**: Generates 2D spatial contour plots for hardness and modulus data, interpolating values over X and Y positions to visualize the surface distribution.

### 4. `2D-class`
- Contains codes for implementing the **Gaussian Mixture Model (GMM) from scratch**.

## Licensing Details

This code is provided for educational and research purposes. Redistribution and modification are allowed under the following conditions:
1. **Non-Commercial Use:** The code cannot be used for commercial purposes without explicit permission.
  
2. **Proper Citation:** If this code is used in academic publications, the following article must be cited appropriately:
  - L. Ortiz-Membrado, R. Benítez, L. Llanes, E. Jiménez-Piqué, High-speed Nanoindentation Data Analysis of WC-based Cemented Carbides using Gaussian Mixture Model Clustering and Skew-normal Mixture: Beyond Gaussian Deconvolution, International Journal of Refractory Metals and Hard Materials, Volume 126, 2025.

# Information About the Data Format

All files are Google Colab Jupyter Notebook (.ipynb)

This repository contains code for analyzing hardmetals data obtained using the iMicro KLA nanoindenter with the Nanoblitz4D method. The data columns and their units are structured as follows:

| Markers       | INDENT | X        | Y        | DEPTH      | LOAD       | STIFFNESS   | HARDNESS    | MODULUS     |
|---------------|--------|----------|----------|------------|------------|-------------|-------------|-------------|
|               |        | (µm)     | (µm)     | (nm)       | (mN)       | (N/m)       | (GPa)       | (GPa)       |
| Surface Index | 0      | 0        | -2.01044 | 0.000314   | -          | -           | -           | -           |
| 1             | 0      | 0        | 97.26101 | 6.82422    | 186834.7   | 31.07447    | 464.907     |


If you are using data with a different format or material, make sure to update the parameters in the data import functions and modify the plotting scripts accordingly.

