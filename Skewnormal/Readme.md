# Skew-Normal Mixture Fitting for Hardmetals

## Overview

This repository contains code for fitting skew-normal mixture models to nanoindentation data in hardmetals. The code is designed to handle fitting with 2, 3, or 4 skew-normal distributions to represent different phases in the material. The fitting is applied separately to hardness and modulus data, and statistical criteria are used to select the optimal model. Spatial visualization of the results is also provided for X and Y coordinates.

## Code Structure

### 1. Main Functions:
- **`mixture_skew_normal_2`, `mixture_skew_normal_3`, `mixture_skew_normal_4`**: These functions define mixtures of 2, 3, and 4 skew-normal distributions, respectively. Each function models the probability density function (PDF) for different numbers of phases. The parameters that define the distributions are: weight, location, scale, and skewness.

- **`error_function`**: Calculates the error between the experimental data and the model using a mixture of skew-normal distributions. This function is used to optimize parameters.

### 2. Fitting the Model:
- The fitting is done using **`curve_fit`** from the `scipy` library. It adjusts the parameters of the skew-normal distributions to minimize the difference between the fitted model and the hardness or modulus data.
- The code applies fitting for 2, 3, and 4 distributions, separately for both hardness and modulus values.

### 3. Model Selection:
- The **Bayesian Information Criterion (BIC)** and **Akaike Information Criterion (AIC)** are calculated for each number of distributions (2, 3, and 4). These criteria help in selecting the optimal number of phases for the data by balancing fit quality and model complexity. 

### 4. Spatial visualization:
- After fitting the 1D data (hardness and modulus), the code interpolates the values over a grid of X and Y positions using **`griddata`** to visualize spatial variations.
- Contour plots are generated to show the hardness and modulus distribution across the material surface.

## How to Use the Code

1. **Prepare Data**: Load your hardness and modulus data, along with X and Y positions for each indent.
2. **Run Fitting**: The code will fit the data using 2, 3, and 4 skew-normal distributions, optimizing the model parameters.
3. **Generate Plots**: The code will generate 1D and 2D plots to visualize the fitted distributions and spatial variations in hardness and modulus.
4. **Interpret Results**: Use the statistical criteria (BIC and AIC) and visualizations to interpret the number of phases in your material.
