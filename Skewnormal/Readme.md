# README for Skew-Normal Mixture Fitting Code
Overview
This repository contains code for fitting skew-normal mixture models, as described in the corresponding research article. The code is designed to adjust using both 2-cluster and 3-cluster configurations. These clusters represent the Binder phase and carbide phases: in the 2-cluster model, we group binder and carbide (1 phase), and in the 3-cluster model, we distinguish binder, carbide phase 1, and carbide phase 2.

Binder typically has a hardness of 5-10 GPa.
Carbides generally show hardness between 25-30 GPa.
This code can also be extended to work with composite materials with similar properties.

Code Description
Main Functions: The code includes functions for fitting skew-normal mixture models to hardness data. These functions optimize the model parameters (such as means, variances, and skewness) to best fit the observed data.

Values Optimized:

Mean and variance of hardness for each phase.
Skewness to capture the asymmetry in the distribution of hardness values.
Relationships: The code evaluates the distances between phases (e.g., using a distance metric 
𝑑
d) to better differentiate clusters and refine the mixture model.

Clusters: You can adjust for either:

2 clusters: Binder and carbide phase 1.
3 clusters: Binder, carbide phase 1, and carbide phase 2.
Files Included
Python Code File (skew_normal_fitting.py):

This file contains the Python code that can be copied and run in any Python environment for fitting skew-normal mixture models.
Google Colab Notebook (Skew_normal_for_hardmetals.ipynb):

This is a notebook designed to run in Google Colab, with all necessary code and explanations. It is ready to import into Google Colab for immediate use.
How to Import the Google Colab File
To import the Google Colab file and run it, follow these steps:

Go to Google Colab.
On the welcome page or menu, click File.
Select Upload notebook.
Choose the file Skew_normal_for_hardmetals.ipynb from your local drive and click Open.
The notebook will be loaded into Colab, and you can run the code directly from there.
Usage
Copy the Python code from the provided .py file to your preferred environment or directly use the Colab notebook for execution.
Adjust the parameters (number of clusters, hardness values, etc.) as needed based on your material's properties.
Make sure to refer to the corresponding article for more details on the methodology and applications of the skew-normal mixture model.

Let me know if you'd like to adjust or add any further details!
