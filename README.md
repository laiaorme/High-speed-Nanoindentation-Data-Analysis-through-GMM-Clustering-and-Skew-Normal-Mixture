# High-speed-Nanoindentation-Data-Analysis-through-GMM-Clustering-and-Skew-Normal-Mixture
This is the code used in the scientific paper "High-speed Nanoindentation Data Analysis of WC-base cemented carbides through Gaussian Mixture Model Clustering and Skew-Normal Mixture. Beyond Gaussian Deconvolution".

# Information About the Data Format

This repository contains code for analyzing hardmetals data obtained using the iMicro KLA nanoindenter with the Nanoblitz4D method. The data columns and their units are structured as follows:

| Markers       | INDENT | X        | Y        | DEPTH      | LOAD       | STIFFNESS   | HARDNESS    | MODULUS     |
|---------------|--------|----------|----------|------------|------------|-------------|-------------|-------------|
|               |        | (µm)     | (µm)     | (nm)       | (mN)       | (N/m)       | (GPa)       | (GPa)       |
| Surface Index | 0      | 0        | -2.01044 | 0.000314   | -          | -           | -           | -           |
| 1             | 0      | 0        | 97.26101 | 6.82422    | 186834.7   | 31.07447    | 464.907     |
| -             | 0      | 1.000125 | -3.76080 | 0.000585   | -          | -           | -           |
| 2             | 0      | 1.000125 | 94.95239 | 6.66585    | 178292.5   | 32.37595    | 455.2633    |
| -             | 0      | 2.00025  | -1.80635 | 0.000344   | -          | -           | -           |

If you are using data with a different format, make sure to update the parameters in the data import functions and modify the plotting scripts accordingly.

