import numpy as np
import pandas as pd
class GMM:
    '''
        This class is the implementation of the Gaussian Mixture Models 
        inspired by sci-kit learn implementation.
    '''
    def __init__(self, n_components, max_iter = 100, comp_names=None):
        '''
            This functions initializes the model by seting the following paramenters:
                :param n_components: int
                    The number of clusters in which the algorithm must split
                    the data set
                :param max_iter: int, default = 100
                    The number of iteration that the algorithm will go throw to find the clusters
                :param comp_names: list of strings, default=None
                    In case it is setted as a list of string it will use to
                    name the clusters
        '''
        self.n_components = n_components
        self.max_iter = max_iter
        if comp_names == None:
            self.comp_names = [f"comp{index}" for index in range(self.n_components)]
        else:
            self.comp_names = comp_names
        # pi list contains the fraction of the dataset for every cluster
        self.pi = [1/self.n_components for comp in range(self.n_components)]

    def multivariate_normal(self, X, mean_vector, covariance_matrix):
        '''
            This function implements the multivariat normal derivation formula,
            the normal distribution for vectors it requires the following parameters
                :param X: 1-d numpy array
                    The row-vector for which we want to calculate the distribution
                :param mean_vector: 1-d numpy array
                    The row-vector that contains the means for each column
                :param covariance_matrix: 2-d numpy array (matrix)
                    The 2-d matrix that contain the covariances for the features
        '''
        return (2*np.pi)**(-len(X)/2)*np.linalg.det(covariance_matrix)**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)

    def fit(self, X):
        '''
        The function for training the model
            :param X: 2-d numpy array
                The data must be passed to the algorithm as 2-d array, 
                where columns are the features and the rows are the samples
        '''
        # Splitting the data into n_components sub-sets
        new_X = [np.array(x, dtype=np.float64) for x in np.array_split(X, self.n_components)]
        # Initial computation of the mean-vector and covariance matrix
        self.mean_vector = np.zeros((self.n_components, X.shape[1]), dtype=np.float64)
        self.covariance_matrixes = [np.cov(x.T) for x in new_X]
        # Deleting the new_X matrix because we will not need it anymore
        del new_X
        
        # Initialize self.r
        self.r = np.zeros((len(X), self.n_components), dtype=np.float64)
        
        # Convert arrays to float64 data type
        self.mean_vector = self.mean_vector.astype(np.float64)
        X = X.astype(np.float64)
        
        for iteration in range(self.max_iter):
            ''' --------------------------   E - STEP   -------------------------- '''
            # Initiating the r matrix, every row contains the probabilities
            # for every cluster for this row
            self.r = np.zeros((len(X), self.n_components))
            # Calculating the r matrix
            for n in range(len(X)):
                for k in range(self.n_components):
                    self.r[n][k] = self.pi[k] * self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                    
                    # Avoid division by very small values or negative values
                    denominator = sum([self.pi[j]*self.multivariate_normal(X[n], self.mean_vector[j], self.covariance_matrixes[j]) for j in range(self.n_components)])
                    if denominator < 1e-10:  # Small value threshold
                        self.r[n][k] = 0.0
                    else:
                        self.r[n][k] /= denominator
            
            # Print debugging information
            # print("Before update - mean_vector[0]:", self.mean_vector[0])
            # print("r[0][0]:", self.r[0][0])
            # print("X[0]:", X[0])
            
            # Update the mean vector
            for k in range(self.n_components):
                for n in range(len(X)):
                    self.mean_vector[k] += self.r[n][k] * X[n]
            
            # Convert mean_vector to the correct data type
            self.mean_vector = self.mean_vector.astype(np.float64)
            
            # Calculate N
            N = np.sum(self.r, axis=0)

            ''' --------------------------   M - STEP   -------------------------- '''
            self.mean_vector /= N[:, np.newaxis] # Divide each row by the corresponding N
          
            # Initiating the list of the covariance matrices
            self.covariance_matrixes = [np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_components)]
            # Updating the covariance matrices
            for k in range(self.n_components):
                # Regularization term added to the covariance matrices
                self.covariance_matrixes[k] = np.cov(X.T, aweights=(self.r[:, k]), ddof=0) + 1e-6 * np.identity(len(X[0]))
            
            # Updating the pi list
            self.pi = [N[k]/len(X) for k in range(self.n_components)]
                       


    print("fit done")
    
    def predict(self, X):
        '''
            The predicting function
                :param X: 2-d array numpy array
                    The data on which we must predict the clusters
        '''
        probas = []
        for n in range(len(X)):
            probas.append([self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                           for k in range(self.n_components)])
        cluster = []
        for proba in probas:
            cluster.append(self.comp_names[proba.index(max(proba))])
        return cluster
