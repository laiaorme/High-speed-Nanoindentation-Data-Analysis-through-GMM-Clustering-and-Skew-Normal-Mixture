import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

class Skew_GMM:
    def __init__(self, n_components, max_iter=100, tol=1e-6, init_method='kmeans++', reg_covar=1e-6, comp_names=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.reg_covar = reg_covar
        if comp_names is None:
            self.comp_names = [f"comp{index}" for index in range(self.n_components)]
        else:
            self.comp_names = comp_names
        self.pi = [1/self.n_components for _ in range(self.n_components)]

    def multivariate_skew_normal_pdf(self, X, mean_vector, covariance_matrix, skew_vector):
        X = np.asarray(X)
        mean_vector = np.asarray(mean_vector)
        covariance_matrix = np.asarray(covariance_matrix)

        d = X.shape[0]  # Dimensionality of X
        X_minus_mu = X - mean_vector
        inv_covmat = np.linalg.inv(covariance_matrix)
        exponent = -0.5 * np.dot(X_minus_mu.T, np.dot(inv_covmat, X_minus_mu))

        norm_pdf = np.exp(exponent) / np.sqrt((2 * np.pi)**d * np.linalg.det(covariance_matrix))
        skew_cdf = norm.cdf(np.dot(skew_vector, np.dot(inv_covmat, X_minus_mu)))

        density = 2 * norm_pdf * skew_cdf
        return density


    def kmeans_plusplus_initialization(self, X):
        # Asegurarse de que X es un numpy array de tipo float
        try:
            X = np.array(X, dtype=float)
        except Exception as e:
            return

        kmeans = KMeans(n_clusters=self.n_components).fit(X)
        centers = [kmeans.cluster_centers_[0]]

        for _ in range(1, self.n_components):
            dist_sq = np.array([min([np.inner(c - x, c - x) for c in centers]) for x in X])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break
            centers.append(X[i])

        self.mean_vector = np.array(centers)

        # Revisar elementos individuales de X
        try:
            for i, row in enumerate(X):
                for j, val in enumerate(row):
                    if not np.isfinite(val):
                        print(f"Non-finite value found at X[{i},{j}]: {val}")
        except Exception as e:
            print("Error checking elements of X:", e)

        # Verificar si np.cov da un error con una matriz de prueba
        try:
            test_cov = np.cov(np.array([[1.0, 2.0], [3.0, 4.0]]), rowvar=False)
            #print('Test np.cov() with simple data:', test_cov)
        except Exception as e:
            print("Error in np.cov with test data:", e)

        try:
            print('np.cov() result with X:', np.cov(X, rowvar=False))
        except Exception as e:
            print("Error in np.cov with X:", e)

        try:
            covariance_matrixes = [np.cov(X, rowvar=False) for _ in range(self.n_components)]
            self.covariance_matrixes = [covariance_matrix + self.reg_covar * np.eye(X.shape[1]) for covariance_matrix in covariance_matrixes]
            self.skew_vectors = [np.zeros(X.shape[1]) for _ in range(self.n_components)]
        except Exception as e:
            print("Error calculating covariance matrices:", e)


    def initialize_parameters(self, X):
        if self.init_method == 'random':
            new_X = [np.array(x, dtype=np.float64) for x in np.array_split(X, self.n_components)]
            self.mean_vector = np.zeros((self.n_components, X.shape[1]), dtype=np.float64)
            self.covariance_matrixes = [np.cov(x.T) + self.reg_covar * np.identity(X.shape[1]) for x in new_X]
            self.skew_vectors = [np.zeros(X.shape[1]) for _ in range(self.n_components)]
            del new_X
        elif self.init_method == 'kmeans++':
            self.kmeans_plusplus_initialization(X)
            pass
        else:
            raise ValueError("Initialization method not supported")

    def fit(self, X):
        self.initialize_parameters(X)
        self.r = np.zeros((len(X), self.n_components), dtype=np.float64)
        X = X.astype(np.float64)
        print("Shape of X:", X.shape)  # Add this line

        for iteration in range(self.max_iter):
            print(f"Iteration {iteration + 1}/{self.max_iter}")  # Add this line
            ''' --------------------------   E - STEP   -------------------------- '''
            for n in range(len(X)):
                for k in range(self.n_components):
                    self.r[n][k] = self.pi[k] * self.multivariate_skew_normal_pdf(X[n], self.mean_vector[k], self.covariance_matrixes[k], self.skew_vectors[k])

                denominator = sum(self.pi[j] * self.multivariate_skew_normal_pdf(X[n], self.mean_vector[j], self.covariance_matrixes[j], self.skew_vectors[j]) for j in range(self.n_components))
                if denominator < 1e-10:
                    self.r[n] = 0.0
                else:
                    self.r[n] /= denominator


            # Update the mean vector
            N = np.sum(self.r, axis=0)

            ''' --------------------------   M - STEP   -------------------------- '''
            for k in range(self.n_components):
                self.mean_vector[k] = np.sum(self.r[:, k][:, np.newaxis] * X, axis=0) / N[k]

            new_covariance_matrixes = []
            for k in range(self.n_components):
                diff = X - self.mean_vector[k]
                weighted_diff = self.r[:, k][:, np.newaxis] * diff
                new_covariance_matrix = np.dot(weighted_diff.T, diff) / N[k] + self.reg_covar * np.identity(X.shape[1])
                new_covariance_matrixes.append(new_covariance_matrix)

            new_covariance_matrixes = np.array(new_covariance_matrixes)
            new_pi = N / len(X)

            # Check for convergence
            if np.allclose(self.mean_vector, self.mean_vector, atol=self.tol) and \
               np.allclose(self.covariance_matrixes, new_covariance_matrixes, atol=self.tol) and \
               np.allclose(self.pi, new_pi, atol=self.tol):
                break

            self.covariance_matrixes = new_covariance_matrixes
            self.pi = new_pi
        print("Fit function completed")  # Add this line

    def predict(self, X):
        probas = np.zeros((len(X), self.n_components))
        for n in range(len(X)):
            for k in range(self.n_components):
                probas[n, k] = self.pi[k] * self.multivariate_skew_normal_pdf(X[n], self.mean_vector[k], self.covariance_matrixes[k], self.skew_vectors[k])
        cluster = np.argmax(probas, axis=1)
        return [self.comp_names[c] for c in cluster]
