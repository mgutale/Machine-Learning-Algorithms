import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    """
    Gaussian mixture model clustering algorithm.

    Parameters:
    -----------
    n_components : int
        The number of Gaussian components.
    max_iter : int
        The maximum number of iterations.
    tol : float
        The convergence tolerance.

    Attributes:
    -----------
    weights_ : np.ndarray, shape (n_components,)
        The weights of each Gaussian component.
    means_ : np.ndarray, shape (n_components, n_features)
        The means of each Gaussian component.
    covariances_ : np.ndarray, shape (n_components, n_features, n_features)
        The covariance matrices of each Gaussian component.
    clusters_ : np.ndarray, shape (n_samples,)
        Cluster labels for each sample.
    """

    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.clusters_ = None

    def fit(self, X):
        """
        Fit the GMM clustering model.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.
        """
        n_samples, n_features = X.shape

        # Initialize model parameters
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = X[np.random.choice(n_samples, size=self.n_components, replace=False)]
        self.covariances_ = np.array([np.eye(n_features)] * self.n_components)

        # EM algorithm
        for _ in range(self.max_iter):
            # E-step
            probabilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                probabilities[:, k] = self.weights_[k] * multivariate_normal.pdf(X, self.means_[k], self.covariances_[k])
            self.clusters_ = np.argmax(probabilities, axis=1)

            # M-step
            for k in range(self.n_components):
                Nk = np.sum(self.clusters_ == k)
                self.weights_[k] = Nk / n_samples
                self.means_[k] = np.sum(X[self.clusters_ == k], axis=0) / Nk
                self.covariances_[k] = np.dot((X[self.clusters_ == k] - self.means_[k]).T,
                                               (X[self.clusters_ == k] - self.means_[k])) / Nk

            # Check for convergence
            log_likelihood = np.sum(np.log(np.sum(probabilities, axis=1)))
            if _ > 0 and np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

    def predict(self, X):
        """
        Assign cluster labels to input data.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        np.ndarray, shape (n_samples,)
            Cluster labels for each sample.
        """
        if self.clusters_ is None:
            raise NotFittedError("Model has not been fitted yet.")
        probabilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            probabilities[:, k] = self.weights_[k] * multivariate_normal.pdf(X, self.means_[k], self.covariances_[k])
        return np.argmax(probabilities, axis=1)

