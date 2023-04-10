import numpy as np
from scipy.spatial.distance import cdist

class DBSCAN:
    """
    DBSCAN clustering algorithm.

    Parameters:
    -----------
    eps : float
        The maximum distance between two samples for them to be considered as part of the same cluster.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point.

    Attributes:
    -----------
    core_samples_ : np.ndarray, shape (n_core_samples,)
        Indices of core samples.
    clusters_ : np.ndarray, shape (n_samples,)
        Cluster labels for each sample.
    noise_ : np.ndarray, shape (n_samples,)
        Boolean array indicating noise points.
    """

    def __init__(self, eps=1.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.core_samples_ = None
        self.clusters_ = None
        self.noise_ = None

    def fit(self, X):
        """
        Fit the DBSCAN clustering model.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.
        """
        n_samples = X.shape[0]
        self.clusters_ = -np.ones(n_samples, dtype=int)

        # Find core samples
        D = cdist(X, X)
        is_core = np.sum(D <= self.eps, axis=1) >= self.min_samples
        self.core_samples_ = np.where(is_core)[0]

        # Assign cluster labels
        cluster = 0
        for i in self.core_samples_:
            if self.clusters_[i] == -1:
                self._expand_cluster(X, i, cluster, is_core, D)
                cluster += 1

        # Assign noise points
        self.noise_ = self.clusters_ == -1

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
        D = cdist(X, X[self.core_samples_])
        is_member = np.any(D <= self.eps, axis=1)
        clusters = -np.ones(X.shape[0], dtype=int)
        clusters[is_member] = self.clusters_[np.argmax(D[is_member], axis=1)]
        return clusters

    def _expand_cluster(self, X, i, cluster, is_core, D):
        """
        Expand the cluster starting from the given core sample.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.
        i : int
            Index of the core sample to start from.
        cluster : int
            The cluster label to assign to the new points.
        is_core : np.ndarray, shape (n_samples,)
            Boolean array indicating core points.
        D : np.ndarray, shape (n_samples, n_samples)
            Pairwise distance matrix.
        """
        seeds = set([i])
        while seeds:
            j = seeds.pop()
            if self.clusters_[j] == -1:
                self.clusters_[j] = cluster
                if is_core[j]:
                    seeds |= set(np.where(D[j] <= self.eps)[0])

