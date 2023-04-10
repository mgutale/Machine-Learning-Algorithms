import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import euclidean_distances


class HierarchicalClustering(BaseEstimator, ClusterMixin):
    """
    Hierarchical agglomerative clustering algorithm.

    Parameters:
    -----------
    n_clusters : int
        The number of clusters to form.
    linkage : str
        The linkage criterion to use. Can be one of {'single', 'complete', 'average'}.

    Attributes:
    -----------
    clusters_ : np.ndarray, shape (n_samples,)
        Cluster labels for each sample.
    """

    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.clusters_ = None

    def fit(self, X):
        """
        Fit the hierarchical clustering model.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.
        """
        n_samples = X.shape[0]
        self.clusters_ = np.arange(n_samples)

        # Define linkage criterion function
        if self.linkage == 'single':
            linkage_criterion = self._single_linkage
        elif self.linkage == 'complete':
            linkage_criterion = self._complete_linkage
        elif self.linkage == 'average':
            linkage_criterion = self._average_linkage
        else:
            raise ValueError("Invalid linkage criterion. "
                             f"Expected one of ['single', 'complete', 'average'], but got {self.linkage}.")

        # Main agglomerative clustering loop
        for _ in range(n_samples - self.n_clusters):
            min_distance = float('inf')
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if self.clusters_[i] == self.clusters_[j]:
                        continue
                    distance = linkage_criterion(X, self.clusters_, i, j)
                    if distance < min_distance:
                        min_distance = distance
                        merge_i = i
                        merge_j = j
            self._merge_clusters(merge_i, merge_j)

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
        n_samples = X.shape[0]
        clusters = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            clusters[i] = self._find_cluster(i)
        return clusters

    def _merge_clusters(self, i, j):
        """
        Merge the clusters containing samples i and j.

        Parameters:
        -----------
        i : int
            Index of first sample.
        j : int
            Index of second sample.
        """
        cluster_i = self._find_cluster(i)
        cluster_j = self._find_cluster(j)
        self.clusters_[self.clusters_ == cluster_j] = cluster_i

    def _find_cluster(self, i):
        """
        Find the cluster containing sample i.

        Parameters:
        -----------
        i : int
            Index of sample.

        Returns:
        --------
        int
            The cluster label.
        """
        while self.clusters_[i] != i:
            i = self.clusters_[i]
        return i

    def _single_linkage(self, X, clusters, i, j):
        """
        Calculate the single linkage distance between two clusters.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.
        clusters : np.ndarray, shape (n_samples,)
            The current cluster labels for each sample.
        i : int
            Index of first cluster.
        j : int
            Index of second cluster.

        Returns:
        --------
        float
            The distance between the two clusters.
        """
        return np.min([self.distances_[i, k] for k in clusters[self.clusters_[j]]])

    def _complete_linkage(self, X, clusters, i, j):
        """
        Calculate the complete linkage distance between two clusters.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.
        clusters : np.ndarray, shape (n_samples,)
            The current cluster labels for each sample.
        i : int
            Index of first cluster.
        j : int
            Index of second cluster.

        Returns:
        --------
        float
            The distance between the two clusters.
        """
        return np.max([self.distances_[i, k] for k in clusters[self.clusters_[j]]])

    def _average_linkage(self, X, clusters, i, j):
        """
        Calculate the average linkage distance between two clusters.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            The input data.
        clusters : np.ndarray, shape (n_samples,)
            The current cluster labels for each sample.
        i : int
            Index of first cluster.
        j : int
            Index of second cluster.

        Returns:
        --------
        float
            The distance between the two clusters.
        """
        return np.mean([self.distances_[i, k] for k in clusters[self.clusters_[j]]])

