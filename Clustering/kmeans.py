import numpy as np

class KMeans:
    """
    KMeans clustering algorithm for unsupervised machine learning.
    
    Parameters:
    -----------
    k: int
        The number of clusters to form.
    
    max_iters: int
        Maximum number of iterations for the algorithm to converge.
        
    random_state: int
        Seed value for random initialization of centroids.
    
    Attributes:
    -----------
    k: int
        The number of clusters.
    
    max_iters: int
        Maximum number of iterations.
    
    clusters: list
        A list of arrays where each array contains the data points belonging to a cluster.
        
    centroids: numpy array
        A 2D numpy array of shape (k, n_features) where each row represents a centroid of a cluster.
    """
    
    def __init__(self, k=8, max_iters=300, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.clusters = []
        self.centroids = None
        self.random_state = random_state
    
    def fit(self, X):
        """
        Fit the KMeans clustering algorithm on the given dataset.
        
        Parameters:
        -----------
        X: numpy array
            A 2D numpy array of shape (n_samples, n_features) where each row represents a data point.
        """
        # Randomly initialize the centroids
        rng = np.random.RandomState(self.random_state)
        self.centroids = X[rng.choice(X.shape[0], self.k, replace=False), :]
        
        # Run the main loop for the given number of iterations
        for _ in range(self.max_iters):
            # Assign each data point to the closest centroid
            self.clusters = [[] for _ in range(self.k)]
            for i, x in enumerate(X):
                distances = np.linalg.norm(x - self.centroids, axis=1)
                closest_centroid_idx = np.argmin(distances)
                self.clusters[closest_centroid_idx].append(i)
            
            # Update the centroids to the mean of the assigned data points
            new_centroids = np.zeros_like(self.centroids)
            for i, cluster in enumerate(self.clusters):
                if len(cluster) > 0:
                    new_centroids[i] = np.mean(X[cluster], axis=0)
                else:
                    new_centroids[i] = self.centroids[i]
            
            # If the centroids don't change anymore, exit the loop
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
    
    def predict(self, X):
        """
        Predict the cluster labels for the given data points.
        
        Parameters:
        -----------
        X: numpy array
            A 2D numpy array of shape (n_samples, n_features) where each row represents a data point.
            
        Returns:
        --------
        predictions: numpy array
            A 1D numpy array of length n_samples containing the cluster label assigned to each data point.
        """
        distances = np.linalg.norm(X - self.centroids[:, np.newaxis], axis=2)
        predictions = np.argmin(distances, axis=0)
        return predictions
