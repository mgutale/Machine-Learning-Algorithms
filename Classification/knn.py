import numpy as np
from collections import Counter

class KNNClassifier:
    """
    K-Nearest Neighbors classifier.

    Parameters:
    -----------
    k : int
        Number of neighbors to consider. Default is 3.

    Attributes:
    -----------
    X_train : array-like, shape = [n_samples, n_features]
        Training samples.
    y_train : array-like, shape = [n_samples,]
        Target values (class labels).
    """

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        """
        Fit KNN to training data.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Training samples.
        y : array-like, shape = [n_samples,]
            Target values (class labels).

        Returns:
        --------
        self : object
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Test samples.

        Returns:
        --------
        y_pred : array-like, shape = [n_samples,]
            Predicted class labels.
        """
        predictions = []
        for x in X:
            distances = []
            for x_train in self.X_train:
                distance = np.sqrt(np.sum((x - x_train)**2))
                distances.append(distance)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        return np.array(predictions)
