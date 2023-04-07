import numpy as np

class NaiveBayesClassifier:
    """
    Naive Bayes classifier using Gaussian likelihood.

    Parameters:
    -----------
    alpha : float
        Smoothing parameter for the likelihoods. Default is 1.

    Attributes:
    -----------
    classes : array-like, shape = [n_classes,]
        Unique class labels.
    mean : array-like, shape = [n_classes, n_features]
        Mean of each feature for each class.
    var : array-like, shape = [n_classes, n_features]
        Variance of each feature for each class.
    prior : array-like, shape = [n_classes,]
        Prior probability of each class.
    """

    def __init__(self, alpha=1):
        self.alpha = alpha
    
    def fit(self, X, y):
        """
        Fit Naive Bayes to training data.

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
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.prior = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.var[i, :] = X_c.var(axis=0)
            self.prior[i] = X_c.shape[0] / float(n_samples)
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
        likelihood = np.zeros((X.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            class_likelihood = self._gaussian_likelihood(X, self.mean[i, :], self.var[i, :])
            prior = np.log(self.prior[i])
            likelihood[:, i] = np.sum(np.log(class_likelihood), axis=1) + prior
        return self.classes[np.argmax(likelihood, axis=1)]
    
    def _gaussian_likelihood(self, X, mean, var):
        """
        Compute Gaussian likelihood for each feature.

        Parameters:
        -----------
        X : array-like, shape = [n_samples, n_features]
            Samples to compute likelihood for.
        mean : array-like, shape = [n_features,]
            Mean of each feature.
        var : array-like, shape = [n_features,]
            Variance of each feature.

        Returns:
        --------
        likelihood : array-like, shape = [n_samples, n_features]
            Gaussian likelihood for each sample and feature.
        """
        eps = 1e-4
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = -0.5 * (np.square(X - mean) / (var + eps))
        return coeff * np.exp(exponent)
