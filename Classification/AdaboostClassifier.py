import numpy as np

class AdaBoostClassifier:
    """
    AdaBoost (Adaptive Boosting) classifier.

    Parameters:
    -----------
    base_classifier : object
        The base classifier to use for boosting.
    n_estimators : int
        The number of base classifiers to use in the ensemble.
    
    Attributes:
    -----------
    base_classifier : object
        The base classifier to use for boosting.
    n_estimators : int
        The number of base classifiers to use in the ensemble.
    classifiers : list
        The list of base classifiers in the ensemble.
    alphas : list
        The list of weights for the base classifiers in the ensemble.

    Methods:
    --------
    fit(X, y)
        Fit the AdaBoost classifier to the training data.
    predict(X)
        Predict the class labels for the given test data.

    Notes:
    ------
    This implementation uses the exponential loss function for binary classification.

    References:
    -----------
    [1] Schapire, R. E. (1999). Theoretical views of boosting and applications.
        In Computational learning theory (pp. 13-50). Springer, Berlin, Heidelberg.
    [2] Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization
        of on-line learning and an application to boosting. In European conference
        on computational learning theory (pp. 23-37). Springer, Berlin, Heidelberg.

    """
    def __init__(self, base_classifier, n_estimators=10):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.classifiers = []
        self.alphas = []
    
    def fit(self, X, y):
        """
        Fit the AdaBoost classifier to the training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input training data.
        y : array-like, shape (n_samples,)
            The target values.

        Returns:
        --------
        self : AdaBoostClassifier
            The fitted AdaBoost classifier.

        """
        n_samples = X.shape[0]
        weights = np.full(n_samples, (1 / n_samples))
        for i in range(self.n_estimators):
            classifier = self.base_classifier()
            classifier.fit(X, y, sample_weight=weights)
            y_pred = classifier.predict(X)
            err = np.sum(weights * (y_pred != y))
            alpha = 0.5 * np.log((1 - err) / err)
            self.classifiers.append(classifier)
            self.alphas.append(alpha)
            weights *= np.exp(-alpha * y * y_pred)
            weights /= np.sum(weights)
        return self
    
    def predict(self, X):
        """
        Predict the class labels for the given test data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input test data.

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            The predicted class labels.

        """
        predictions = np.array([classifier.predict(X) for classifier in self.classifiers])
        weighted_predictions = np.apply_along_axis(lambda x: np.sum(x * self.alphas), axis=0, arr=predictions)
        return np.sign(weighted_predictions)
