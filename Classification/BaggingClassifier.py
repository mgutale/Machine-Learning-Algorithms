class BaggingClassifier:
    """
    A Bagging Classifier that fits base classifiers on random subsets of the training data and aggregates their predictions.

    Parameters:
    -----------
    base_classifier: class
        The base classifier to be bagged.
    n_estimators: int, default=10
        The number of base classifiers to fit.

    Attributes:
    -----------
    classifiers: list
        A list of the fitted base classifiers.

    Methods:
    --------
    fit(X, y):
        Fits the Bagging Classifier on the training data.
        
        Parameters:
        -----------
        X: numpy array, shape (n_samples, n_features)
            The training data.
        y: numpy array, shape (n_samples,)
            The target values.

    predict(X):
        Predicts the target values for the input data.

        Parameters:
        -----------
        X: numpy array, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        predictions: numpy array, shape (n_samples,)
            The predicted target values.
    """
    def __init__(self, base_classifier, n_estimators=10):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.classifiers = []
    
    def fit(self, X, y):
        """
        Fits the Bagging Classifier on the training data.

        Parameters:
        -----------
        X: numpy array, shape (n_samples, n_features)
            The training data.
        y: numpy array, shape (n_samples,)
            The target values.
        """
        for i in range(self.n_estimators):
            classifier = self.base_classifier()
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_subset = X[indices]
            y_subset = y[indices]
            classifier.fit(X_subset, y_subset)
            self.classifiers.append(classifier)
    
    def predict(self, X):
        """
        Predicts the target values for the input data.

        Parameters:
        -----------
        X: numpy array, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        predictions: numpy array, shape (n_samples,)
            The predicted target values.
        """
        predictions = np.array([classifier.predict(X) for classifier in self.classifiers])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
