import numpy as np

class AdaBoostRegressor:
    """
    AdaBoost for regression using decision trees as base estimators.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        Number of estimators in the ensemble.
    max_depth : int, optional (default=1)
        Maximum depth of each decision tree. If None, then the tree is grown until all leaves are pure.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor objects
        List of decision trees in the ensemble.
    weights_ : array-like, shape (n_estimators,)
        Weight of each estimator in the ensemble.

    Methods
    -------
    fit(X, y)
        Fit AdaBoost ensemble on training data (X, y).
    predict(X)
        Predict target values for input data (X).

    """

    def __init__(self, n_estimators=100, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators_ = []
        self.weights_ = np.zeros(n_estimators)

    def fit(self, X, y):
        """
        Fit AdaBoost ensemble on training data (X, y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.

        """
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            # Create decision tree and fit on training data with sample weights
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, y, sample_weights)
            self.estimators_.append(tree)

            # Compute estimator weight and update sample weights
            y_pred = tree.predict(X)
            error = np.mean(sample_weights * (y - y_pred) ** 2)
            self.weights_[i] = 0.5 * np.log((1 - error) / error)
            sample_weights *= np.exp(-self.weights_[i] * (y - y_pred) ** 2)

        return self

    def predict(self, X):
        """
        Predict target values for input data (X).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted target values.

        """
        y_pred = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            y_pred += self.weights_[i] * self.estimators_[i].predict(X)
        return y_pred
