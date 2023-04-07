import numpy as np

class BaggingRegressor:
    """
    Bagging for regression using decision trees as base estimators.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        Number of trees in the ensemble.
    max_depth : int, optional (default=None)
        Maximum depth of each decision tree. If None, then the tree is grown until all leaves are pure.
    max_features : int or float, optional (default=1.0)
        Maximum number of features to consider when splitting a node. If float, then it represents the fraction of features to consider.
    bootstrap : bool, optional (default=True)
        Whether to use bootstrap samples when training each tree.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor objects
        List of decision trees in the ensemble.

    Methods
    -------
    fit(X, y)
        Fit bagging regressor on training data (X, y).
    predict(X)
        Predict target values for input data (X).

    """

    def __init__(self, n_estimators=100, max_depth=None, max_features=1.0, bootstrap=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.estimators_ = []

    def fit(self, X, y):
        """
        Fit bagging regressor on training data (X, y).

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
        n_samples, n_features = X.shape

        for i in range(self.n_estimators):
            # Create decision tree and fit on training data
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap = X
                y_bootstrap = y

            if isinstance(self.max_features, int):
                features = np.random.choice(n_features, self.max_features, replace=False)
            elif isinstance(self.max_features, float):
                n_features_subset = int(self.max_features * n_features)
                features = np.random.choice(n_features, n_features_subset, replace=False)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_bootstrap[:, features], y_bootstrap)
            self.estimators_.append(tree)

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
        for tree in self.estimators_:
            y_pred += tree.predict(X[:, features])
        return y_pred / len(self.estimators_)
