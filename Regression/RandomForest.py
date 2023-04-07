import numpy as np

class RandomForestRegressor:
    """
    Random forest for regression using decision trees as base estimators.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        Number of trees in the forest.
    max_depth : int, optional (default=None)
        Maximum depth of each decision tree. If None, then the tree is grown until all leaves are pure.
    min_samples_split : int, optional (default=2)
        Minimum number of samples required to split a node.
    min_samples_leaf : int, optional (default=1)
        Minimum number of samples required in a leaf node.

    Attributes
    ----------
    trees_ : list of DecisionTreeRegressor objects
        List of decision trees in the forest.

    Methods
    -------
    fit(X, y)
        Fit random forest on training data (X, y).
    predict(X)
        Predict target values for input data (X).

    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees_ = []

    def fit(self, X, y):
        """
        Fit random forest on training data (X, y).

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
        for i in range(self.n_estimators):
            # Bootstrap sample the data
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Create decision tree and add to forest
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees_.append(tree)

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
        for tree in self.trees_:
            y_pred += tree.predict(X)
        return y_pred / len(self.trees_)
