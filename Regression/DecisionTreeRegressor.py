import numpy as np

class DecisionTreeRegressor:
    """
    Decision tree for regression using mean squared error as the splitting criterion.

    Parameters
    ----------
    max_depth : int, optional (default=None)
        Maximum depth of the decision tree. If None, then the tree is grown until all leaves are pure.
    min_samples_split : int, optional (default=2)
        Minimum number of samples required to split a node.
    min_samples_leaf : int, optional (default=1)
        Minimum number of samples required in a leaf node.

    Attributes
    ----------
    tree_ : dict
        Dictionary representing the decision tree.

    Methods
    -------
    fit(X, y)
        Fit decision tree on training data (X, y).
    predict(X)
        Predict target values for input data (X).

    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def _split_node(self, X, y, depth):
        n_samples, n_features = X.shape

        # Determine best split for current node
        best_split_feature = None
        best_split_value = None
        best_split_score = np.inf
        for feature in range(n_features):
            for value in np.unique(X[:, feature]):
                left_mask = X[:, feature] <= value
                right_mask = X[:, feature] > value
                if np.sum(left_mask) >= self.min_samples_leaf and np.sum(right_mask) >= self.min_samples_leaf:
                    left_y = y[left_mask]
                    right_y = y[right_mask]
                    score = np.mean((left_y - np.mean(left_y)) ** 2) + np.mean((right_y - np.mean(right_y)) ** 2)
                    if score < best_split_score:
                        best_split_feature = feature
                        best_split_value = value
                        best_split_score = score

        # Create current node and recursively split child nodes
        node = {}
        node['feature'] = best_split_feature
        node['value'] = best_split_value
        node['left'] = None
        node['right'] = None
        if depth < self.max_depth and best_split_score < np.inf:
            left_mask = X[:, best_split_feature] <= best_split_value
            right_mask = X[:, best_split_feature] > best_split_value
            node['left'] = self._split_node(X[left_mask], y[left_mask], depth + 1)
            node['right'] = self._split_node(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        """
        Fit decision tree on training data (X, y).

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
        self.tree_ = self._split_node(X, y, 0)

        return self

    def _predict_sample(self, x, node):
        if node['left'] is None and node['right'] is None:
            return np.mean(node['y'])
        elif x[node['feature']] <= node['value']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X):
       
