import numpy as np
from DecisionTreeClassifier import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Random Forest classifier using Decision Trees.

    Parameters:
    -----------
    n_estimators : int
        Number of trees in the forest. Default is 100.
    max_depth : int or None
        Maximum depth of the Decision Trees. If None, the trees are grown until all leaves are pure.
        Default is None.
    min_samples_split : int
        Minimum number of samples required to split an internal node. Default is 2.
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node. Default is 1.

    Attributes:
    -----------
    trees : list
        List of DecisionTreeClassifier objects.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
    
    def fit(self, X, y):
        """
        Fit Random Forest to training data.

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
        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf)
            tree_indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree_X = X[tree_indices]
            tree_y = y[tree_indices]
            tree.fit(tree_X, tree_y)
            self.trees.append(tree)
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
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
