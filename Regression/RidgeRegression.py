import numpy as np

class RidgeRegression:
    """
    Ridge regression model using closed-form solution.

    Parameters
    ----------
    alpha : float, optional (default=1)
        Regularization strength.

    Attributes
    ----------
    weights : array-like, shape (n_features,)
        Coefficients of the linear regression model.
    bias : float
        Intercept (bias) of the linear regression model.

    Methods
    -------
    fit(X, y)
        Fit ridge regression model on training data (X, y).
    predict(X)
        Predict target values for input data (X).

    """

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit ridge regression model on training data (X, y).

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

        # Add bias term to input data
        X = np.c_[X, np.ones(n_samples)]

        # Compute weights using closed-form solution
        A = np.dot(X.T, X) + self.alpha * np.eye(n_features + 1)
        B = np.dot(X.T, y)
        self.weights = np.linalg.solve(A, B)
        self.bias = self.weights[-1]
        self.weights = self.weights[:-1]

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
        n_samples = X.shape[0]

        # Add bias term to input data
        X = np.c_[X, np.ones(n_samples)]

        return np.dot(X, self.weights) + self.bias
