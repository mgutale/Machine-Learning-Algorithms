import numpy as np

class LassoRegression:
    """
    Lasso regression model using coordinate descent.

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
        Fit lasso regression model on training data (X, y).
    predict(X)
        Predict target values for input data (X).

    """

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y, max_iter=1000, tol=1e-4):
        """
        Fit lasso regression model on training data (X, y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training input data.
        y : array-like, shape (n_samples,)
            Target values.
        max_iter : int, optional (default=1000)
            Maximum number of iterations for coordinate descent algorithm.
        tol : float, optional (default=1e-4)
            Tolerance for stopping criterion.

        Returns
        -------
        self : object
            Returns self.

        """
        n_samples, n_features = X.shape

        # Add bias term to input data
        X = np.c_[X, np.ones(n_samples)]

        # Initialize weights using least squares solution
        self.weights = np.linalg.lstsq(X, y, rcond=None)[0]
        self.bias = self.weights[-1]
        self.weights = self.weights[:-1]

        # Implement coordinate descent algorithm for Lasso
        for i in range(max_iter):
            old_weights = self.weights.copy()

            for j in range(n_features):
                # Compute residual
                r = y - np.dot(X, self.weights) - self.bias + self.weights[j] * X[:, j]

                # Compute coordinate-wise update
                z_j = np.dot(X[:, j], r) / n_samples
                if z_j < -self.alpha / 2:
                    self.weights[j] = (z_j + self.alpha / 2) / np.linalg.norm(X[:, j])
                elif z_j > self.alpha / 2:
                    self.weights[j] = (z_j - self.alpha / 2) / np.linalg.norm(X[:, j])
                else:
                    self.weights[j] = 0

            # Compute bias term
            self.bias = np.mean(y - np.dot(X[:, :-1], self.weights))

            # Check stopping criterion
            if np.max(np.abs(self.weights - old_weights)) < tol:
                break

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
