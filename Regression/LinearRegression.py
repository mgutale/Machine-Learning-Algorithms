import numpy as np

class LinearRegression:
    """
    Linear regression model using gradient descent algorithm for minimizing mean squared error.

    Parameters
    ----------
    learning_rate : float, optional (default=0.01)
        Learning rate for gradient descent.
    iterations : int, optional (default=1000)
        Number of iterations for gradient descent.

    Attributes
    ----------
    weights : array-like, shape (n_features,)
        Coefficients of the linear regression model.
    bias : float
        Intercept (bias) of the linear regression model.

    Methods
    -------
    fit(X, y)
        Fit linear regression model on training data (X, y).
    predict(X)
        Predict target values for input data (X).

    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit linear regression model on training data (X, y).

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
        # Initialize weights and bias to zero
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for i in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
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
        return np.dot(X, self.weights) + self.bias
