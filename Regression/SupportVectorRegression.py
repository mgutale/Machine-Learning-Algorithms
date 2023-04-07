import numpy as np
from scipy.optimize import minimize


class SVR:
    """
    Support Vector Regression model using epsilon-insensitive loss.

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    epsilon : float, optional (default=0.1)
        Epsilon-tube within which no penalty is associated.

    Attributes
    ----------
    dual_coef_ : array-like, shape (n_support_vectors,)
        Dual coefficients of the support vectors.
    support_vectors_ : array-like, shape (n_support_vectors, n_features)
        Support vectors.
    bias : float
        Intercept (bias) of the linear regression model.

    Methods
    -------
    fit(X, y)
        Fit SVR model on training data (X, y).
    predict(X)
        Predict target values for input data (X).

    """

    def __init__(self, C=1.0, epsilon=0.1):
        self.C = C
        self.epsilon = epsilon
        self.dual_coef_ = None
        self.support_vectors_ = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit SVR model on training data (X, y).

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

        # Define objective function and constraints for quadratic programming
        def objective(dual_coef):
            return 0.5 * np.sum(dual_coef ** 2) - np.sum(self.epsilon * np.abs(y - np.dot(X, dual_coef)))

        def constraint(dual_coef):
            return np.dot(y - np.dot(X, dual_coef), np.ones(n_samples))  # Sum of dual_coef * y = 0

        constraints = [{'type': 'eq', 'fun': constraint}]
        bounds = [(0, self.C) for i in range(n_samples)]

        # Solve quadratic programming problem
        initial_guess = np.zeros(n_samples)
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        # Compute support vectors and dual coefficients
        self.dual_coef_ = result.x
        self.support_vectors_ = X[self.dual_coef_ > 0]
        self.dual_coef_ = self.dual_coef_[self.dual_coef_ > 0] * (y[self.dual_coef_ > 0] - self.epsilon * np.sign(y[self.dual_coef_ > 0] - np.dot(self.support_vectors_, self.dual_coef_)))
        self.bias = np.mean(y - np.dot(self.support_vectors_, self.dual_coef_))

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
        return np.dot(X, self.dual_coef_) + self.bias
