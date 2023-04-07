import numpy as np

class BayesianRegression:
    """
    Bayesian linear regression model using MCMC sampling.

    Parameters
    ----------
    n_samples : int, optional (default=10000)
        Number of samples to draw from the posterior distribution.
    burn_in : int, optional (default=1000)
        Number of burn-in samples to discard.
    alpha : float, optional (default=1)
        Hyperparameter for the prior distribution on the weights.

    Attributes
    ----------
    weights : array-like, shape (n_features,)
        Coefficients of the linear regression model.
    bias : float
        Intercept (bias) of the linear regression model.

    Methods
    -------
    fit(X, y)
        Fit Bayesian linear regression model on training data (X, y).
    predict(X)
        Predict target values for input data (X).

    """

    def __init__(self, n_samples=10000, burn_in=1000, alpha=1):
        self.n_samples = n_samples
        self.burn_in = burn_in
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit Bayesian linear regression model on training data (X, y).

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

        # Initialize weights and bias
        self.weights = np.zeros(n_features + 1)
        self.bias = 0

        # Compute posterior distribution using MCMC sampling
        posterior_samples = []
        for i in range(self.n_samples + self.burn_in):
            # Sample weights from posterior distribution
            weights_sample = np.random.multivariate_normal(self.weights, self.alpha * np.eye(n_features + 1))

            # Compute likelihood and prior probabilities
            likelihood = np.exp(-0.5 * np.sum((y - np.dot(X, weights_sample)) ** 2))
            prior = np.exp(-0.5 * np.dot(weights_sample, np.dot(np.eye(n_features + 1), weights_sample)))

            # Compute acceptance probability and accept or reject sample
            acceptance_prob = likelihood * prior
            if i >= self.burn_in and np.random.uniform() < acceptance_prob:
                posterior_samples.append(weights_sample)

        # Compute mean of posterior distribution as final weights and bias
        self.weights = np.mean(posterior_samples, axis=0)[:-1]
        self.bias = np.mean(posterior_samples, axis=0)[-1]

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
