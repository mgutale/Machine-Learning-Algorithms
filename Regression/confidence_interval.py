import numpy as np
from scipy.stats import t

def confidence_interval(X, y, weights, alpha=0.05):
    """
    Compute confidence interval for the regression coefficients.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,)
        Target values.
    weights : array-like, shape (n_features,)
        Coefficients of the linear regression model.
    alpha : float, optional (default=0.05)
        Significance level.

    Returns
    -------
    conf_int : array-like, shape (n_features, 2)
        Confidence interval for each regression coefficient.

    """
    # Compute residuals and standard error of coefficients
    y_pred = np.dot(X, weights)
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / (X.shape[0] - X.shape[1] - 1)
    se = np.sqrt(np.diag(np.linalg.inv(np.dot(X.T, X))) * mse)

    # Compute t-statistic and confidence interval for each coefficient
    t_stat = t.ppf(1 - alpha/2, X.shape[0] - X.shape[1] - 1)
    conf_int = np.zeros((X.shape[1], 2))
    conf_int[:, 0] = weights - t_stat * se
    conf_int[:, 1] = weights + t_stat * se
    
    return conf_int
