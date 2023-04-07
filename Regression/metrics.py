

def mean_squared_error(y_true, y_pred):
    """
    Compute mean squared error between true and predicted target values.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True target values.
    y_pred : array-like, shape (n_samples,)
        Predicted target values.

    Returns
    -------
    mse : float
        Mean squared error between true and predicted target values.

    """
    return np.mean((y_true - y_pred) ** 2)


def adjusted_r_squared(y_true, y_pred, X, n_features):
    """
    Compute the adjusted R-squared metric for a linear regression model.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True target values.
    y_pred : array-like, shape (n_samples,)
        Predicted target values.
    X : array-like, shape (n_samples, n_features)
        Input data.
    n_features : int
        Number of features in the linear regression model.

    Returns
    -------
    r2_adj : float
        Adjusted R-squared metric.

    """
    n_samples = len(y_true)
    ssr = np.sum((y_pred - y_true) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ssr / sst)
    r2_adj = 1 - (1 - r2) * ((n_samples - 1) / (n_samples - n_features - 1))
    return r2_adj
