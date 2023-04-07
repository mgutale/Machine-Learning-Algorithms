
def cross_validation(X, y, k=5):
    """
    Perform k-fold cross-validation for linear regression.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,)
        Target values.
    k : int, optional (default=5)
        Number of folds for cross-validation.

    Returns
    -------
    mse_scores : array-like, shape (k,)
        Mean squared error for each fold.
    r2_scores : array-like, shape (k,)
        R-squared metric for each fold.

    """
    n_samples = len(y)
    fold_size = n_samples // k
    mse_scores = []
    r2_scores = []

    # Shuffle data
    idx = np.random.permutation(n_samples)
    X_shuffled, y_shuffled = X[idx], y[idx]

    # Perform k-fold cross-validation
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_test, y_test = X_shuffled[start:end], y_shuffled[start:end]
        X_train, y_train = np.concatenate((X_shuffled[:start], X_shuffled[end:])), np.concatenate((y_shuffled[:start], y_shuffled[end:]))

        # Fit linear regression model on training data
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate model on test data
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    return np.array(mse_scores), np.array(r2_scores)
