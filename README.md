# Linear Regression Algorithms

This repository contains Python implementations of the linear regression algorithm and related functions, implemented from scratch without using any external libraries.

## Files
    linear_regression.py: Implementation of the LinearRegression class for fitting a linear regression model using the gradient descent algorithm.
    metrics.py: Implementation of the mean squared error and adjusted R-squared functions for evaluating the performance of a linear regression model.
    cross_validation.py: Implementation of k-fold cross-validation for linear regression.

## Usage
1. Import the required functions and classes:

    from linear_regression import LinearRegression
    from metrics import mean_squared_error, adjusted_r_squared
    from cross_validation import cross_validation

2. Create an instance of the LinearRegression class and fit the model on the training data:
    model = LinearRegression()
    model.fit(X_train, y_train)

3. Predict the target values for the test data:
    y_pred = model.predict(X_test)

4. Evaluate the performance of the model using the mean squared error and adjusted R-squared metrics:
    mse = mean_squared_error(y_test, y_pred)
    r2 = adjusted_r_squared(y_test, y_pred, X_test, n_features)

5. Alternatively, you can perform k-fold cross-validation to evaluate the performance of the model:
    mse_scores, r2_scores = cross_validation(X, y, k=5)

## dependencies
The code in this repository does not depend on any external libraries, except for numpy which is used for matrix and array operations.
    import numpy as np

## License
This project is licensed under the MIT License - see the LICENSE file for details.
