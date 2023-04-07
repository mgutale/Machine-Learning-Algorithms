import numpy as np

class LogisticRegression:
    """
    Logistic Regression classifier.

    Parameters:
    -----------
    lr : float
        Learning rate for gradient descent. Default is 0.1.
    n_iters : int
        Number of iterations for gradient descent. Default is 1000.

    Attributes:
    -----------
    weights : array-like, shape = [n_features,]
        Weights for each feature.
    bias : float
        Bias term.
    """

    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Fit Logistic Regression to training data.

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
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
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
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_cls)
    
    def _sigmoid(self, x):
        """
        Compute sigmoid function.

        Parameters:
        -----------
        x : float
            Input to sigmoid function.

        Returns:
        --------
        output : float
            Sigmoid function output.
        """
        return 1 / (1 + np.exp(-x))
