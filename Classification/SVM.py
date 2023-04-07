import numpy as np

class SVM:
    """
    Support Vector Machine (SVM) classifier using gradient descent.

    Parameters:
    -----------
    lr : float
        Learning rate for gradient descent. Default is 0.1.
    C : float
        Regularization parameter. Default is 1.
    n_iters : int
        Number of iterations for gradient descent. Default is 1000.

    Attributes:
    -----------
    w : array-like, shape = [n_features,]
        Weight vector.
    b : float
        Bias term.
    """

    def __init__(self, lr=0.1, C=1, n_iters=1000):
        self.lr = lr
        self.C = C
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        """
        Fit SVM to training data.

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
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = y_[i] * (np.dot(X[i], self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.C * self.w)
                else:
                    self.w -= self.lr * (2 * self.C * self.w - np.dot(X[i], y_[i]))
                    self.b -= self.lr * y_[i]
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
        linear_model = np.dot(X, self.w) - self.b
        y_pred = np.sign(linear_model)
        return y_pred
