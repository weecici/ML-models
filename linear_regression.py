import numpy as np
from icecream.icecream import ic


class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.eta = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        ic(X.shape)
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights = self.eta * dw
            self.bias -= self.eta * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
