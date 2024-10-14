import numpy as np
from icecream.icecream import ic


class LinearRegression:

    def __init__(self, learning_rate=0.001, max_iter=1000):
        self.eta = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        ic(X.shape)
        for _ in range(self.max_iter):
            y_predicted = np.dot(X, self.w) + self.b

            gradient_w = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            gradient_b = (1 / n_samples) * np.sum(y_predicted - y)
            self.w = self.eta * gradient_w
            self.b -= self.eta * gradient_b

    def predict(self, X):
        return np.dot(X, self.w) + self.b
