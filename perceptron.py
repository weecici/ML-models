import numpy as np
from icecream.icecream import ic


class Perceptron:
    def __init__(self, learning_rate=0.1, max_iter=1000):
        self.w = None
        self.b = None
        self.max_iter = max_iter
        self.eta = learning_rate

    def linear(self, X):
        return np.dot(X, self.w) + self.b

    def activation_func(self, z):
        return 1 if z >= 0 else 0

    def loss_func(self, y_pred, y):
        return y_pred - y

    def predict(self, X):
        Z = self.linear(X)
        return np.array([self.activation_func(z) for z in Z])

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.max_iter):
            errors = self.loss_func(self.predict(X), y)

            self.w -= self.eta * np.dot(X.T, errors)
            self.b -= self.eta * np.dot(y, errors)
