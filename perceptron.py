import numpy as np


class Perceptron:
    def __init__(self, learning_rate=1e-2, max_iter=1000):
        self.w = None
        self.b = None
        self.max_iter = max_iter
        self.eta = learning_rate

    def linear(self, X):
        return np.dot(X, self.w) + self.b

    def activation_func(self, z):
        return 1 if z >= 0 else 0

    def loss_func(self, prediction, target):
        return np.sum(prediction - target)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        z = self.linear(x)
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.max_iter):
            for x, yi in zip(X, y):
                y_pred = self._predict(x)
                errors = self.loss_func(y_pred, yi)

                self.w -= self.eta * errors * x
                self.b -= self.eta * errors * yi
