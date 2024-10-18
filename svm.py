import numpy as np


class SVC:
    def __init__(self, learning_rate=1e-3, __lambda=1e-2, max_iter=1000):
        self.eta = learning_rate
        self.__lambda = __lambda
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                cond = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if cond:
                    self.w -= self.eta * 2 * self.__lambda * self.w
                else:
                    self.w -= self.eta * (
                        2 * self.__lambda * self.w - np.dot(x_i, y[idx])
                    )
                    self.b += y[idx]

    def predict(self, X):
        linear_approximation = np.dot(X, self.w) + self.b
        return np.sign(linear_approximation)
