import numpy as np


class LogisticRegression:
    def __init__(self, lr=1e-4, max_iters=1000) -> None:
        self.w = None
        self.b = None
        self.lr = lr
        self.max_iters = max_iters

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __predict(self, X):
        return self.__sigmoid(np.dot(X, self.w) + self.b)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iters):
            y_predict = self.__predict(X)

            diff = y_predict - y

            self.w -= self.lr / n_samples * np.dot(X.T, diff)
            self.b -= self.lr / n_samples * np.sum(diff)

    def predict(self, X):
        y_predict = self.__predict(X)
        return [1 if yi > 0.5 else 0 for yi in y_predict]
