import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=1e-4, max_iter=1000) -> None:
        self.w = None
        self.b = None
        self.eta = learning_rate
        self.max_iter = max_iter

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _predict(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):
            y_predict = self._predict(X)

            diff = y_predict - y

            self.w -= self.eta / n_samples * np.dot(X.T, diff)
            self.b -= self.eta / n_samples * np.sum(diff)

    def predict(self, X):
        y_predict = self._predict(X)
        return [1 if yi > 0.5 else 0 for yi in y_predict]
