import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=1e-4, max_iter=1000) -> None:
        self.eta = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __predict_probabilities(self, X):
        return self.__sigmoid(np.dot(X, self.w) + self.b)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        # optimize with GD
        for _ in range(self.max_iter):
            errors = self.__predict_probabilities(X) - y

            self.w -= self.eta / n_samples * np.dot(X.T, errors)
            self.b -= self.eta / n_samples * np.sum(errors)

    def predict(self, X):
        y_pred = self.__predict_probabilities(X)
        return [1 if yi >= 0.5 else 0 for yi in y_pred]
