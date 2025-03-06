import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, max_iter=1000):
        self.eta = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def __weighted_sum(self, X):
        return np.dot(X, self.w) + self.b

    def __sign(self, z):
        return 1 if z >= 0 else 0

    def __loss_func(self, y_pred, y):
        return y_pred - y

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.max_iter):
            errors = self.__loss_func(self.predict(X), y)

            self.w -= self.eta * np.dot(X.T, errors)
            self.b -= self.eta * np.dot(y, errors)

    def predict(self, X):
        Z = self.__weighted_sum(X)
        return np.array([self.__sign(z) for z in Z])
