import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=1e-2, max_iter=1000):
        self.eta = learning_rate
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def gradient(self, X, n_samples, errors):
        gradient_w = (1 / n_samples) * np.dot(X.T, errors)
        gradient_b = (1 / n_samples) * np.sum(errors)
        return gradient_w, gradient_b

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):

            errors = y - self.predict(X)

            gradient_w, gradient_b = self.gradient(X, n_samples, errors)

            self.w -= self.eta * gradient_w
            self.b -= self.eta * gradient_b

    def predict(self, X):
        return np.dot(X, self.w) + self.b


class LinearRegressionL2(LinearRegression):

    def __init__(self, learning_rate=1e-2, max_iter=1000, l2_coef=1e-2):
        super().__init__(learning_rate, max_iter)
        self.l2 = l2_coef

    def gradient(self, X, n_samples, errors):
        gradient_w = (-1 / n_samples) * np.dot(X.T, errors) + self.l2 * self.w
        gradient_b = (-1 / n_samples) * np.sum(errors)
        return gradient_w, gradient_b
