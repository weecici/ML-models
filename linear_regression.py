import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=1e-2, max_iter=1000):
        self.eta = learning_rate
        self.max_iter = max_iter
        self.w = None

    def gradient(self, X, errors):
        gradient_w = (-2 / X.shape[0]) * np.dot(X.T, errors)
        gradient_b = (-2 / X.shape[0]) * np.sum(errors)
        return gradient_w, gradient_b

    def gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features + 1)

        for _ in range(self.max_iter):
            errors = y - self.predict(X)

            gradient_w, gradient_b = self.gradient(X, errors)

            self.w[1:] -= self.eta * gradient_w
            self.w[0] -= self.eta * gradient_b

    def normal_equation(self, X, y):
        # X_bar = X with a col full of 1's
        X_bar = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        self.w = np.dot(np.linalg.pinv(np.dot(X_bar.T, X_bar)), np.dot(X_bar.T, y))

    def fit(self, X, y):
        self.normal_equation(X, y)

    def predict(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]


class LinearRegressionL2(LinearRegression):

    def __init__(self, learning_rate=1e-2, l2_coef=1e-2, max_iter=1000):
        super().__init__(learning_rate, max_iter)
        self.l2 = l2_coef

    def gradient(self, X, n_samples, errors):
        gradient_w = (-1 / n_samples) * np.dot(X.T, errors) + self.l2 * self.w
        gradient_b = (-1 / n_samples) * np.sum(errors)
        return gradient_w, gradient_b

    def normal_equation(self, X, y):
        X_bar = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        # A = pseudoinverse(X^T*X + 2*lambda*I)
        A = np.linalg.pinv(
            np.dot(X_bar.T, X_bar) + 2 * self.l2 * np.identity(X.shape[1] + 1)
        )
        self.w = np.dot(A, np.dot(X_bar.T, y))
