import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class PolynomialRegression:
    def __init__(
        self, learning_rate: float = 1e-2, degree: int = 1, max_iter: int = 1000
    ):
        self.eta = learning_rate
        self.degree = degree
        self.max_iter = max_iter
        self.w = None
        self.poly = PolynomialFeatures(degree=self.degree)

    def transform(self, X):
        # X -> [1, X, X^2, ... X^{self.degree}]
        return self.poly.fit_transform(X)

    def fit(self, X, y):

        X = self.transform(X)
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))

        for _ in range(self.max_iter):
            y_pred = self._predict(X)
            self.w -= self.eta / n_samples * np.dot(X.T, y_pred - y)

    def _predict(self, X):
        return np.dot(X, self.w)

    def predict(self, X):
        X = self.transform(X)
        return self._predict(X)
