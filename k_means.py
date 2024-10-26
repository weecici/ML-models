import numpy as np
from scipy.spatial.distance import cdist


class K_means:
    def __init__(self, max_iters=1000, k=3):
        self.max_iters = max_iters
        self.k = k

    def find_labels(self, X):
        D = cdist(X, self.centers)
        return np.argmin(D, axis=1)

    def find_centers(self, X):
        labels = self.find_labels(X)
        self.centers = np.zeros((self.k, X.shape[1]))

        for i in range(self.k):
            Xi = X[labels == i, :]
            self.centers[i, :] = np.mean(Xi, axis=0)

    def fit(self, X):

        self.centers = X[: self.k, :]

        for _ in range(self.max_iters):
            self.find_centers(X)
