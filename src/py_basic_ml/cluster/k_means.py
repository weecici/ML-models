import numpy as np
from scipy.spatial.distance import cdist


class K_means:
    def __init__(self, k=3, max_iters=1000):
        self.max_iters = max_iters
        self.k = k

    def __find_labels(self, X):
        # calculate all distanes from all samples to all centers
        D = cdist(X, self.centers)

        # find the index of column that has smallest distance
        return np.argmin(D, axis=1)

    def __find_centers(self, X):
        # fix lables, find centers
        labels = self.__find_labels(X)
        self.centers = np.zeros((self.k, X.shape[1]))

        for i in range(self.k):
            Xi = X[labels == i, :]
            self.centers[i, :] = np.mean(Xi, axis=0)

    def fit(self, X):

        self.centers = X[: self.k, :]

        for i in range(self.max_iters):

            old_centers = set([tuple(center) for center in self.centers])
            self.__find_centers(X)

            # check if centers we found have converged
            if set([tuple(center) for center in self.centers]) == old_centers:
                break
