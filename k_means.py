import numpy as np
from scipy.spatial.distance import cdist


class K_means:
    def __init__(self, max_iters=1000, k=3):
        self.max_iters = max_iters
        self.k = k

    def find_labels(self, X):
        # calculate all distanes from all samples to all centers
        D = cdist(X, self.centers)

        # find the column index that has smallest distance
        return np.argmin(D, axis=1)

    def find_centers(self, X):
        labels = self.find_labels(X)
        self.centers = np.zeros((self.k, X.shape[1]))

        for i in range(self.k):
            Xi = X[labels == i, :]
            self.centers[i, :] = np.mean(Xi, axis=0)

    def fit(self, X):

        self.centers = X[: self.k, :]

        for i in range(self.max_iters):

            old_centers = set([tuple(center) for center in self.centers])
            self.find_centers(X)

            # check if centers we found have converged
            if set([tuple(center) for center in self.centers]) == old_centers:
                break
