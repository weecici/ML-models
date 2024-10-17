import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self.__predict(x) for x in X]
        return np.array(predicted_labels)

    def __euclidianDistance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def __predict(self, x):
        distances = [self.__euclidianDistance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_neareast_labels = [self.y_train[k_index] for k_index in k_indices]
        return Counter(k_neareast_labels).most_common(1)[0][0]
