import numpy as np
import math


class NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for cl in self.classes:
            X_cl = X[cl == y]
            self.mean[cl] = X_cl.mean(axis=0)
            self.var[cl] = X_cl.var(axis=0)
            self.priors[cl] = X_cl.shape[0] / n_samples

    def gaussian(self, x, cl):
        return np.log(
            np.exp(-((x - self.mean[cl]) ** 2) / (2 * self.var[cl]))
            / np.sqrt(2 * math.pi * self.var[cl])
        )

    def _predict(self, x):
        posteriors = []
        for i, cl in enumerate(self.classes):
            prior = np.log(self.priors[cl])
            class_conditional = np.sum(self.gaussian(x, i))
            posterior = class_conditional + prior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
