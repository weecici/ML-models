import numpy as np


class SoftmaxRegression:
    def __init__(self, learning_rate=1e-4, max_iter=1000) -> None:
        self.eta = learning_rate
        self.max_iter = max_iter
        self.W = None

    def __softmax(self, Z):
        # calculate stable softmax with a constant c = max_i x_i to avoid overflow
        e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        A = e_Z / e_Z.sum(axis=0)
        return A

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # add a column of 1's to X
        _X = np.hstack((np.ones((X.shape[0], 1)), X))

        # one-hot coding y
        _Y = np.eye(n_classes)[y]

        self.W = np.zeros((n_classes, n_features + 1))

        for _ in range(self.max_iter):
            Z = np.dot(self.W, _X.T)
            A = self.__softmax(Z)
            self.W -= self.eta * np.dot((A - _Y.T), _X)

    def predict(self, X):

        # add a column of 1's to X
        _X = np.hstack((np.ones((X.shape[0], 1)), X))
        A = self.__softmax(np.dot(self.W, _X.T))
        return np.argmax(A[:, :], axis=0)
