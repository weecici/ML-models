# try to implement a NN without tensorflow, pytorch

import numpy as np


def ReLU(s):
    s[s < 0] = 0


class NeuralNetwork:
    def __init__(self, n_neurons_in_hidden_layers: list[int]):
        self.n_neurons = [0, *n_neurons_in_hidden_layers, 0]
        self.n_layers = len(self.n_neurons) - 1
        self.W = None
        self.B = None

    def fit(self, X, y):

        self.n_samples = X.shape[0]
        self.n_neurons[0] = X.shape[1]
        self.n_neurons[-1] = y.shape[0]

        self.W = [
            np.random.uniform(
                -0.1, 0.1, size=(self.n_neurons[i], self.n_neurons[i + 1])
            )
            for i in range(self.n_layers)
        ]

        self.B = [
            np.random.uniform(-0.1, 0.1, size=(1, self.n_neurons[i]))
            for i in range(1, self.n_layers + 1)
        ]

        for i in range(self.n_layers):
            y_pred = self.predict(X[i])

    def predict(self, X):
        y_pred = X
        for i in range(self.n_layers):
            y_pred = np.dot(y_pred, self.W[i]) + self.B[i]
            ReLU(y_pred)
        return y_pred


test = NeuralNetwork([3])
test.fit(np.zeros((3, 2)), np.zeros(2))
test.predict(np.zeros((1, 2)))
