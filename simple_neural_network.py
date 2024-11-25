# try to implement a NN without tensorflow, pytorch

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from icecream.icecream import ic
from scipy import sparse

matplotlib.use("TKagg")


def softmax(V):
    e_V = np.exp(V - np.max(V, axis=1, keepdims=True))
    Z = e_V / e_V.sum(axis=1)[:, np.newaxis]
    return Z


class NeuralNetwork:
    def __init__(self, n_neurons_in_hidden_layers: list[int], learning_rate=1e-2):
        self.eta = learning_rate
        self.n_neurons = [0, *n_neurons_in_hidden_layers, 0]
        self.n_layers = len(self.n_neurons) - 1
        self.W = None
        self.B = None

    def fit(self, X, y):
        y = np.eye(len(np.unique(y)))[y]

        n_samples = X.shape[0]
        self.n_neurons[0] = X.shape[1]
        self.n_neurons[-1] = y.shape[1]

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

        # SGD
        for i in range(100):

            # feedforward
            Z = []
            A = [X]

            y_hat = X
            for j in range(self.n_layers):
                y_hat = np.dot(y_hat, self.W[j]) + self.B[j]
                Z.append(y_hat)
                if j != self.n_layers - 1:
                    y_hat = np.maximum(y_hat, 0)
                    A.append(y_hat)

            # ic(y_hat)
            y_hat = softmax(y_hat)

            gradient_Bi = y_hat - y
            gradient_Wi = np.dot(A[-1].T, gradient_Bi)

            self.W[-1] -= self.eta * gradient_Wi
            self.B[-1] -= self.eta * np.sum(gradient_Bi, axis=0)

            for j in range(self.n_layers - 2, -1, -1):
                gradient_Bi = np.dot(gradient_Bi, self.W[j + 1].T)
                gradient_Bi[gradient_Bi < 0] = 0
                gradient_Wi = np.dot(A[j].T, gradient_Bi)

                self.W[j] -= self.eta * gradient_Wi
                self.B[j] -= self.eta * np.sum(gradient_Bi, axis=0)

    def predict(self, X):
        y_pred = X
        for i in range(self.n_layers):
            y_pred = np.dot(y_pred, self.W[i]) + self.B[i]
            if j != self.n_layers - 1:
                y_pred = np.maximum(y_pred, 0)
        y_pred = softmax(y_pred)
        return np.argmax(y_pred, axis=1)


N = 10  # number of points per class
d0 = 2  # dimensionality
C = 3  # number of classes
X = np.zeros((d0, N * C))  # data matrix (each row = single example)
y = np.zeros(N * C, dtype="int")  # class labels

for j in range(C):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[:, ix] = np.c_[r * np.sin(t), r * np.cos(t)].T
    y[ix] = j


test = NeuralNetwork([10])
ic(X)
test.fit(X.T, y)
ic(X)
y_pred = test.predict(X.T)
acc = np.sum(y_pred == y) / len(y)
ic(acc)
