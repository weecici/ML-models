# try to implement a NN without tensorflow, pytorch
# references: https://machinelearningcoban.com/2017/02/24/mlp/

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from icecream.icecream import ic
from scipy import sparse

matplotlib.use("TKagg")


def convert_labels(y, C=3):
    Y = sparse.coo_matrix(
        (np.ones_like(y), (y, np.arange(len(y)))), shape=(C, len(y))
    ).toarray()
    return Y


def softmax(V):
    e_V = np.exp(V - np.max(V, axis=0, keepdims=True))
    Z = e_V / e_V.sum(axis=0)
    return Z


def cost(y, y_hat):
    return -np.sum(y * np.log(y_hat)) / y.shape[1]


class NeuralNetwork:
    def __init__(self, n_neurons_in_hidden_layers: list[int], learning_rate=1e-4):
        self.eta = learning_rate
        self.n_neurons = [0, *n_neurons_in_hidden_layers, 0]
        self.n_layers = len(self.n_neurons) - 1
        self.W = None
        self.B = None

    def fit(self, X, y):
        y = convert_labels(y)

        self.n_neurons[0] = X.shape[0]
        self.n_neurons[-1] = y.shape[0]
        N = X.shape[1]

        self.W = [
            np.random.uniform(-1, 1, size=(self.n_neurons[i], self.n_neurons[i + 1]))
            for i in range(self.n_layers)
        ]

        self.B = [np.zeros((self.n_neurons[i], 1)) for i in range(1, self.n_layers + 1)]

        # Gradient Descent
        for i in range(10001):

            # feedforward step
            Z = []
            A = [X]

            y_hat = X
            for j in range(self.n_layers):
                y_hat = np.dot(self.W[j].T, y_hat) + self.B[j]
                Z.append(y_hat)
                if j != self.n_layers - 1:
                    y_hat = np.maximum(y_hat, 0)
                    A.append(y_hat)

            y_hat = softmax(y_hat)

            # check cost after 1000 iterations
            if i % 1000 == 0:
                ic(i)
                ic(cost(y, y_hat))

            # backpropagation step

            E = (y_hat - y) / N
            gradient_Wi = np.dot(A[-1], E.T)

            self.W[-1] -= self.eta * gradient_Wi
            self.B[-1] -= self.eta * np.sum(E, axis=1, keepdims=True)

            for j in range(self.n_layers - 2, -1, -1):
                E = np.dot(self.W[j + 1], E)
                E[Z[j] <= 0] = 0
                gradient_Wi = np.dot(A[j], E.T)

                self.W[j] -= self.eta * gradient_Wi
                self.B[j] -= self.eta * np.sum(E, axis=1, keepdims=True)

    def predict(self, X):
        y_hat = X
        for j in range(self.n_layers):
            y_hat = np.dot(self.W[j].T, y_hat) + self.B[j]
            if j != self.n_layers - 1:
                y_hat = np.maximum(y_hat, 0)

        y_hat = softmax(y_hat)
        return np.argmax(y_hat, axis=0)


N = 10  # number of points per class
d0 = 2  # dimensionality
C = 3  # number of classes
X = np.zeros((d0, N * C))  # data matrix (each row = single example)
y = np.zeros(N * C, dtype="uint8")  # class labels

for j in range(C):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[:, ix] = np.c_[r * np.sin(t), r * np.cos(t)].T
    y[ix] = j

# init a NN with following structure: input -> 100 neurons -> 50 neurons -> 10 neurons -> output
test = NeuralNetwork([100, 50, 10])
test.fit(X, y)
y_pred = test.predict(X)
acc = np.sum(y_pred == y) / len(y)

# plt.plot(X[:N, 0], X[:N, 1], "bs", markersize=7)
# plt.plot(X[N : 2 * N, 0], X[N : 2 * N, 1], "ro", markersize=7)
# plt.plot(X[2 * N :, 0], X[2 * N :, 1], "g^", markersize=7)
# plt.xlim([-1.5, 1.5])
# plt.ylim([-1.5, 1.5])
# cur_axes = plt.gca()
# cur_axes.axes.get_xaxis().set_ticks([])
# cur_axes.axes.get_yaxis().set_ticks([])
# plt.show()

ic(acc)
