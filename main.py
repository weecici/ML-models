import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
from sklearn import preprocessing as pp
from softmax_regression import SoftmaxRegression
from mnist import MNIST
from sklearn.metrics import accuracy_score

matplotlib.use("TKagg")


path = "/home/cici/Downloads/mnist/"

mntrain = MNIST(path)
mntrain.load_training()
X_train = np.asarray(mntrain.train_images) / 255
y_train = np.array(mntrain.train_labels)


mntest = MNIST(path)
mntest.load_testing()
X_test = np.asarray(mntest.test_images) / 255
y_test = np.array(mntest.test_labels)
