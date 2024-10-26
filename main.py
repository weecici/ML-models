import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing as pp

import matplotlib
import matplotlib.pyplot as plt
from k_means import K_means

matplotlib.use("TKagg")

X, y_true = datasets.make_blobs(
    n_samples=300, centers=3, cluster_std=0.60, random_state=0
)

km = K_means()
km.fit(X)


plt.scatter(X[:, 0], X[:, 1], color="blue")
plt.scatter(km.centers[:, 0], km.centers[:, 1], color="red")
plt.show()
