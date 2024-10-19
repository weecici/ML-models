import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing as pp

import matplotlib
import matplotlib.pyplot as plt
from linear_regression import LinearRegression, LinearRegressionL2

matplotlib.use("TKagg")

X, y = datasets.make_regression(n_samples=200, n_features=1, noise=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

reg = LinearRegression()
reg.fit(X_train, y_train)

print(reg.w)

reg = LinearRegressionL2()
reg.fit(X_train, y_train)

print(reg.w)

y_pred = reg.predict(X_test)

plt.scatter(X_test, y_pred, color="blue")
plt.scatter(X_test, y_test, color="red")
plt.show()
