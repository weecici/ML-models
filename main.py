import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing as pp

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TKagg")


from linear_regression import LinearRegression, LinearRegressionL2

X, y = datasets.make_regression(300, 1, noise=1, random_state=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.w)
print(reg.b)

reg = LinearRegressionL2()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(reg.w)
print(reg.b)

plt.scatter(X_test, y_test, color="blue")
plt.scatter(X_test, y_pred, color="red")
plt.show()
