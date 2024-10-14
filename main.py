import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing as pp

import matplotlib
import matplotlib.pyplot as plt

from polynomial_regression import PolynomialRegression

matplotlib.use("TKagg")

n_samples = 300
degree = 3

poly = pp.PolynomialFeatures(degree=degree)
np.random.seed(0)

X = 6 * np.random.rand(n_samples, 1) - 3
X_poly = poly.fit_transform(X)


coef = 5 * np.random.rand(degree + 1, 1)
y = np.dot(X_poly, coef) + np.random.randn(n_samples, 1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = PolynomialRegression(degree=3, max_iter=10000)
reg.fit(X_train, y_train)
y_pred = reg.predict(X)
print(coef)
print(reg.w)

# Visualization of original data and regression line
plt.scatter(X, y, color="blue", label="Original data")
plt.scatter(X, y_pred, color="red", label="Predicted data")
plt.legend()
plt.show()
