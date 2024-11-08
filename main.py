import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing as pp

import matplotlib
import matplotlib.pyplot as plt
from softmax_regression import SoftmaxRegression

matplotlib.use("TKagg")

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


reg = SoftmaxRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(y_pred)
print(y_test)
acc = np.sum(y_pred == y_test) / len(y_test)
print(acc)
