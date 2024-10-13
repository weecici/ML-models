import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
from icecream.icecream import ic

from perceptron import Perceptron

matplotlib.use("TKagg")


X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

perc = Perceptron()
perc.fit(X_train, y_train)
pred = perc.predict(X_test)
print(pred)

acc = np.sum(pred == y_test) / len(y_test)
print(acc)
