# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from explanator.local_exploration import local_gradient


x1 = np.random.uniform(0, 2, size=(1000, 3))
y1 = np.zeros(len(x1))
x2 = np.random.uniform(1, 2, size=(1000, 3))
y2 = np.ones(len(x1))
x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])


def test_logistic_neighbors():
    lr = LogisticRegression()
    lr.fit(x, y)
    grad = local_gradient(lr, x[100])
    print(lr.coef_)
    print(grad)


def test_svm_neighbors():
    svc = SVC(probability=True)
    svc.fit(x, y)
    grad = local_gradient(svc, x[1100])
    print(grad)


if __name__ == '__main__':
    test_logistic_neighbors()
    test_svm_neighbors()
