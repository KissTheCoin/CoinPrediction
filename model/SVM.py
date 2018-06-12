# -*- coding: utf-8 -*-
# @Time    : 2018/6/12 10:37
# @Author  : Ross
# @File    : SVM.py

from sklearn.svm import SVR


def build(C=1.0, epsilon=0.1, degree=3, gamma='auto'):
    model = SVR(C=C, epsilon=epsilon, degree=degree, gamma=gamma)
    return model
