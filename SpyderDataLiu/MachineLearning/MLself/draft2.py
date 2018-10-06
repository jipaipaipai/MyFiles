# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:35:22 2018

@author: haoyu
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_digits()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)

##OvO和OvR方法1
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()#默认OvR
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

log_reg2 = LogisticRegression(multi_class='multinomial', solver='newton-cg')#调用OvO
log_reg2.fit(X_train, y_train)
log_reg2.score(X_test, y_test)


##OvO和OvR方法2
from sklearn.multiclass import OneVsRestClassifier
ovr = OneVsRestClassifier(log_reg)
ovr.fit(X_train, y_train)
ovr.score(X_test, y_test)

from sklearn.multiclass import OneVsOneClassifier
ovo = OneVsOneClassifier(log_reg)
ovo.fit(X_train, y_train)
ovo.score(X_test, y_test)
