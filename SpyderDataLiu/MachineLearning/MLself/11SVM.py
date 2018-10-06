# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 09:31:02 2018

@author: haoyu
"""

#有条件的最优化问题
'''scikit-learn中的SVM算法封装'''
#使用SVN同使用kNN一样，要先将数据标准化
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y<2,:2]#SVM二分类，取两种花，为了可视化，取前两列特征
y = y[y<2]
plt.scatter(X[y==0,0], X[y==0,1], color='r')
plt.scatter(X[y==1,0], X[y==1,1], color='b')
plt.show()

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(X)
X_standard = standardScaler.transform(X)
from sklearn.svm import LinearSVC#SVC(SupportVectorClassifier),目前是线性SVM
svc = LinearSVC(C=1e9)
svc.fit(X_standard, y)


'''SVM中使用多项式特征'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#不使用真使数据集，使用datasets的生成数据函数
X, y = datasets.make_moons()
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()
X, y = datasets.make_moons(noise=0.15, random_state=666)#太规整，生成噪音
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

#使用多项式特征的SVM
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
def PolynomialSVC(degree, C=1.0):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('linearSVC', LinearSVC(C=C))
    ])
poly_svc = PolynomialSVC(degree=3)
poly_svc.fit(X, y)


'''使用多项式核函数的SVM'''
from sklearn.svm import SVC
def PolynomialKernelSVC(degree, C=1.0):
    return Pipeline([
        
        ('std_scaler', StandardScaler()),
        ('kernelSVC', SVC(kernel='poly', degree=degree, C=C))
    ])
poly_kernel_svc = PolynomialKernelSVC(degree=3)
poly_kernel_svc.fit(X, y)


'''SVM思想解决回归问题'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
y = boston.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
from sklearn.svm import LinearSVR#SVR=SupportVectorRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
def StandardLinearSVR(epsilon=0.1):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('linearSVR', LinearSVR(epsilon=epsilon))
    ])
svr = StandardLinearSVR()
svr.fit(X_train, y_train)
svr.score(X_test, y_test)















