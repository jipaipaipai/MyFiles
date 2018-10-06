# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:13:14 2018

@author: haoyu
"""

'''sigmoid函数'''
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

x = np.linspace(-10, 10, 500)
y = sigmoid(x)
plt.plot(x, y)
plt.show()

'''实现逻辑回归'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y<2, :2]
y = y[y<2]
plt.scatter(X[y==0,0], X[y==0,1], color='r')
plt.scatter(X[y==1,0], X[y==1,1], color='b')
plt.show()
#使用逻辑回归
from MLself.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
from MLself.LogisticRegression import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

log_reg.predict_proba(X_test)
log_reg.predict(X_test)
y_test

##
def x2(x1):
    return (-log_reg.coef_[0] * x1 - log_reg.intercept_) / log_reg.coef_[1]

x1_plot = np.linspace(4, 8, 1000)
x2_plot = x2(x1_plot)
plt.scatter(X[y==0,0], X[y==0,1], color='r')
plt.scatter(X[y==1,0], X[y==1,1], color='b')
plt.plot(x1_plot, x2_plot)
plt.show()


'''逻辑回归中添加多项式特征'''
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array(X[:,0]**2 + X[:,1]**2 < 1.5, dtype='int')
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

#使用逻辑回归
from MLself.LogisticRegression import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)
log_reg.score(X, y)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
def PolynomialLogisticRegression(degree):
    return Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('stan_scaler', StandardScaler()),
            ('log_reg', LogisticRegression())
            ])

poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X, y)
poly_log_reg.score(X, y)


'''sklearn中的逻辑回归'''
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(666)
X = np.random.normal(0, 1, size=(200, 2))
y = np.array(X[:,0]**2 +X[:,1] < 1.5, dtype='int')
for _ in range(20):
    y[np.random.randint(200)] = 1
plt.scatter(X[y==0,0], X[y==0,1])
plt.scatter(X[y==1,0], X[y==1,1])
plt.show()

from MLself.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)
#使用sklearn中的逻辑回归
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_test, y_test)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
def PolynomialLogisticRegression(degree):
    return Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('stan_scaler', StandardScaler()),
            ('log_reg', LogisticRegression())
            ])

poly_log_reg = PolynomialLogisticRegression(degree=2)
poly_log_reg.fit(X_train, y_train)
poly_log_reg.score(X_test, y_test)


'''OvR和OvO:用逻辑回归处理多分类任务'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

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









