# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 13:34:10 2018

@author: haoyu
"""

'''simple linear regression1'''

import numpy as np

class SimpleLinearRegression1:
    
    def __inint__(self):
        '''初始化SLG模型'''
        self.a_ = None
        self.b_ = None
        
    def fit(self, x_train, y_train):
        '''训练数据集'''
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2
        
        self.a_ = num / d
        self.b_ = y_mean - self.a_ *x_mean
        
        return self
    
    def predict(self, x_predict):
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
    
    def __repr__(self):
        return 'SimpleLinearRegression1()'
    
reg1 = SimpleLinearRegression1()
reg1.fit([1,2,3], [1,2,3]) 
reg1.predict([0.2])        
        
        
'''向量化运算'''
import numpy as np

class SimpleLinearRegression2:
    
    def __inint__(self):
        '''初始化SLG模型'''
        self.a_ = None
        self.b_ = None
        
    def fit(self, x_train, y_train):
        '''训练数据集'''
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        
        self.a_ = num / d
        self.b_ = y_mean - self.a_ *x_mean
        
        return self
    
    def predict(self, x_predict):
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
    
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)
    
    def __repr__(self):
        return 'SimpleLinearRegression2()'
    
reg2 = SimpleLinearRegression1()
reg2.fit([1,2,3], [1,2,3]) 
reg2.predict([0.2])        

m = 10000
big_x = np.random.random(size=m)
big_y = big_x * 2.0 + 3.0 + np.random.normal(size=m)
#%timeit reg1.fit(big_x, big_y)
#%timeit reg2.fit(big_x, big_y)


'''衡量回归算法的标准:MSE,RMSE'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
x = boston.data[:,5]#采用房间数量这个变量
y = boston.target
plt.scatter(x, y)
plt.show()
x = x[y < 50.0]
y = y[y < 50.0]
plt.scatter(x, y)
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)
reg = SimpleLinearRegression2()
reg.fit(x_train, y_train)
reg.a_
reg.b_
plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color='r')
plt.show()

y_predict = reg.predict(x_test)

#MSE
mse_test = np.sum((y_predict - y_test)**2) / len(y_test)
mse_test
#RMSE
from math import sqrt
rmse_test = sqrt(mse_test)
rmse_test
#MAE
mae_test = np.sum(np.absolute(y_predict - y_test)) / len(y_test)
mae_test

#scikit-learn中的MSE和MAE
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
mean_squared_error(y_test, y_predict)
mean_absolute_error(y_test, y_predict)
#R Square
1 - mean_squared_error(y_test, y_predict) / np.var(y_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_predict)








