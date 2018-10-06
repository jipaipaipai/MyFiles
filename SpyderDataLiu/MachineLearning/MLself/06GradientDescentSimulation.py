# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 08:40:25 2018

@author: haoyu
"""

'''梯度下降法模拟'''
import numpy as np
import matplotlib.pyplot as plt

plot_x = np.linspace(-1,6,141)
plot_y = (plot_x-2.5) ** 2 - 1
plt.plot(plot_x, plot_y)
plt.show()

def dJ(theta):
    return 2*(theta - 2.5)

def J(theta):
    return (theta-2.5)**2-1

theta = 0.0
eta = 0.1
epsilon = 1e-8
while True :
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient
    
    if(abs(J(theta) - J(last_theta)) < epsilon):
        break

print(theta)
print(J(theta))

#增加记录theta信息
theta = 0.0
theta_history = [theta]
eta = 0.1#学习率
epsilon = 1e-8

while True :
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient
    theta_history.append(theta)
    
    if(abs(J(theta) - J(last_theta)) < epsilon):
        break

plt.plot(plot_x, J(plot_x))
plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
plt.show()
len(theta_history)

#封装
def dJ(theta):
    return 2*(theta - 2.5)

def J(theta):
    return (theta-2.5)**2-1

def gradient_descent(initial_theta, eta, n_iters = 1e4, epsilon=1e-8):
    
    theta = initial_theta
    theta_history.append(initial_theta)
    i_iter = 0
    
    while i_iter < n_iters :
        gradient = dJ(theta)
        last_theta = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        
        if(abs(J(theta) - J(last_theta)) < epsilon):
            break
        
        i_iter += 1

def plot_theta_history():
    plt.plot(plot_x, J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
    plt.show()

eta = 0.01
theta_history = []
gradient_descent(0., eta)
plot_theta_history()

len(theta_history)


'''封装线性回归梯度算法'''
from MLself.LinearRegression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y < 50.0]
y = y[y < 50.0]

lin_reg = LinearRegression()
lin_reg.fit_gd(X, y)
lin_reg.coef_


'''scikit-learn中的SGD'''
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor()
sgd_reg.fit(X_train, y_train)
sgd_reg.score(X_test, y_test)
















