# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:54:05 2018

@author: haoyu
"""

import pkgutil as pu
import numpy as np
import matplotlib as mpl
import scipy as sp
import pydoc

'''
矩阵的逆
'''
A=np.mat('2 4 6;4 2 6;10 -4 18')
print('A\n',A)
inverse=np.linalg.inv(A)
print('inverse of A\n',inverse)
B=A*inverse
B-np.eye(3)

'''
线性方程求解
'''
A=np.mat('1 -2 1;0 2 -8;-4 5 9')
A
b=np.array([0,8,-9])
b
x=np.linalg.solve(A,b)
x
np.dot(A,x)#向量相乘

'''
特征值和特征向量
'''
A=np.mat('3 -2;1 0')
A
np.linalg.eigvals(A)
eigenvalues,eigenvectors=np.linalg.eig(A)
eigenvalues
eigenvectors
for i in range(len(eigenvalues)):
    print('left',np.dot(A,eigenvectors[:,i]))
    print('right',eigenvalues[i]*eigenvectors[:,i])
    print()

'''
随机数
'''
#二项分布  模拟赌场
from matplotlib.pyplot import plot,show

cash=np.zeros(10000)
cash[0]=1000
outcome=np.random.binomial(9,0.5,size=len(cash))

for i in range(1,len(cash)):
    if outcome[i] < 5:
        cash[i]=cash[i-1]-1
    elif outcome[i]<10:
        cash[i]=cash[i-1]+1
    else:
        raise AssertionError('Unexpected outcome'+outcome)
print(outcome.min(),outcome.max())
plot(np.arange(len(cash)),cash)
show()

#正态分布采样
import matplotlib.pyplot as plt
N=10000
normal_values=np.random.normal(size=N)
n,bins,patches=plt.hist(normal_values,int(np.sqrt(N)),normed=True,lw=1)
sigma=1
mu=0
plt.plot(bins,1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/(2*sigma**2)),lw=2)
plt.show()







    
