# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:58:21 2018

@author: haoyu
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets

iris=datasets.load_iris()
iris.keys()
print(iris.DESCR)
iris.data
iris.data.shape
iris.target
iris.target_names
x=iris.data[:,:2]

plt.scatter(x[:,0],x[:,1])
plt.show()

y=iris.target
plt.scatter(x[y==0,0],x[y==0,1],color='red')
plt.scatter(x[y==1,0],x[y==1,1],color='blue')
plt.scatter(x[y==2,0],x[y==2,1],color='green')
plt.show()
