# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:20:35 2018

@author: haoyu
"""

'''数据归一化'''
import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X):
        assert X.ndim == 2, 'the dimension of X must be 2'
        
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.std_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

        return self
    
    def transform(self, X):
        resX = np.empty(shape = X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col] - self.mean_[col]) / self.scale_[col]
        return resX