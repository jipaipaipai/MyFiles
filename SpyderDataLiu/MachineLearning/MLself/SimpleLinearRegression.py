# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:33:06 2018

@author: haoyu
"""
import numpy as np
from .metrics import r2_score

class SimpleLinearRegression:
    
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
        return 'SimpleLinearRegression()'