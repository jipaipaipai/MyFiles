# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 10:02:58 2018

@author: haoyu
"""


'''
可以认为没有模型的算法
训练数据集本身就是模型
'''

#%%time
'''
普通算法示例
'''
import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k, X_train, y_train, x):
    
    assert 1<= k <= X_train.shape[0], 'k must be valid'
    assert X_train.shape[0] == y_train.shape[0],\
        'the size of X_train must equal to the size of y_train'
    assert X_train.shape[1] == x.shape[0], \
        'the feature number of x must be equal to X_train'
    
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest= np.argsort(distances)
    
    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)
    
    return votes.most_common(1)[0][0]

predict_y = kNN_classify()


'''
scikit-learn算法
'''
from sklearn.neighbors import KNeighborsClassifier
kNN_classifier = KNeighborsClassifier(n_neighbors=6)#面向对象封装
kNN_classifier.fit(X_train, y_train)
x_predict = x.reshape(1,-1)
y_predict = kNN_classifier.predict(x_predict)

#%%time
'''
重新整理kNN普通算法
'''
class kNNclassifier:
    
    def __init__(self, k):
        '''初始化kNN分类器'''
        assert k >= 1, 'k must be balid'
        self.k = k
        self._X_train = None#防止更改
        self._y_train = None
        
    def fit(self, X_train, y_train):
        '''根据训练数据集X_train和y_train训练kNN分类器'''
        assert X_train.shape[0] == y_train.shape[0],\
            'the size of X_train must equal to the size of y_train'
        assert self.k <= X_train.shape[0], \
            'the size of X_train must be at least k.'
        
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def predict(self, X_predict):
        '''给定待预测数据集X_predict,返回表示X_predict的结果向量'''
        assert self._X_train is not None and self._y_train is not None, \
            'must fit before predict!'
        assert X_predict.shape[1] == self._X_train.shape[1], \
            'the feature number of X_predict must be equal to X_train'
            
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    
    def _predict(self, x):
        '''给定单个待预测数据x，返回x的预测结果值'''
        assert x.shape[0] == self._X_train.shape[1], \
            'the feature number of x must be equal to X_train'
        
        distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
        nearest= np.argsort(distances)
    
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
    
        return votes.most_common(1)[0][0]
    
    def score(self, X_test, y_test):
        '''根据测试数据集X_test和y_test确定当前模型的准确度'''
        y_predict = self.predict(X_test)
        return sum(y_test == y_predict) / len(y_test)
    
    def __repr__(self):
        return 'kNN(k=%d)' % self.k
    
knn_clf = kNNClassifier(k=6)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_predict)
y_predict
y_predict[0]

#%%time
'''
测试自己写的算法
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

#train_test_split
def train_test_split(X, y, test_ratio = 0.2, seed = None):
    '''将数据X和y按照test_ratio分割成X_train,X_test,y_train,y_test'''
    assert X.shape[0] == y.shape[0], \
        'the size of X must be equal to the size of y'
    assert 0.0 <= test_ratio <=1.0, \
        'test_ratio must be valid'
    
    if seed:
        np.random.seed(seed)
        
    shuffle_indexes = np.random.permutation(len(X))#打乱顺序获得索引

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]
    
    return X_train, X_test, y_train, y_test

#
X_train, X_test, y_train, y_test = train_test_split(X, y)
my_knn_clf = kNNclassifier(k=3)
my_knn_clf.fit(X_train, y_train)
y_predict = my_knn_clf.predict(X_test)
#准确度
sum(y_predict == y_test)
sum(y_predict == y_test)/len(y_test)


'''
sklearn中的train_test_split
'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%time
'''
分类准确度
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
digits.keys()

X = digits.data
y = digits.target

#将其中一个数据可视化
some_digit = X[666]
y[666]

some_digit_image = some_digit.reshape(8,8)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary)
plt.show()
#

X_train, X_test, y_train, y_test = train_test_split(X, y)
my_knn_clf = kNNclassifier(k=3)
my_knn_clf.fit(X_train, y_train)
y_predict = my_knn_clf.predict(X_test)

sum(y_predict == y_test)
sum(y_predict == y_test)/len(y_test)

###准确度函数
def accuracy_score(y_true, y_predict):
    '''计算y_true和y_predict之间的准确率'''
    assert y_true.shape[0] == y_predict.shape[0], \
        'the size of y_true must be equal to the size of y predict'
    
    return sum(y_true == y_predict) / len(y_true)

accuracy_score(y_test, y_predict)

my_knn_clf.score(X_test, y_test)
###

#sklearn总的accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)#面向对象封装
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_test)
knn_clf.score(X_test, y_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)

#%%time
'''
超参数：在算法运行前需要决定的参数
'''
import numpy as np
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
y = digits.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=666)


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)#面向对象封装
knn_clf.fit(X_train, y_train)
knn_clf.score(X_test, y_test)

#寻找最好的k和method
best_method = ''
best_score = 0.0
best_k = -1
for method in ['uniform','distance']:
    for k in range(1, 11):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method)
        knn_clf.fit(X_train,y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_method = method

print('best_k = ', best_k,'\n','best_score = ', best_score,'\n','best_method = ', best_method, sep='')


#探索明可夫斯基距离相应的p
#%%time
best_p = -1
best_score = 0.0
best_k = -1
for k in range(1, 11):
    for p in range(1,6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights='distance', p=p)
        knn_clf.fit(X_train,y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_p = p

print('best_k = ', best_k,'\n','best_score = ', best_score,'\n','best_p = ', best_p, sep='')

#%%time
'''
数据归一化
'''
'''
对于训练集和测试集不能分开归一化，测试集是模拟真实环境的，真实环境很可能无法得到所有的测试数据对的均值和方差
所以测试集的数据归一化应该用：(x_test-mean_train)/std_train
在sklearn中封装了scalar的类进行归一化
'''
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
X[:10,:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

#standardScaler
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(X_train)
standardScaler.mean_
standardScaler.scale_
standardScaler.transform( X_train)
#自编
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
#%%